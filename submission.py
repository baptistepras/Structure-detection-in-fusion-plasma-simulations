"""
Codabench submission script for Tokam2D blob detection on density videos.

Overview
This submission trains a YOLO detector on a tiny labeled set, improves it with
pseudo-labeling on unlabeled frames, then learns a lightweight patch-based CNN
to remove false positives at inference time.

Current pipeline
1) Dataset conversion (YOLO format)
   - Convert labeled sequences (blob_dwi: H5 + XML) into PNG images + YOLO txt labels.
   - Convert unlabeled sequences (turb_i: H5 only) into PNG images without labels.

2) Train YOLO (round-1) on blob_dwi only
   - YOLOv8n, grayscale inputs, light data augmentation.

3) Train an MLP scorer on synthetic candidate boxes from blob_dwi
   - For each GT box, generate positives and negatives by jittering (IoU-based rules).
   - Extract hand-crafted features (geometry, intensity stats, gradients, ring contrast,
     small resized patch).
   - Standardize features and train a small MLP (BCEWithLogits + early stopping).

4) Pseudo-label turb_i using YOLO round-1 + MLP
   - Run YOLO on turb_i to get candidate boxes.
   - Score each box with the MLP.
   - Keep TOP_K boxes per image (by MLP probability) and write them as YOLO labels.

5) Train YOLO (round-2) on blob_dwi + pseudo-labeled turb_i
   - Same architecture/augmentations, trained from round-1 weights.

6) Mine blob_i to build real training data for a CNN post-filter
   - Convert blob_i to get images + GT labels.
   - Run YOLO round-2 on blob_i.
   - Build a classification dataset:
       * positives: TP predictions (IoMean >= threshold) + FN GT boxes
       * negatives: FP predictions (no GT match)
   - For each candidate box, extract:
       * a 32x32 grayscale patch
       * a small set of physics-inspired features (left/right asymmetry,
         structure-tensor orientation score, convexity proxy, aspect ratio,
         mean intensity).

7) Inference (Codabench model forward)
   - Run YOLO.predict on input frames.
   - Apply post-processing:
       * right-edge filter (remove right border artifacts)
       * drop nested container boxes
       * blackness filter (drop very dark boxes)
       * drop a known bottom-right recurring artifact
   - Apply the CNN post-filter on remaining boxes:
       * high-confidence YOLO boxes are kept (CNN veto disabled above a threshold)
       * low-confidence boxes situated on the left, top-right or bottom-right part 
         of the frame are dropped if CNN probability < threshold
   - Outputs final boxes + YOLO scores (+ optional CNN probabilities for debugging).

Notable attempts that did not improve the final score (kept out of the final code)
- Ridge-map based filtering using structure tensor coherence (soft score penalty + hard drop).
- Alternative normalizations / CLAHE / smoothing.
- Test-time augmentation (TTA) by predicting on the image and it's vertical flip, then merging boxes with NMS.
- Using the MLP as an inference-time post-filter (instead of CNN + geometric filters).
- Alternative NMS/conf heuristics beyond the current post-processing.
- Alternative color modes beyond grayscale (plasma / physics-inspired), worse than grayscale.
- Larger YOLO backbones (yolov8s/m/l, yolov11n), no gain on small data.
- Self-supervised pre-training attempts (e.g., VAE-style), no gain.
- Synthetic data generation attempts (paste/inpainting/VAE generation), degraded or no gain.
- Alternative fusions between YOLO confidence and MLP score (avg/product/meta-model), no gain.
- Alternative pseudo-label selection strategies (threshold vs top-k), top-k was best.
- Different uses of each dataset (for example using blob_i for training alongside blob_dwi).

Author: Baptiste PRAS
Contact: baptiste.pras@universite-paris-saclay.fr

Last Update: 01-Mar-2026
Python Version: python 3.10+
"""

import math
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
import torch.nn as nn
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from ultralytics import YOLO

# =========================
# GLOBALS
# =========================


def get_device() -> str:
    """Return the training device string used by PyTorch and Ultralytics."""
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_infer_device() -> str:
    """Return the inference device string used by Ultralytics YOLO.predict."""
    if torch.cuda.is_available():
        return "cuda:0"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()
RANDOM_SEED = 42
DTYPE = torch.float32

# =========================
# CONFIGS
# =========================

BLACKNESS_THR = 245.0

# CNN post-filter config
CNN_IOMEAN_MATCH_THR = 0.5
CNN_PATCH_SIZE = 32
CNN_KEEP_SCORE_THR = 0.10  # YOLO score below which the CNN can remove a YOLO prediction
CNN_PROB_THR = 0.10  # Minimum CNN probability to keep a box (boxes below are dropped)
LEFT_EDGE_MARGIN = 5  # Only apply CNN filter to boxes whose left edge is within this margin
TOP_EDGE_MARGIN = 15  # Only apply CNN filter to boxes whose top/bottom edge is within this margin...
LEFT_TOP_PADDING = 150  # ...and not in a left corner...

CONFIG_ID = "c02"

CONFIGS = {
    "c02": dict(
        cfg_id="c02_y8n_gray_07253_mosaic",
        model_ckpt="yolov8n.pt",
        color_mode="gray",
        infer=dict(conf=0.01, iou=0.50, max_det=300),
        aug=dict(
            fliplr=0.0,
            flipud=0.2,
            scale=0.2,
            translate=0.08,
            degrees=5.0,
            hsv_v=0.0,
            hsv_s=0.0,
            mosaic=0.1,
            mixup=0.05,
        ),
        yolo=dict(batch=8, epochs=1000, patience=30),
    ),
}

MLP_CONFIGS = {
    "c02": dict(
        hidden=(256,),
        dropout=0.1,
        lr=0.002,
        wd=0.0001,
        batch=128,
        epochs=200,
        patience=12,
        mlp_threshold=0.7,
        hi_neg_iou=0.35,
        lo_neg_iou=0.05,
        pos_iou=0.5,
        n=5,
    ),
}

# CNN classifier config (trained on real TP/FP/FN from blob_i)
CNN_CONFIGS = {
    "c02": dict(
        lr=0.001,
        wd=0.0001,
        batch=32,
        epochs=100,
        patience=15,
    ),
}


def get_config(config_id: str) -> dict:
    """Fetch the selected config dict and print it for debugging."""
    if config_id not in CONFIGS:
        raise ValueError(
            f"Unknown config_id='{config_id}'. Allowed: {sorted(CONFIGS.keys())}"
        )
    print(f"[INFO] Using config_id='{config_id}'")
    return CONFIGS[config_id]


def build_train_kwargs(base: dict, cfg: dict) -> dict:
    """Build Ultralytics train kwargs and merge augmentation parameters."""
    kw = dict(base)

    aug = cfg["aug"]
    kw.update(
        fliplr=aug["fliplr"],
        flipud=aug["flipud"],
        scale=aug["scale"],
        degrees=aug["degrees"],
        translate=aug["translate"],
        hsv_v=aug["hsv_v"],
        hsv_s=aug["hsv_s"],
        mosaic=aug["mosaic"],
        mixup=aug["mixup"],
    )
    return kw


# ============================================================
# YOLO WRAPPER (Codabench-compatible)
# ============================================================


class YOLOWrapper(torch.nn.Module):
    """Torch wrapper exposing YOLO inference with the exact same post-processing."""

    def __init__(
        self,
        yolo_model: YOLO,
        color_mode: str = "gray",
        conf: float = 0.01,
        iou: float = 0.55,
        max_det: int = 300,
        # CNN post-filter (optional; None = disabled)
        cnn_model: nn.Module | None = None,
    ) -> None:
        super().__init__()
        self._yolo = yolo_model
        self.network = yolo_model.model
        self.color_mode = color_mode

        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

        # CNN post-filter stored as a submodule so it moves with .to(device) calls
        self.cnn_model = cnn_model

    def _density_tensor_to_rgb_uint8(self, t: torch.Tensor) -> np.ndarray:
        """
        Convert a (C,H,W) tensor into an RGB uint8 image for YOLO inference.
        Supports 1-channel density inputs or 3-channel images.
        """
        t = t.detach().cpu()

        if t.ndim != 3:
            raise ValueError(f"Expected tensor (C,H,W), got shape={tuple(t.shape)}")

        C, _, _ = t.shape
        if C == 1:
            frame = t[0].numpy().astype(np.float32)
            gray_u8 = density_to_uint8_norm(frame)
            rgb = apply_color_mode_opencv(gray_u8, self.color_mode)
            return rgb

        if C == 3:
            img = t.permute(1, 2, 0).numpy()
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = img * 255.0
                img = np.clip(img, 0, 255).astype(np.uint8)
            return img

        raise ValueError(f"Unsupported channel count C={C}, expected 1 or 3")

    def forward(self, x: torch.Tensor) -> list[dict]:
        """
        Run YOLO inference and apply post-processing.
        Outputs CPU tensors to avoid device surprises in Codabench.
        """
        if isinstance(x, list):
            xs = x
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            xs = [x]
        else:
            xs = list(x)

        infer_device_str = get_infer_device()
        out_device = torch.device("cpu")

        inputs = [self._density_tensor_to_rgb_uint8(t) for t in xs]

        print(f"[YOLO] Running inference on {len(inputs)} frames...")
        preds = self._yolo.predict(
            inputs,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=infer_device_str,
            verbose=False,
        )

        outputs: list[dict] = []
        total_boxes = 0
        total_boxes_removed = 0
        for i, p in enumerate(preds):
            if len(p.boxes) == 0:
                outputs.append(
                    {
                        "boxes": torch.zeros((0, 4), device=out_device),
                        "scores": torch.zeros((0,), device=out_device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=out_device),
                    }
                )
                continue

            # Keep everything on CPU for consistency
            boxes = p.boxes.xyxy.detach().cpu().float()
            scores_yolo = p.boxes.conf.detach().cpu().float()
            labels = p.boxes.cls.detach().cpu().long()

            # Post-processing
            frames = boxes.shape[0]
            total = frames
            total_boxes += total
            total_removed = 0
            print(f"[CNN] Frame{i} has {frames} raw boxes before post-processing...")
            # Filter right-edge boxes (common artifact area)
            print(f"[CNN] Filtering right-edge boxes...")
            boxes, scores_yolo, labels = filter_right_edge_xyxy_torch(
                boxes, scores_yolo, labels, width=512, margin_right=10
            )
            removed = frames - boxes.shape[0]
            frames = boxes.shape[0]
            total_removed += removed
            print(f"[CNN] Removed {removed} right-edge boxes.")

            # Filter nested boxes (drop container boxes, typically bigger ones)
            print(f"[CNN] Filtering nested boxes...")
            boxes, scores_yolo, labels = drop_nested_boxes_torch(
                boxes,
                scores_yolo,
                labels,
                inside_thr=0.97,
                max_iters=10,
                mode="drop_outer",
            )
            removed = frames - boxes.shape[0]
            frames = boxes.shape[0]
            total_removed += removed
            print(f"[CNN] Removed {removed} nested boxes.")
            

            gray_u8 = tensor_to_gray_u8(xs[len(outputs)], color_mode=self.color_mode)
            b_np = boxes.detach().cpu().numpy().astype(np.float32)
            s_np = scores_yolo.detach().cpu().numpy().astype(np.float32)

            # Filter blobs with high blackness (mean intensity in box is very low, common artifact)
            print(f"[CNN] Filtering boxes with high blackness...")
            b_np, s_np = filter_blackness_np(gray_u8, b_np, s_np, thr=BLACKNESS_THR)
            removed = frames - b_np.shape[0]
            frames = b_np.shape[0]
            total_removed += removed
            print(f"[CNN] Removed {removed} blackness-filtered boxes.")

            # Filter the specific bottom-right corner artifact
            print(f"[CNN] Filtering bottom-right corner artifact...")
            b_np, s_np = drop_bottom_right_artifact_np(b_np, s_np, width=512, height=512)
            removed = frames - b_np.shape[0]
            frames = b_np.shape[0]
            total_removed += removed
            print(f"[CNN] Removed {removed} bottom-right corner artifact boxes.")

            # CNN post-filter: drop likely FPs using the patch-based CNN
            print(f"[CNN] Applying CNN post-filter on remaining boxes...")
            cnn_np = np.zeros((b_np.shape[0],), dtype=np.float32)
            if self.cnn_model is not None and b_np.shape[0] > 0:
                b_np, s_np, cnn_np = cnn_filter_boxes_np(
                    gray_u8=gray_u8,
                    boxes=b_np,
                    scores=s_np,
                    cnn_model=self.cnn_model,
                    keep_score_thr=CNN_KEEP_SCORE_THR,
                    prob_thr=CNN_PROB_THR,
                    left_edge_margin=LEFT_EDGE_MARGIN,
                    top_edge_margin=TOP_EDGE_MARGIN,
                    left_top_padding=LEFT_TOP_PADDING,
                )
            removed = frames - b_np.shape[0]
            frames = b_np.shape[0]
            total_removed += removed
            print(f"[CNN] Removed {removed} CNN-filtered boxes.")

            # Rebuild torch outputs (labels are kept for surviving boxes)
            keep_n = b_np.shape[0]
            boxes = torch.from_numpy(b_np).to(out_device).float()
            scores_yolo = torch.from_numpy(s_np).to(out_device).float()
            labels = torch.zeros((keep_n,), dtype=torch.int64, device=out_device)
            cnn_np = torch.from_numpy(cnn_np).to(out_device).float()
            
            outputs.append(
                {
                    "boxes": boxes.to(out_device),
                    "scores": scores_yolo.to(out_device),
                    "labels": labels.to(out_device),
                    "cnn_probs": cnn_np.to(out_device),
                }
            )
            total_boxes_removed += total_removed
            print(f"[CNN] Frame {i} has {boxes.shape[0]} boxes after post-processing...")
            print(f"[CNN] Total boxes removed for frame {i}: {total_removed} out of {total} total boxes.")

        print(f"[CNN] Total boxes before post-processing: {total_boxes}")
        print(f"[CNN] Total boxes removed by post-processing: {total_boxes_removed}")
        print(f"[CNN] Total boxes after post-processing: {total_boxes - total_boxes_removed}")
        return outputs

    def train(self, mode: bool = True):
        """Mirror nn.Module.train while also switching the underlying YOLO torch model."""
        self.training = mode
        self.network.train(mode)
        return self

    def eval(self):
        """Switch module to eval mode."""
        return self.train(False)


# ============================================================
# POST-PROCESSING FILTERS
# ============================================================


def tensor_to_gray_u8(t: torch.Tensor, color_mode: str = "gray") -> np.ndarray:
    """
    Convert a (C,H,W) input tensor into uint8 grayscale (H,W) used by ridgemap/blackness.
    Supports C=1 (density) and C=3 (RGB-like).
    """
    t = t.detach().cpu()
    if t.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(t.shape)}")

    C, _, _ = t.shape
    if C == 1:
        frame = t[0].numpy().astype(np.float32)
        return density_to_uint8_norm(frame)

    if C == 3:
        img = t.permute(1, 2, 0).numpy()
        if img.dtype != np.uint8:
            if img.max() <= 1.0:
                img = img * 255.0
            img = np.clip(img, 0, 255).astype(np.uint8)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        return gray

    raise ValueError(f"Unsupported channel count C={C}, expected 1 or 3")


def filter_blackness_np(gray_u8: np.ndarray, boxes: np.ndarray, scores: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop boxes whose mean blackness is above thr.
    blackness = 255 - mean_intensity_in_box
    """
    if boxes.size == 0:
        return boxes, scores

    H, W = gray_u8.shape
    keep = np.ones((boxes.shape[0],), dtype=bool)

    for i in range(boxes.shape[0]):
        x1, y1, x2, y2 = boxes[i].tolist()
        x1i = max(0, min(int(round(x1)), W))
        x2i = max(0, min(int(round(x2)), W))
        y1i = max(0, min(int(round(y1)), H))
        y2i = max(0, min(int(round(y2)), H))
        if x2i <= x1i or y2i <= y1i:
            keep[i] = False
            continue
        roi = gray_u8[y1i:y2i, x1i:x2i]
        mean_int = float(roi.mean()) if roi.size else 0.0
        blackness = 255.0 - mean_int
        if blackness > float(thr):
            keep[i] = False

    return boxes[keep], scores[keep]


def iou_xyxy_np(a: np.ndarray, b: np.ndarray) -> float:
    """IoU between two XYXY boxes (numpy arrays shape (4,))."""
    ax1, ay1, ax2, ay2 = map(float, a.tolist())
    bx1, by1, bx2, by2 = map(float, b.tolist())

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = aa + ba - inter
    return float(inter / union) if union > 1e-12 else 0.0


def drop_bottom_right_artifact_np(
    boxes: np.ndarray,
    scores: np.ndarray,
    width: int = 512,
    height: int = 512,
    iou_thr: float = 0.6,
    score_thr: float = 0.12,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Drop a very specific recurring artifact near the bottom-right corner.
    This is intentionally over-specific to avoid removing other boxes.
    """
    if boxes.size == 0:
        return boxes, scores

    # [x1, y1, x2, y2] of the bottom-right artifact
    br = np.array([410.0, 460.0, 500.0, 512.0], dtype=np.float32)

    keep = np.ones((boxes.shape[0],), dtype=bool)
    for i in range(boxes.shape[0]):
        b = boxes[i]
        s = float(scores[i])

        cx = 0.5 * (float(b[0]) + float(b[2]))
        cy = 0.5 * (float(b[1]) + float(b[3]))

        if (cx > 410.0) and (cy > 460.0) and (s < score_thr):
            if iou_xyxy_np(b, br) >= iou_thr:
                keep[i] = False

    return boxes[keep], scores[keep]


# ============================================================
# CNN POST-FILTER (physics-inspired patch classifier)
# ============================================================


def extract_cnn_features(gray_u8: np.ndarray, box_xyxy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract a (CNN_PATCH_SIZE, CNN_PATCH_SIZE) normalized patch and a physics feature
    vector for a single candidate box.

    Physics features capture blob-specific structure:
      - left/right intensity asymmetry (blobs are brighter on their left/center side)
      - dominant gradient orientation via structure tensor (blobs have horizontal gradients)
      - convexity proxy via contour analysis (blobs are convex, filaments are elongated)
      - aspect ratio (blobs are more square, filaments are elongated)
    """
    H, W = gray_u8.shape
    x1, y1, x2, y2 = box_xyxy.tolist()
    x1i = max(0, min(int(round(x1)), W))
    x2i = max(0, min(int(round(x2)), W))
    y1i = max(0, min(int(round(y1)), H))
    y2i = max(0, min(int(round(y2)), H))

    if x2i <= x1i or y2i <= y1i:
        patch_norm = np.zeros((CNN_PATCH_SIZE, CNN_PATCH_SIZE), dtype=np.float32)
        return patch_norm, np.zeros(5, dtype=np.float32)

    roi = gray_u8[y1i:y2i, x1i:x2i].astype(np.float32)

    # Resize patch to fixed CNN input size
    roi_pil = Image.fromarray(roi.astype(np.uint8))
    roi_resized = np.array(
        roi_pil.resize((CNN_PATCH_SIZE, CNN_PATCH_SIZE), resample=Image.BILINEAR),
        dtype=np.float32,
    ) / 255.0

    # Feature 1: left/right intensity asymmetry
    # Blobs are brighter on their left half; the dark interior is to the right.
    # A positive value means left is brighter (expected for a blob front).
    half = roi.shape[1] // 2
    left_mean = float(roi[:, :half].mean()) if half > 0 else 0.0
    right_mean = float(roi[:, half:].mean()) if roi.shape[1] - half > 0 else 0.0
    lr_asymmetry = (left_mean - right_mean) / 255.0  # in [-1, 1]

    # Feature 2: dominant gradient orientation
    # Compute structure tensor on the patch to find the dominant gradient direction.
    # For blobs the dominant gradient is mostly horizontal (angle near 0).
    g = roi / 255.0
    Ix = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    Jxx = float((Ix * Ix).mean())
    Jyy = float((Iy * Iy).mean())
    Jxy = float((Ix * Iy).mean())
    # Dominant eigenvector angle in [-pi/2, pi/2]; 0 = horizontal gradient
    angle = 0.5 * math.atan2(2.0 * Jxy, Jxx - Jyy + 1e-9)
    # cos(2*angle) is 1 for horizontal, -1 for vertical
    orientation_score = float(math.cos(2.0 * angle))

    # Feature 3: convexity proxy
    # Threshold the patch and compute area / convex hull area.
    # High convexity (near 1) => blob-like shape. Low => filament/irregular.
    thr_val = float(np.percentile(roi, 50))
    binary = ((roi >= thr_val).astype(np.uint8) * 255)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    convexity = 0.0
    if contours:
        cnt = max(contours, key=cv2.contourArea)
        cnt_area = float(cv2.contourArea(cnt))
        hull = cv2.convexHull(cnt)
        hull_area = float(cv2.contourArea(hull))
        if hull_area > 1.0:
            convexity = cnt_area / hull_area

    # Feature 4: aspect ratio (w/h)
    bw = float(x2i - x1i)
    bh = float(y2i - y1i)
    aspect = bw / (bh + 1e-6)

    # Feature 5: normalized mean intensity ---
    mean_intensity = float(roi.mean()) / 255.0

    phys_feat = np.array(
        [lr_asymmetry, orientation_score, convexity, aspect, mean_intensity],
        dtype=np.float32,
    )

    return roi_resized, phys_feat


class BlobCNN(nn.Module):
    """
    Lightweight CNN patch classifier for blob vs. non-blob post-filtering.

    Architecture:
      - Small conv backbone on the (CNN_PATCH_SIZE x CNN_PATCH_SIZE) patch
      - Physics feature vector fused at the FC head
      - Single binary output logit (sigmoid -> probability of being a blob)
    """

    def __init__(self, patch_size: int = CNN_PATCH_SIZE, n_phys: int = 5) -> None:
        super().__init__()

        # Conv backbone: 3 conv blocks with max-pool
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> patch_size/2

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),  # -> patch_size/4

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((4, 4)),  # -> 4x4 regardless of patch_size
        )
        conv_out_dim = 64 * 4 * 4  # 1024

        # FC head that fuses conv features + physics features
        self.fc = nn.Sequential(
            nn.Linear(conv_out_dim + n_phys, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, patches: torch.Tensor, phys: torch.Tensor) -> torch.Tensor:
        """Run the CNN on a batch of patches and physics features."""
        x = self.conv(patches)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, phys], dim=1)
        return self.fc(x).squeeze(1)


class PatchDataset(Dataset):
    """
    Torch Dataset for CNN training.
    Holds (patch, phys_feat, label) tuples extracted from image crops.
    """

    def __init__(
        self,
        patches: np.ndarray,  # (N, H, W) float32 in [0,1]
        phys: np.ndarray,     # (N, n_phys) float32
        labels: np.ndarray,   # (N,) int64
    ) -> None:
        self.patches = torch.from_numpy(patches).unsqueeze(1).float()  # (N,1,H,W)
        self.phys = torch.from_numpy(phys).float()
        self.labels = torch.from_numpy(labels).long()

    def __len__(self) -> int:
        return self.patches.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.patches[idx], self.phys[idx], self.labels[idx]


def iomean_xyxy_np(a: np.ndarray, b: np.ndarray) -> float:
    """
    IoMean between two XYXY boxes (numpy arrays shape (4,)).
    IoMean = 2 * inter / (area_a + area_b), matching the challenge metric.
    """
    ax1, ay1, ax2, ay2 = map(float, a.tolist())
    bx1, by1, bx2, by2 = map(float, b.tolist())

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    denom = aa + ba
    return float(2.0 * inter / denom) if denom > 1e-12 else 0.0


def mine_blob_i_for_cnn(
    yolo_model: YOLO,
    blob_i_images_dir: Path,
    blob_i_labels_dir: Path,
    infer_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run the trained YOLO model on blob_i (which has GT annotations) and collect
    patch samples for CNN training:

      - Positives (label=1):
          * TP: predicted boxes that match a GT (IoMean >= CNN_IOMEAN_MATCH_THR)
          * FN: GT boxes not matched by any prediction
      - Negatives (label=0):
          * FP: predicted boxes with no GT match
    """
    # Only process blob_i frames (stem starts with "blob_i_")
    all_pairs = list_image_label_pairs(blob_i_images_dir, blob_i_labels_dir)
    blob_i_pairs = [(ip, lp) for ip, lp in all_pairs if ip.stem.startswith("blob_i_")]

    if not blob_i_pairs:
        print("[CNN] No blob_i frames found — CNN mining skipped.")
        return (
            np.zeros((0, CNN_PATCH_SIZE, CNN_PATCH_SIZE), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    # Run YOLO on all blob_i images at once
    img_paths = [ip for ip, _ in blob_i_pairs]
    print(f"[CNN] Running YOLO on {len(img_paths)} blob_i images for mining...")
    raw_boxes_dict = yolo_predict_xyxy_on_images(
        model=yolo_model,
        image_paths=img_paths,
        conf=infer_cfg["conf"],
        iou=infer_cfg["iou"],
        max_det=infer_cfg["max_det"],
    )

    all_patches: list[np.ndarray] = []
    all_phys: list[np.ndarray] = []
    all_labels: list[int] = []

    n_tp = n_fn = n_fp = 0

    for img_path, lab_path in blob_i_pairs:
        gray = read_gray_u8(img_path)
        H, W = gray.shape

        # Load GT boxes
        gt_labels = load_yolo_txt(lab_path)
        gt_boxes_xyxy = np.array(
            [
                [
                    (xc - bw / 2) * W,
                    (yc - bh / 2) * H,
                    (xc + bw / 2) * W,
                    (yc + bh / 2) * H,
                ]
                for _, xc, yc, bw, bh in gt_labels
            ],
            dtype=np.float32,
        )  # shape (G, 4)

        pred_boxes_xyxy = raw_boxes_dict.get(img_path.stem, np.zeros((0, 4), dtype=np.float32))

        # Greedy matching: each GT matched at most once, each pred matched at most once
        gt_matched = np.zeros(len(gt_boxes_xyxy), dtype=bool)
        pred_matched = np.zeros(pred_boxes_xyxy.shape[0], dtype=bool)

        for pi in range(pred_boxes_xyxy.shape[0]):
            best_score = -1.0
            best_gi = -1
            for gi in range(len(gt_boxes_xyxy)):
                if gt_matched[gi]:
                    continue
                score = iomean_xyxy_np(pred_boxes_xyxy[pi], gt_boxes_xyxy[gi])
                if score > best_score:
                    best_score = score
                    best_gi = gi
            if best_gi >= 0 and best_score >= CNN_IOMEAN_MATCH_THR:
                gt_matched[best_gi] = True
                pred_matched[pi] = True

        # TP predictions -> positive samples
        for pi in range(pred_boxes_xyxy.shape[0]):
            if pred_matched[pi]:
                patch, phys = extract_cnn_features(gray, pred_boxes_xyxy[pi])
                all_patches.append(patch)
                all_phys.append(phys)
                all_labels.append(1)
                n_tp += 1

        # FP predictions -> negative samples
        for pi in range(pred_boxes_xyxy.shape[0]):
            if not pred_matched[pi]:
                patch, phys = extract_cnn_features(gray, pred_boxes_xyxy[pi])
                all_patches.append(patch)
                all_phys.append(phys)
                all_labels.append(0)
                n_fp += 1

        # FN (unmatched GTs) -> positive samples (actual blobs the model missed)
        for gi in range(len(gt_boxes_xyxy)):
            if not gt_matched[gi]:
                patch, phys = extract_cnn_features(gray, gt_boxes_xyxy[gi])
                all_patches.append(patch)
                all_phys.append(phys)
                all_labels.append(1)
                n_fn += 1

    print(f"[CNN-MINE] blob_i mining: TP={n_tp}, FN={n_fn}, FP={n_fp}")
    print(f"[CNN-MINE] Total samples: pos={n_tp + n_fn}, neg={n_fp}")

    if not all_patches:
        return (
            np.zeros((0, CNN_PATCH_SIZE, CNN_PATCH_SIZE), dtype=np.float32),
            np.zeros((0, 5), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )

    return (
        np.stack(all_patches).astype(np.float32),
        np.stack(all_phys).astype(np.float32),
        np.array(all_labels, dtype=np.int64),
    )


def train_blob_cnn(
    patches: np.ndarray,
    phys: np.ndarray,
    labels: np.ndarray,
    cnn_cfg: dict,
) -> BlobCNN:
    """
    Train the BlobCNN post-filter on mined TP/FN/FP patches from blob_i.

    Uses BCEWithLogitsLoss with class-balanced weights and early stopping
    on training loss (same pattern as train_mlp).
    """
    n_pos = int(labels.sum())
    n_neg = int((1 - labels).sum())
    n_total = len(labels)

    if n_total == 0 or n_pos == 0 or n_neg == 0:
        print("[CNN-TRAIN] Not enough samples to train CNN (need pos and neg). Skipping.")
        model = BlobCNN().to(DEVICE)
        model.eval()
        return model

    print(f"[CNN-TRAIN] Training BlobCNN: total={n_total} pos={n_pos} neg={n_neg}")

    # Class-balanced loss weight: upweight minority class
    pos_weight = torch.tensor([n_neg / (n_pos + 1e-6)], dtype=torch.float32).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    model = BlobCNN(patch_size=CNN_PATCH_SIZE, n_phys=phys.shape[1]).to(DEVICE)
    optim = torch.optim.AdamW(model.parameters(), lr=cnn_cfg["lr"], weight_decay=cnn_cfg["wd"])

    dataset = PatchDataset(patches, phys, labels)
    loader = DataLoader(dataset, batch_size=cnn_cfg["batch"], shuffle=True, num_workers=0)

    best_state: dict | None = None
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(cnn_cfg["epochs"]):
        model.train()
        total_loss = 0.0
        n = 0

        for patch_b, phys_b, label_b in loader:
            patch_b = patch_b.to(DEVICE)
            phys_b = phys_b.to(DEVICE)
            label_b = label_b.float().to(DEVICE)

            optim.zero_grad(set_to_none=True)
            logits = model(patch_b, phys_b)
            loss = loss_fn(logits, label_b)
            loss.backward()
            optim.step()

            bs = patch_b.size(0)
            total_loss += loss.item() * bs
            n += bs

        avg_loss = total_loss / max(1, n)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cnn_cfg["patience"]:
            print(f"[CNN-TRAIN] Early stopping at epoch {epoch + 1}.")
            break

        print(
            f"[CNN-TRAIN] epoch {epoch + 1}/{cnn_cfg['epochs']}  "
            f"loss={avg_loss:.4f}  best={best_loss:.4f}  bad={bad_epochs}/{cnn_cfg['patience']}"
        )

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return model


def cnn_filter_boxes_np(
    gray_u8: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    cnn_model: nn.Module,
    keep_score_thr: float = CNN_KEEP_SCORE_THR,
    prob_thr: float = CNN_PROB_THR,
    left_edge_margin: int = LEFT_EDGE_MARGIN,
    top_edge_margin: int = TOP_EDGE_MARGIN,
    left_top_padding: int = LEFT_TOP_PADDING,
 ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """
    Apply the trained BlobCNN to filter likely FP boxes at inference time.
    Only filter in the leftmost part of the frame or near the top and
    bottom rightmost part of the frame.

    High-confidence YOLO boxes (score >= keep_score_thr) are never dropped
    by the CNN, to protect against accidentally removing real blobs.
    """
    if boxes.shape[0] == 0:
        return boxes, scores

    all_patches: list[np.ndarray] = []
    all_phys: list[np.ndarray] = []

    for i in range(boxes.shape[0]):
        patch, phys = extract_cnn_features(gray_u8, boxes[i])
        all_patches.append(patch)
        all_phys.append(phys)

    patch_tensor = torch.from_numpy(
        np.stack(all_patches).astype(np.float32)
    ).unsqueeze(1).to(DEVICE)  # (N,1,H,W)
    phys_tensor = torch.from_numpy(
        np.stack(all_phys).astype(np.float32)
    ).to(DEVICE)  # (N,5)

    with torch.no_grad():
        logits = cnn_model(patch_tensor, phys_tensor)
        probs = torch.sigmoid(logits).detach().cpu().numpy()  # (N,)

    H, W = gray_u8.shape
    keep = np.ones(boxes.shape[0], dtype=bool)
    for i in range(boxes.shape[0]):
        x1 = float(boxes[i, 0])
        y1 = float(boxes[i, 1])
        y2 = float(boxes[i, 3])

        # Never remove high-confidence YOLO predictions
        if float(scores[i]) >= keep_score_thr:
            continue

        # Determine if the box is in a filterable zone
        in_left_strip = (x1 < left_edge_margin)  # left vertical strip
        in_top_strip = (y1 < top_edge_margin) and (x1 >= left_top_padding)  # top horizontal strip (excluding left corner)
        in_bottom_strip = (y2 > H - top_edge_margin) and (x1 >= left_top_padding)  # bottom horizontal strip (excluding left corner)

        if not (in_left_strip or in_top_strip or in_bottom_strip):
            continue

        # Drop boxes with low CNN blob probability
        if float(probs[i]) < prob_thr:
            keep[i] = False

    return boxes[keep], scores[keep], probs[keep]


# ============================================================
# DATA CONVERSION (H5 + XML -> YOLO PNG + TXT)
# ============================================================


def density_to_uint8_norm(frame: np.ndarray) -> np.ndarray:
    """
    Robustly normalize a density frame (H,W) float -> uint8 (H,W) in [0,255].
    Uses percentile clipping (1%, 99%) for contrast robustness.
    """
    frame = frame.astype(np.float32)
    p1, p99 = np.percentile(frame, [1, 99])
    if p99 > p1:
        frame = np.clip(frame, p1, p99)
        frame = (frame - p1) / (p99 - p1)
    else:
        frame = np.zeros_like(frame)
    return (frame * 255.0).astype(np.uint8)


def apply_color_mode_opencv(gray_u8: np.ndarray, color_mode: str) -> np.ndarray:
    """Convert a uint8 grayscale density image into a 3-channel RGB image."""
    if color_mode == "gray":
        return np.stack([gray_u8, gray_u8, gray_u8], axis=-1)
    raise ValueError(f"Unknown color_mode='{color_mode}'. Use: gray")


def density_to_rgb(frame: np.ndarray, color_mode: str = "gray") -> np.ndarray:
    """Convert a density frame to an RGB uint8 image using the selected color mode."""
    gray_u8 = density_to_uint8_norm(frame)
    return apply_color_mode_opencv(gray_u8, color_mode=color_mode)


def load_h5_video_and_indices(h5_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a video array (T,H,W) from a .h5 file and return (video, indices).
    Prefers a dataset named 'density' when available.
    """
    with h5py.File(h5_path, "r") as f:
        if "density" in f:
            video = np.array(f["density"])
        else:
            video = None
            for k in f.keys():
                arr = np.array(f[k])
                if arr.ndim == 3:
                    video = arr
                    break
            if video is None:
                raise ValueError(f"No 3D dataset found in {h5_path}. Keys={list(f.keys())}")

        if "indices" in f:
            indices = np.array(f["indices"]).astype(int)
        else:
            offset = int(f.attrs.get("offset", 0))
            stride = int(f.attrs.get("stride", 1))
            indices = offset + stride * np.arange(video.shape[0], dtype=int)

    if video.shape[0] != indices.shape[0]:
        raise ValueError(f"{h5_path}: video T={video.shape[0]} != indices len={indices.shape[0]}")
    return video, indices


def convert_datasets(
    training_path: str | Path,
    yolo_root: str | Path,
    color_mode: str = "gray",
    blobi: bool = False,
) -> None:
    """
    Convert challenge files into a YOLO-like folder with PNG images and TXT labels.
    Also exports unlabeled frames from H5-only files.
    """
    training_path = Path(training_path)
    yolo_root = Path(yolo_root)

    img_dir_train = yolo_root / "images/train"
    lab_dir_train = yolo_root / "labels/train"
    img_dir_unlab = yolo_root / "images/unlabeled"

    img_dir_train.mkdir(parents=True, exist_ok=True)
    lab_dir_train.mkdir(parents=True, exist_ok=True)
    img_dir_unlab.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(training_path.glob("*.xml"))
    h5_files = sorted(training_path.glob("*.h5"))

    xml_by_stem = {p.stem: p for p in xml_files}
    h5_by_stem = {p.stem: p for p in h5_files}

    def allowed(stem: str, blobi_flag: bool) -> bool:
        """Keep the exact original gating for blob_i inclusion."""
        if stem == "blob_i":
            return blobi_flag
        return True

    annotated_stems = sorted(
        s
        for s in (set(xml_by_stem.keys()) & set(h5_by_stem.keys()))
        if allowed(s, blobi_flag=blobi)
    )
    unlabeled_stems = sorted(
        s
        for s in (set(h5_by_stem.keys()) - set(xml_by_stem.keys()))
        if allowed(s, blobi_flag=blobi)
    )

    n_annot_frames = 0
    n_unlab_frames = 0
    n_annot_boxes = 0

    # Export labeled frames (XML + H5)
    for stem in annotated_stems:
        xml_path = xml_by_stem[stem]
        h5_path = h5_by_stem[stem]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        video, indices = load_h5_video_and_indices(h5_path)
        T = video.shape[0]
        idx_map = {int(t): i for i, t in enumerate(indices.tolist())}
        skipped = 0

        for img_tag in root.findall("image"):
            name_noext = img_tag.get("name").replace(".png", "")
            t = int(name_noext)

            if t not in idx_map:
                skipped += 1
                continue

            frame_id = idx_map[t]
            frame = video[frame_id]
            frame_rgb = density_to_rgb(frame, color_mode=color_mode)

            out_name = f"{stem}_{t}"
            Image.fromarray(frame_rgb).save(img_dir_train / f"{out_name}.png")

            w = float(img_tag.get("width"))
            h = float(img_tag.get("height"))

            boxes = img_tag.findall("box")
            with open(lab_dir_train / f"{out_name}.txt", "w") as f:
                for box in boxes:
                    xtl = float(box.get("xtl"))
                    ytl = float(box.get("ytl"))
                    xbr = float(box.get("xbr"))
                    ybr = float(box.get("ybr"))
                    xc = (xtl + xbr) / 2 / w
                    yc = (ytl + ybr) / 2 / h
                    bw = (xbr - xtl) / w
                    bh = (ybr - ytl) / h
                    f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

            n_annot_frames += 1
            n_annot_boxes += len(boxes)

        if skipped > 0:
            print(f"[WARN] {stem}: skipped {skipped} XML images because t not in H5 indices (T={T})")

    # Export unlabeled frames (H5 only)
    for stem in unlabeled_stems:
        h5_path = h5_by_stem[stem]
        video, indices = load_h5_video_and_indices(h5_path)

        for i in range(video.shape[0]):
            t = int(indices[i])
            frame_rgb = density_to_rgb(video[i], color_mode=color_mode)
            out_name = f"{stem}_{t}"
            Image.fromarray(frame_rgb).save(img_dir_unlab / f"{out_name}.png")
            n_unlab_frames += 1

    print("\n=== Dataset conversion summary ===")
    print(f"Annotated files (xml+h5): {len(annotated_stems)} -> {annotated_stems}")
    print(f"Unlabeled files (h5 only): {len(unlabeled_stems)} -> {unlabeled_stems}")
    print(f"Annotated frames exported: {n_annot_frames}")
    print(f"Annotated boxes exported:  {n_annot_boxes}")
    print(f"Unlabeled frames exported: {n_unlab_frames}")
    print("================================\n")


# ============================================================
# PSEUDO-LABELING UTILITIES
# ============================================================


def list_unlabeled_images(images_unlabeled_dir: str | Path) -> list[Path]:
    """List unlabeled images (PNG/JPG/...) for pseudo-label inference."""
    images_unlabeled_dir = Path(images_unlabeled_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(images_unlabeled_dir.iterdir()) if p.suffix.lower() in exts]


def yolo_predict_xyxy_on_images(
    model: YOLO,
    image_paths: list[Path],
    conf: float,
    iou: float,
    max_det: int,
) -> dict[str, np.ndarray]:
    """Run YOLO prediction on a list of image paths."""
    if not image_paths:
        return {}

    infer_device_str = get_infer_device()
    preds = model.predict(
        [str(p) for p in image_paths],
        conf=float(conf),
        iou=float(iou),
        max_det=int(max_det),
        device=infer_device_str,
        verbose=False,
    )

    out: dict[str, np.ndarray] = {}
    for path, p in zip(image_paths, preds):
        if len(p.boxes) == 0:
            out[path.stem] = np.zeros((0, 4), dtype=np.float32)
        else:
            out[path.stem] = p.boxes.xyxy.detach().cpu().numpy().astype(np.float32)
    return out


def xyxy_to_yolo_norm(
    xyxy: np.ndarray, W: int, H: int
) -> list[tuple[float, float, float, float]]:
    """Convert XYXY absolute pixels -> YOLO normalized (xc,yc,w,h) in [0,1]."""
    out = []
    for x1, y1, x2, y2 in xyxy:
        x1 = float(np.clip(x1, 0, W - 1))
        y1 = float(np.clip(y1, 0, H - 1))
        x2 = float(np.clip(x2, 0, W - 1))
        y2 = float(np.clip(y2, 0, H - 1))
        if x2 <= x1 or y2 <= y1:
            continue

        xc = 0.5 * (x1 + x2) / W
        yc = 0.5 * (y1 + y2) / H
        bw = (x2 - x1) / W
        bh = (y2 - y1) / H

        xc = float(np.clip(xc, 0.0, 1.0))
        yc = float(np.clip(yc, 0.0, 1.0))
        bw = float(np.clip(bw, 0.0, 1.0))
        bh = float(np.clip(bh, 0.0, 1.0))
        out.append((xc, yc, bw, bh))
    return out


def write_pseudo_labels_yolo(
    images_unlabeled_dir: str | Path,
    labels_unlabeled_dir: str | Path,
    boxes_by_stem_xyxy: dict[str, np.ndarray],
) -> int:
    """Write pseudo YOLO labels into labels_unlabeled_dir for images/unlabeled."""
    images_unlabeled_dir = Path(images_unlabeled_dir)
    labels_unlabeled_dir = Path(labels_unlabeled_dir)
    labels_unlabeled_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for stem, xyxy in boxes_by_stem_xyxy.items():
        img_candidates = list(images_unlabeled_dir.glob(stem + ".*"))
        if not img_candidates:
            continue
        img_path = img_candidates[0]

        im = Image.open(img_path)
        W, H = im.size
        yolo_boxes = xyxy_to_yolo_norm(xyxy, W=W, H=H)

        lab_path = labels_unlabeled_dir / f"{stem}.txt"
        lab_path.touch(exist_ok=True)
        with open(lab_path, "w") as f:
            for (xc, yc, bw, bh) in yolo_boxes:
                f.write(f"0 {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}\n")

        total += len(yolo_boxes)

    return total


def make_combined_yolo_yaml(
    yolo_root: str | Path,
    out_yaml_path: str | Path,
) -> None:
    """
    Create a YOLO data.yaml where:
      train = [images/train, images/unlabeled]
      val   = images/train
    and labels must exist for both (labels/train and labels/unlabeled).
    """
    yolo_root = Path(yolo_root)
    out_yaml_path = Path(out_yaml_path)

    data_cfg = {
        "path": str(yolo_root.resolve()),
        "train": ["images/train", "images/unlabeled"],
        "val": "images/train",
        "nc": 1,
        "names": ["blob"],
    }
    with open(out_yaml_path, "w") as f:
        yaml.dump(data_cfg, f)


def filter_right_edge_xyxy_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    width: int = 512,
    margin_right: int = 20,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """Keep only boxes whose right side x2 is strictly before width - margin_right."""
    if boxes.numel() == 0:
        return boxes, scores, labels

    keep = boxes[:, 2] < float(width - margin_right)

    boxes = boxes[keep]
    if scores is not None:
        scores = scores[keep]
    if labels is not None:
        labels = labels[keep]
    return boxes, scores, labels


def drop_nested_boxes_torch(
    boxes: torch.Tensor,
    scores: torch.Tensor | None = None,
    labels: torch.Tensor | None = None,
    inside_thr: float = 0.95,
    max_iters: int = 10,
    mode: str = "drop_outer",
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    """
    Drop nested boxes iteratively.

    Definition of nesting:
      i inside j if inter(i,j) / area(i) >= inside_thr

    mode:
      - "drop_outer": drop the container (typically bigger) box j
      - "drop_inner": drop the contained (typically smaller) box i
    """
    if boxes.numel() == 0 or boxes.shape[0] <= 1:
        return boxes, scores, labels

    boxes = boxes.float()
    n_iter = 0

    while n_iter < max_iters and boxes.shape[0] > 1:
        n_iter += 1

        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        w = (x2 - x1).clamp(min=0.0)
        h = (y2 - y1).clamp(min=0.0)
        area = (w * h).clamp(min=1e-12)

        xx1 = torch.maximum(x1[:, None], x1[None, :])
        yy1 = torch.maximum(y1[:, None], y1[None, :])
        xx2 = torch.minimum(x2[:, None], x2[None, :])
        yy2 = torch.minimum(y2[:, None], y2[None, :])
        iw = (xx2 - xx1).clamp(min=0.0)
        ih = (yy2 - yy1).clamp(min=0.0)
        inter = iw * ih

        inside_ratio = inter / area[:, None]
        inside = inside_ratio >= inside_thr
        inside.fill_diagonal_(False)

        if not bool(inside.any().item()):
            break

        if mode == "drop_outer":
            to_drop = inside.any(dim=0)
        elif mode == "drop_inner":
            to_drop = inside.any(dim=1)
        else:
            raise ValueError(f"Unknown mode='{mode}'. Use 'drop_outer' or 'drop_inner'.")

        if not bool(to_drop.any().item()):
            break

        keep = ~to_drop
        boxes = boxes[keep]
        if scores is not None:
            scores = scores[keep]
        if labels is not None:
            labels = labels[keep]

    return boxes, scores, labels


# ============================================================
# MLP TRAINING UTILITIES
# ============================================================


@dataclass(frozen=True)
class BoxXYXY:
    """Simple XYXY box with helper methods for geometry and clipping."""

    x1: float
    y1: float
    x2: float
    y2: float

    def clip(self, W: int, H: int) -> "BoxXYXY":
        """Clip box coordinates to image bounds and fix inverted corners."""
        x1 = float(np.clip(self.x1, 0, W - 1))
        y1 = float(np.clip(self.y1, 0, H - 1))
        x2 = float(np.clip(self.x2, 0, W - 1))
        y2 = float(np.clip(self.y2, 0, H - 1))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        return BoxXYXY(x1, y1, x2, y2)

    def w(self) -> float:
        """Return box width."""
        return max(0.0, self.x2 - self.x1)

    def h(self) -> float:
        """Return box height."""
        return max(0.0, self.y2 - self.y1)

    def area(self) -> float:
        """Return box area."""
        return self.w() * self.h()

    def center(self) -> tuple[float, float]:
        """Return box center coordinates."""
        return (0.5 * (self.x1 + self.x2), 0.5 * (self.y1 + self.y2))


def yolo_norm_to_xyxy(xc: float, yc: float, bw: float, bh: float, W: int, H: int) -> BoxXYXY:
    """Convert a YOLO-normalized box to absolute XYXY coordinates and clip to image bounds."""
    x_c = xc * W
    y_c = yc * H
    w = bw * W
    h = bh * H
    x1 = x_c - 0.5 * w
    y1 = y_c - 0.5 * h
    x2 = x_c + 0.5 * w
    y2 = y_c + 0.5 * h
    return BoxXYXY(x1, y1, x2, y2).clip(W, H)


def iou(a: BoxXYXY, b: BoxXYXY) -> float:
    """Compute Intersection-over-Union between two boxes."""
    ix1 = max(a.x1, b.x1)
    iy1 = max(a.y1, b.y1)
    ix2 = min(a.x2, b.x2)
    iy2 = min(a.y2, b.y2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a.area() + b.area() - inter
    if union <= 1e-12:
        return 0.0
    return float(inter / union)


def max_iou_with_any(candidate: BoxXYXY, gts: list[BoxXYXY]) -> float:
    """Return the maximum IoU between a candidate and any GT box."""
    if not gts:
        return 0.0
    return max(iou(candidate, gt) for gt in gts)


def load_yolo_txt(txt_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """Load YOLO labels from a .txt file as (class, xc, yc, w, h)."""
    txt_path = Path(txt_path)
    out: list[tuple[int, float, float, float, float]] = []
    if not txt_path.exists():
        return out
    for line in txt_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls = int(float(parts[0]))
        xc, yc, bw, bh = map(float, parts[1:5])
        out.append((cls, xc, yc, bw, bh))
    return out


def list_image_label_pairs(images_dir: str | Path, labels_dir: str | Path) -> list[tuple[Path, Path]]:
    """List (image_path, label_path) pairs where the corresponding YOLO label exists."""
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    images = [p for p in sorted(images_dir.iterdir()) if p.suffix.lower() in exts]
    pairs: list[tuple[Path, Path]] = []
    for img_path in images:
        lab_path = labels_dir / f"{img_path.stem}.txt"
        if lab_path.exists():
            pairs.append((img_path, lab_path))
    return pairs


def read_gray_u8(img_path: str | Path) -> np.ndarray:
    """Read an image and return a uint8 grayscale array (H,W)."""
    im = Image.open(img_path).convert("L")
    return np.array(im, dtype=np.uint8)


def jitter_translate_box(
    gt: BoxXYXY,
    W: int,
    H: int,
    translate_max_frac: float,
    scale_jitter: float,
    rng: random.Random,
) -> BoxXYXY:
    """
    Create a jittered version of a GT box by translating and scaling it.
    Used to synthesize candidate boxes around a target.
    """
    cx, cy = gt.center()
    w = gt.w()
    h = gt.h()

    def sample_shift_frac() -> float:
        """Sample a random shift fraction with a bias towards smaller shifts."""
        u = rng.random()
        if u < 0.70:
            return rng.uniform(-0.20, 0.20)
        return rng.uniform(-1.00, 1.00)

    dx_frac = sample_shift_frac()
    dy_frac = sample_shift_frac()

    dx = dx_frac * w * translate_max_frac * 4.0
    dy = dy_frac * h * translate_max_frac * 4.0

    sw = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    sh = 1.0 + rng.uniform(-scale_jitter, scale_jitter)

    nw = max(2.0, w * sw)
    nh = max(2.0, h * sh)

    ncx = cx + dx
    ncy = cy + dy

    x1 = ncx - 0.5 * nw
    y1 = ncy - 0.5 * nh
    x2 = ncx + 0.5 * nw
    y2 = ncy + 0.5 * nh

    return BoxXYXY(x1, y1, x2, y2).clip(W, H)


def synthesize_pos_neg_for_gt(
    gt: BoxXYXY,
    all_gts: list[BoxXYXY],
    W: int,
    H: int,
    rng: random.Random,
    pos_iou: float,
    neg_low: float,
    neg_high: float,
    n_pos: int,
    n_neg: int,
    max_tries: int,
) -> tuple[list[BoxXYXY], list[BoxXYXY]]:
    """
    Synthesize positive and negative boxes around a GT using IoU thresholds.
    Negatives are constrained to avoid trivial background samples.
    """
    pos: list[BoxXYXY] = []
    neg: list[BoxXYXY] = []
    seen: set[tuple[float, float, float, float]] = set()

    cur_neg_low = neg_low
    cur_neg_high = neg_high

    for t in range(max_tries):
        cand = jitter_translate_box(
            gt,
            W,
            H,
            translate_max_frac=0.25,
            scale_jitter=0.05,
            rng=rng,
        )

        key = (
            round(cand.x1, 1),
            round(cand.y1, 1),
            round(cand.x2, 1),
            round(cand.y2, 1),
        )
        if key in seen:
            continue
        seen.add(key)

        miou = max_iou_with_any(cand, all_gts)

        if miou >= pos_iou:
            if len(pos) < n_pos:
                pos.append(cand)
        elif cur_neg_low <= miou <= cur_neg_high:
            if len(neg) < n_neg:
                neg.append(cand)

        if len(pos) == n_pos and len(neg) == n_neg:
            return pos, neg

        if (t + 1) % 800 == 0:
            cur_neg_high = min(0.49, cur_neg_high + 0.05)
            cur_neg_low = max(0.0, cur_neg_low - 0.02)

    return pos, neg


def sobel_gradients(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute simple Sobel gradients in x and y (manual implementation)."""
    p2 = np.pad(p, ((1, 1), (1, 1)), mode="edge")

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    H, W = p.shape
    gx = np.zeros((H, W), dtype=np.float32)
    gy = np.zeros((H, W), dtype=np.float32)

    for y in range(H):
        for x in range(W):
            patch = p2[y : y + 3, x : x + 3]
            gx[y, x] = float(np.sum(patch * kx))
            gy[y, x] = float(np.sum(patch * ky))

    return gx, gy


def laplacian(p: np.ndarray) -> np.ndarray:
    """Compute a simple Laplacian response (manual implementation)."""
    p2 = np.pad(p, ((1, 1), (1, 1)), mode="edge")
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
    H, W = p.shape
    out = np.zeros((H, W), dtype=np.float32)
    for y in range(H):
        for x in range(W):
            patch = p2[y : y + 3, x : x + 3]
            out[y, x] = float(np.sum(patch * k))
    return out


def entropy_hist(x: np.ndarray, bins: int = 32) -> float:
    """Compute a histogram-based entropy estimate for an array."""
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0, 255).astype(np.uint8).reshape(-1)
    idx = (x.astype(np.int32) * bins) // 256
    hist = np.bincount(idx, minlength=bins).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def crop_patch(gray: np.ndarray, b: BoxXYXY) -> np.ndarray:
    """Crop a grayscale patch corresponding to a box."""
    x1 = int(round(b.x1))
    y1 = int(round(b.y1))
    x2 = int(round(b.x2))
    y2 = int(round(b.y2))

    x1 = max(0, min(x1, gray.shape[1] - 1))
    y1 = max(0, min(y1, gray.shape[0] - 1))
    x2 = max(0, min(x2, gray.shape[1]))
    y2 = max(0, min(y2, gray.shape[0]))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1), dtype=np.uint8)
    return gray[y1:y2, x1:x2]


def crop_ring(gray: np.ndarray, b: BoxXYXY, factor: float = 1.6) -> np.ndarray:
    """
    Extract a ring region around a box (outer crop minus inner box).
    This provides a simple local background estimate.
    """
    H, W = gray.shape
    cx, cy = b.center()
    nw = b.w() * factor
    nh = b.h() * factor

    xb1 = int(round(cx - 0.5 * nw))
    yb1 = int(round(cy - 0.5 * nh))
    xb2 = int(round(cx + 0.5 * nw))
    yb2 = int(round(cy + 0.5 * nh))

    xb1 = max(0, min(xb1, W - 1))
    yb1 = max(0, min(yb1, H - 1))
    xb2 = max(0, min(xb2, W))
    yb2 = max(0, min(yb2, H))

    outer = gray[yb1:yb2, xb1:xb2]
    if outer.size == 0:
        return np.zeros((0,), dtype=np.uint8)

    x1 = int(round(b.x1)) - xb1
    y1 = int(round(b.y1)) - yb1
    x2 = int(round(b.x2)) - xb1
    y2 = int(round(b.y2)) - yb1

    x1 = max(0, min(x1, outer.shape[1]))
    y1 = max(0, min(y1, outer.shape[0]))
    x2 = max(0, min(x2, outer.shape[1]))
    y2 = max(0, min(y2, outer.shape[0]))

    mask = np.ones(outer.shape, dtype=bool)
    if x2 > x1 and y2 > y1:
        mask[y1:y2, x1:x2] = False

    return outer[mask].astype(np.uint8)


def extract_features(gray: np.ndarray, b: BoxXYXY) -> np.ndarray:
    """
    Extract a fixed feature vector for a candidate box.
    Mixes geometry, intensity stats, gradients, and a small resized patch.
    """
    H, W = gray.shape
    bw = max(1.0, b.w())
    bh = max(1.0, b.h())
    cx, cy = b.center()

    patch = crop_patch(gray, b).astype(np.float32)
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.float32)

    mean = float(patch.mean())
    std = float(patch.std())
    vmin = float(patch.min())
    vmax = float(patch.max())
    p10 = float(np.percentile(patch, 10))
    p50 = float(np.percentile(patch, 50))
    p90 = float(np.percentile(patch, 90))
    ent = float(entropy_hist(patch.astype(np.uint8), bins=32))

    p_norm = patch / 255.0
    gx, gy = sobel_gradients(p_norm)
    gmag = np.sqrt(gx * gx + gy * gy)
    gmean = float(gmag.mean())
    gstd = float(gmag.std())
    gp90 = float(np.percentile(gmag, 90))

    lap = laplacian(p_norm)
    lap_abs = np.abs(lap)
    lap_var = float(lap_abs.var())
    edge_density = float((gmag > np.percentile(gmag, 80)).mean())

    ring = crop_ring(gray, b, factor=1.6).astype(np.float32)
    if ring.size > 0:
        ring_mean = float(ring.mean())
        ring_std = float(ring.std())
    else:
        ring_mean = 0.0
        ring_std = 0.0

    contrast_mean = mean - ring_mean
    contrast_std = std - ring_std

    pil_patch = Image.fromarray(patch.astype(np.uint8))
    small = (
        np.array(pil_patch.resize((24, 24), resample=Image.BILINEAR), dtype=np.float32)
        / 255.0
    )
    small_flat = small.reshape(-1)

    geo = np.array(
        [
            cx / W,
            cy / H,
            bw / W,
            bh / H,
            (bw * bh) / (W * H),
            bw / (bh + 1e-6),
            (bw + bh) / (W + H),
            math.log(bw + 1.0),
            math.log(bh + 1.0),
        ],
        dtype=np.float32,
    )

    stats = np.array(
        [
            mean / 255.0,
            std / 255.0,
            vmin / 255.0,
            vmax / 255.0,
            p10 / 255.0,
            p50 / 255.0,
            p90 / 255.0,
            ent / 5.0,
            gmean,
            gstd,
            gp90,
            lap_var,
            edge_density,
            ring_mean / 255.0,
            ring_std / 255.0,
            contrast_mean / 255.0,
            contrast_std / 255.0,
        ],
        dtype=np.float32,
    )

    return np.concatenate([geo, stats, small_flat], axis=0)


def build_dataset(
    images_dir: str | Path,
    labels_dir: str | Path,
    mlp_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build an MLP dataset by generating synthetic positive/negative boxes per GT."""
    rng = random.Random(RANDOM_SEED)

    pos_iou = mlp_cfg["pos_iou"]
    neg_low = mlp_cfg["lo_neg_iou"]
    neg_high = mlp_cfg["hi_neg_iou"]
    n = mlp_cfg["n"]

    pairs = list_image_label_pairs(images_dir, labels_dir)

    X_list: list[np.ndarray] = []
    y_list: list[int] = []
    groups: list[str] = []

    total_gt = 0
    total_skipped = 0

    for img_path, lab_path in pairs:
        gray = read_gray_u8(img_path)
        H, W = gray.shape

        labels = load_yolo_txt(lab_path)
        gts: list[BoxXYXY] = [yolo_norm_to_xyxy(xc, yc, bw, bh, W, H) for _, xc, yc, bw, bh in labels]
        if not gts:
            continue

        for gt_idx, gt in enumerate(gts):
            group_id = f"{img_path.stem}#gt{gt_idx}"
            total_gt += 1

            pos, neg = synthesize_pos_neg_for_gt(
                gt=gt,
                all_gts=gts,
                W=W,
                H=H,
                rng=rng,
                pos_iou=pos_iou,
                neg_low=neg_low,
                neg_high=neg_high,
                n_pos=n,
                n_neg=n,
                max_tries=4000,
            )

            if len(pos) != n or len(neg) != n:
                total_skipped += 1
                continue

            for b in pos:
                X_list.append(extract_features(gray, b))
                y_list.append(1)
                groups.append(group_id)

            for b in neg:
                X_list.append(extract_features(gray, b))
                y_list.append(0)
                groups.append(group_id)

    if not X_list:
        raise RuntimeError("Empty dataset: no samples were generated.")

    X = np.vstack([x.reshape(1, -1) for x in X_list]).astype(np.float32)
    y = np.array(y_list, dtype=np.int64)
    groups_arr = np.array(groups, dtype=object)

    print("=== Dataset summary ===")
    print(f"Frames used: {len(pairs)}")
    print(f"Sequences (groups): {len(set(groups_arr))}")
    print(f"GT total: {total_gt}")
    print(f"GT skip (failed to get 5 pos + 5 neg): {total_skipped}")
    print(f"Samples total: {len(y)} (pos={int(y.sum())}, neg={int((1-y).sum())})")
    print(f"Features dim: {X.shape[1]}")
    print("=======================")

    return X, y, groups_arr


def group_shuffle_split(groups: np.ndarray, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Split indices by shuffling unique group ids.
    Keeps groups together when possible to reduce leakage.
    """
    rng = random.Random(seed)
    n_total = len(groups)

    uniq = list(sorted(set(groups.tolist())))
    rng.shuffle(uniq)

    if len(uniq) >= 2:
        n_target = int(round(test_size * n_total))
        test_groups: set[str] = set()
        n_test = 0

        for g in uniq:
            if n_test >= n_target:
                break
            test_groups.add(g)
            n_test = int(np.sum(np.isin(groups, list(test_groups))))

        if len(test_groups) == 0:
            test_groups.add(uniq[0])
        if len(test_groups) == len(uniq):
            test_groups.remove(next(iter(test_groups)))

        is_test = np.isin(groups, list(test_groups))
        test_idx = np.where(is_test)[0]
        train_idx = np.where(~is_test)[0]

        if len(train_idx) > 0 and len(test_idx) > 0:
            return train_idx, test_idx

    idx = list(range(n_total))
    rng.shuffle(idx)
    n_test = max(1, int(round(test_size * n_total)))
    test_idx = np.array(idx[:n_test], dtype=np.int64)
    train_idx = np.array(idx[n_test:], dtype=np.int64)

    if len(train_idx) == 0:
        train_idx = test_idx[:1]
        test_idx = test_idx[1:] if len(test_idx) > 1 else test_idx

    return train_idx, test_idx


class NumpyDataset(Dataset):
    """Torch Dataset wrapper for numpy arrays."""

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self.X = torch.from_numpy(X).to(DTYPE)
        self.y = torch.from_numpy(y).to(torch.long)

    def __len__(self) -> int:
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def compute_standardizer(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute feature-wise mean and std for standardization."""
    mu = X.mean(axis=0, keepdims=True).astype(np.float32)
    sigma = X.std(axis=0, keepdims=True).astype(np.float32)
    sigma = np.maximum(sigma, 1e-6)
    return mu, sigma


def apply_standardizer(X: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Apply standardization using precomputed mean and std."""
    return ((X - mu) / sigma).astype(np.float32)


class MLP(nn.Module):
    """Small MLP used as a post-filter classifier."""

    def __init__(self, in_dim: int, hidden: tuple[int, ...], dropout: float) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU(inplace=True))
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the MLP, returning logits."""
        return self.net(x).squeeze(1)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, cfg: dict) -> MLP:
    """
    Train the MLP classifier on synthetic samples.
    Uses simple early stopping on training loss.
    """
    model = MLP(
        in_dim=X_train.shape[1],
        hidden=cfg["hidden"],
        dropout=cfg["dropout"],
    ).to(DEVICE)

    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )

    train_ds = NumpyDataset(X_train, y_train)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=0,
    )

    best_state: dict[str, torch.Tensor] | None = None
    best_loss = float("inf")
    bad_epochs = 0

    for epoch in range(cfg["epochs"]):
        model.train()
        total_loss = 0.0
        n = 0

        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.float().to(DEVICE)

            optim.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optim.step()

            bs = xb.size(0)
            total_loss += loss.item() * bs
            n += bs

        avg_loss = total_loss / max(1, n)

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= cfg["patience"]:
            print(f"[MLP] Early stopping at epoch {epoch+1} due to no improvement for {bad_epochs} epochs.")
            break

        print(
            f"[MLP] epoch {epoch+1}/{cfg['epochs']} done  avg_loss={avg_loss:.4f}  best={best_loss:.4f}  bad={bad_epochs}/{cfg['patience']}"
        )

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def mlp_probs_for_boxes(
    img_path: Path,
    boxes_xyxy: np.ndarray,
    mlp_model: nn.Module,
    mlp_mu: np.ndarray,
    mlp_sigma: np.ndarray,
) -> np.ndarray:
    """
    Return MLP probabilities for each candidate box.
    boxes_xyxy: (N,4) float32 -> returns probs: (N,) float32 in [0,1]
    """
    if boxes_xyxy.size == 0:
        return np.zeros((0,), dtype=np.float32)

    gray = read_gray_u8(img_path)
    mlp_dev = next(mlp_model.parameters()).device

    feats = []
    for b in boxes_xyxy:
        box = BoxXYXY(*b.tolist())
        f = extract_features(gray, box)
        f = (f - mlp_mu.squeeze()) / mlp_sigma.squeeze()
        feats.append(f)

    X = torch.from_numpy(np.stack(feats)).to(mlp_dev).float()

    with torch.no_grad():
        logits = mlp_model(X)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)

    return probs.astype(np.float32)


# ============================================================
# TRAIN MODEL (CODABENCH ENTRY POINT)
# ============================================================


def train_model(training_dir: str | Path, blobi: bool = False) -> torch.nn.Module:
    """
    Codabench entry point to train and return a wrapped detection model.

    Pipeline:
      1) Convert datasets to YOLO format (blob_dwi labeled + turb_i unlabeled).
      2) Train YOLO (round-1) on blob_dwi.
      3) Train MLP scorer on synthetic jittered boxes from blob_dwi GT.
      4) Pseudo-label turb_i with YOLO+MLP (top-K per image).
      5) Train YOLO (round-2) on blob_dwi + pseudo-labeled turb_i.
      6) Convert blob_i and mine TP/FP/FN patches using YOLO round-2.
      7) Train BlobCNN post-filter on mined patches.
      8) Return YOLOWrapper with post-processing + CNN filter.
    """
    cfg = get_config(CONFIG_ID)

    color_mode = cfg["color_mode"]
    model_ckpt = cfg["model_ckpt"]
    infer = cfg["infer"]
    yolo = cfg["yolo"]

    training_path = Path(training_dir)
    yolo_root = Path("/tmp/yolo_env")

    # Reset temp folder to avoid stale artifacts
    if yolo_root.exists():
        shutil.rmtree(yolo_root, ignore_errors=True)

    (yolo_root / "images").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels").mkdir(parents=True, exist_ok=True)

    # Step 1: Convert blob_dwi + turb_i (blobi=False keeps blob_i out of training)
    print("[INFO] Path to training data: ", training_path)
    print("[INFO] Converting datasets to YOLO format...")
    convert_datasets(training_path, yolo_root, color_mode=color_mode, blobi=blobi)

    data_yaml = yolo_root / "data.yaml"
    data_cfg = {
        "path": str(yolo_root.resolve()),
        "train": "images/train",
        "val": "images/train",
        "nc": 1,
        "names": ["blob"],
    }
    with open(data_yaml, "w") as f:
        yaml.dump(data_cfg, f)

    # Step 2: Round-1 YOLO training on labeled data (blob_dwi)
    yolo_model = YOLO(model_ckpt)

    print(f"[YOLO] Starting round-1 training on {len(list((yolo_root / 'images/train').glob('*.jpg')))} labeled images...")
    base_train_kwargs = dict(
        data=str(data_yaml),
        imgsz=512,
        batch=yolo["batch"],
        epochs=yolo["epochs"],
        workers=0,
        patience=yolo["patience"],
        device=DEVICE,
        save=True,
        plots=False,
        augment=True,
        project="/tmp/runs",
        name=cfg["cfg_id"],
        exist_ok=True,
        cache=False,
        resume=False,
    )
    train_kwargs = build_train_kwargs(base_train_kwargs, cfg)
    results = yolo_model.train(**train_kwargs)

    run_dir = Path(results.save_dir)
    print(f"[YOLO] Training run dir: {run_dir}")

    w_best = run_dir / "weights" / "best.pt"
    w_last = run_dir / "weights" / "last.pt"
    w_round1 = w_best if w_best.exists() else w_last if w_last.exists() else None

    if w_round1 is None:
        raise RuntimeError(
            f"[YOLO] No weights found in {run_dir}/weights. Files={list((run_dir/'weights').glob('*'))}"
        )

    yolo_model = YOLO(str(w_round1))
    print(f"[YOLO] Round-1 weights: {w_round1}")

    # Step 3: Train MLP on synthetic samples from labeled images (blob_dwi)
    mlp_cfg = MLP_CONFIGS[CONFIG_ID]
    images_dir_labeled = yolo_root / "images/train"
    labels_dir_labeled = yolo_root / "labels/train"

    print("[MLP] Building dataset for MLP training...")
    X, y_arr, groups = build_dataset(images_dir_labeled, labels_dir_labeled, mlp_cfg)

    print("[MLP] Splitting dataset into train/test...")
    train_idx, _ = group_shuffle_split(groups, test_size=0.2, seed=RANDOM_SEED + 1)
    X_train = X[train_idx]
    y_train = y_arr[train_idx]
    print(f"[MLP] Training samples: {len(y_train)} (pos={int(y_train.sum())}, neg={int((1-y_train).sum())})")

    print("[MLP] Computing standardizer for MLP features...")
    mu, sigma = compute_standardizer(X_train)
    X_train_std = apply_standardizer(X_train, mu, sigma)

    print(f"[MLP] Training MLP model on {len(X_train_std)} samples...")
    mlp_model = train_mlp(X_train_std, y_train, mlp_cfg)

    # Step 4: Pseudo-label unlabeled images (turb_i) with top-K by MLP probability
    images_unlabeled_dir = yolo_root / "images/unlabeled"
    labels_unlabeled_dir = yolo_root / "labels/unlabeled"
    unlab_imgs = list_unlabeled_images(images_unlabeled_dir)
    print(f"[PL] Unlabeled images found: {len(unlab_imgs)}")

    TOP_K = 40

    if len(unlab_imgs) > 0:
        print("[PL] Running YOLO inference on unlabeled images...")
        raw_boxes = yolo_predict_xyxy_on_images(
            model=yolo_model,
            image_paths=unlab_imgs,
            conf=infer["conf"],
            iou=infer["iou"],
            max_det=infer["max_det"],
        )

        filt_boxes: dict[str, np.ndarray] = {}
        kept_total = 0
        
        print("[PL] Scoring and filtering boxes with MLP...")
        for img_path in unlab_imgs:
            stem = img_path.stem
            xyxy = raw_boxes.get(stem, np.zeros((0, 4), dtype=np.float32))

            if xyxy.size == 0:
                filt_boxes[stem] = xyxy
                continue

            probs = mlp_probs_for_boxes(
                img_path=img_path,
                boxes_xyxy=xyxy,
                mlp_model=mlp_model,
                mlp_mu=mu,
                mlp_sigma=sigma,
            )

            k = min(TOP_K, probs.shape[0])
            if k <= 0:
                xyxy_top = np.zeros((0, 4), dtype=np.float32)
            else:
                top_idx = np.argsort(-probs)[:k]
                xyxy_top = xyxy[top_idx]

            filt_boxes[stem] = xyxy_top
            kept_total += int(xyxy_top.shape[0])

        n_written = write_pseudo_labels_yolo(
            images_unlabeled_dir=images_unlabeled_dir,
            labels_unlabeled_dir=labels_unlabeled_dir,
            boxes_by_stem_xyxy=filt_boxes,
        )
        print(f"[PL] Pseudo labels written: total_boxes={n_written}, kept_total={kept_total}")

        combined_yaml = yolo_root / "data_pseudo.yaml"
        make_combined_yolo_yaml(yolo_root=yolo_root, out_yaml_path=combined_yaml)

        # Step 5: Round-2 YOLO training on blob_dwi + pseudo-labeled turb_i
        print(f"[YOLO] Starting round-2 training on {len(list((yolo_root / 'images/train').glob('*.jpg')))} labeled images and {n_written} pseudo-labeled images...")

        yolo_model_pl = YOLO(str(w_round1))

        base_train_kwargs_pl = dict(base_train_kwargs)
        base_train_kwargs_pl["data"] = str(combined_yaml)
        base_train_kwargs_pl["name"] = cfg["cfg_id"] + "_pl"
        base_train_kwargs_pl["resume"] = False
        train_kwargs_pl = build_train_kwargs(base_train_kwargs_pl, cfg)

        results_pl = yolo_model_pl.train(**train_kwargs_pl)

        run_dir_pl = Path(results_pl.save_dir)
        print(f"[YOLO] Round-2 run dir: {run_dir_pl}")

        w_best_pl = run_dir_pl / "weights" / "best.pt"
        w_last_pl = run_dir_pl / "weights" / "last.pt"
        w_round2 = w_best_pl if w_best_pl.exists() else w_last_pl if w_last_pl.exists() else None

        if w_round2 is None:
            raise RuntimeError(
                f"[YOLO] No round-2 weights found in {run_dir_pl}/weights. Files={list((run_dir_pl/'weights').glob('*'))}"
            )

        print(f"[YOLO] Round-2 weights: {w_round2}")
        yolo_model = YOLO(str(w_round2))
    else:
        print("[PL] No unlabeled images, skipping pseudo-labeling.")

    # Step 6: Convert blob_i with blobi=True to get its labeled images,
    # then mine TP/FP/FN from YOLO round-2 predictions vs blob_i GT.
    # blob_i images are exported to images/train alongside blob_dwi when blobi=True,
    # but we need them separately here so we use a dedicated temporary yolo_root.
    print("[CNN] Converting blob_i for CNN training data mining...")
    yolo_root_blobi = Path("/tmp/yolo_env_blobi")
    if yolo_root_blobi.exists():
        shutil.rmtree(yolo_root_blobi, ignore_errors=True)

    # blobi=True: include blob_i in the annotated set
    convert_datasets(training_path, yolo_root_blobi, color_mode=color_mode, blobi=True)

    # Mine blob_i frames with the round-2 YOLO
    print(f"[CNN] Mining blob_i for CNN training data...")
    cnn_patches, cnn_phys, cnn_labels = mine_blob_i_for_cnn(
        yolo_model=yolo_model,
        blob_i_images_dir=yolo_root_blobi / "images/train",
        blob_i_labels_dir=yolo_root_blobi / "labels/train",
        infer_cfg=infer,
    )

    # Step 7: Train BlobCNN on mined patches
    cnn_cfg = CNN_CONFIGS[CONFIG_ID]
    cnn_model: BlobCNN | None = None

    if cnn_patches.shape[0] > 0:
        print(f"[CNN] Training BlobCNN on {cnn_patches.shape[0]} mined patches...")
        cnn_model = train_blob_cnn(cnn_patches, cnn_phys, cnn_labels, cnn_cfg)
        print("[CNN] BlobCNN training complete.")
    else:
        print("[CNN] No CNN training data — CNN post-filter disabled.")

    print("[INFO] Cleaning up temporary files...")
    shutil.rmtree(yolo_root_blobi, ignore_errors=True)

    # Step 8: Return wrapped model with CNN post-filter
    return YOLOWrapper(
        yolo_model,
        color_mode=color_mode,
        conf=infer["conf"],
        iou=infer["iou"],
        max_det=infer["max_det"],
        cnn_model=cnn_model,
    )