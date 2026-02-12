"""
Codabench submission script for blob detection on density videos.

This trains a YOLO detector and a lightweight MLP post-filter, then wraps
inference in a torch.nn.Module compatible with Codabench.

Final Implementation:
- Config c02: YOLOv8n with grayscale color mode and moderate data augmentation
- First training YOLO on the 30 annotated images, then pseudo-labeling the 8 unlabeled
  images with an aggressive MLP filter, and retraining YOLO on the combined 38 images.
- MLP trained on synthetic jittered boxes around GTs, with a focus on hard negatives,
  to filter out false positives from YOLO while keeping recall high. Features used
  are simple intensity-based statistics, histogram of gradients, and simple geometric 
  features extracted from the grayscale image within each box.
- At inference, the YOLO predictions are post-filtered with its NMS that removes duplicates
  and then by the MLP that removes boxes unlikely to contain a blob to improve precision.

Different Implementations tried during development (that showed no improvement over the final one):
- Different configs, including color modes (plasma and physique inspired modes were tried),
  augmentation strengths, different batch sizes and patience values, and various YOLO architectures 
  (yolov8s/m/l, yolov11n). Grayscale showed better performance than other color modes, and the final 
  config c02 with moderate augmentation and mosaic/mixup performed best. Other YOLO architectures 
  did not show improvement over yolov8n, likely due to the small dataset size.
- Different post-processing methods were tried to improve YOLO precision, including NMS with different 
  thresholds and confidence thresholding. The MLP post-filter showed the best improvement in precision 
  while maintaining recall, likely because it can learn a more complex decision boundary based on 
  multiple features. We tried to post-filter boxes based on the combined confidence of YOLO and the MLP, 
  using differents methods such as weighted average of them, product or using a meta-model, but it did 
  not show improvement over just using the MLP's prediction, likely because the MLP is already trained 
  to predict the probability of a box containing a blob, so its output is more directly aligned with the 
  final decision than YOLO's confidence.
- For the pseudo-labeling, different post-filtering methods were tried to select the most confident boxes 
  from YOLO for labeling the unlabeled data, including a threshold on MLP or a top-k prediction (selected
  from the confidence of MLP). For the pseudo-labeling part, the top-k was better, while for inference
  the thresholding was better.
- Some self-supervised representation learning methods were also tried to pre-train YOLO's backbone on 
  the unlabeled data with a convolutional VAE, but they did not show improvement over the final approach
  (nor did they degrade the performance), likely because the dataset is small and the domain is quite 
  specific, so the benefits of self-supervised pre-training may be limited.
- Finally, some data generation methods were tried. The first one consisted in removing blobs from the
  annotated images by inpainting them with OpenCV, and then using these modified images as background,
  and pasting blobs from the annotated images at random locations to create new synthetic training images. 
  This degraded the performance a lot but could've been done in a better way, the synthetic images were not
  good at all. The second one used a VAE on all the train images to generate new synthetic images, but it did 
  not show improvement over the final approach, likely because the VAE-generated images were too close to
  the real images (the generated images were great though).
"""

# submission.py
# Baptiste PRAS
# baptiste.pras@universite-paris-saclay.fr
# 08-Feb-2026
# python 3.10+

import shutil
import math
import random
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
from torch.utils.data import Dataset, DataLoader
from ultralytics import YOLO

# =========================
# GLOBALS
# =========================


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_infer_device() -> str:
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
CONFIG_ID = "c02"

CONFIGS = {
    # c01
    "c01": dict(
        cfg_id="c01_y8n_gray_07513",
        model_ckpt="yolov8n.pt",
        color_mode="gray",
        infer=dict(conf=0.01, iou=0.55, max_det=300),
        aug=dict(
            fliplr=0.0,
            flipud=0.2,
            scale=0.2,
            translate=0.08,
            degrees=5.0,
            hsv_v=0.0,
            hsv_s=0.0,
            mosaic=0.0,
            mixup=0.0,
        ),
        yolo=dict(batch=8, epochs=1000, patience=30),
    ),
    # c02
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
    "gray": dict(
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

    # Aug priority: override > cfg["aug"]
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
# Transform Ultralytics YOLO inference into a torch.nn.Module
# that matches Codabench's expected input/output API.
# ============================================================


class YOLOWrapper(torch.nn.Module):
    """Torch wrapper exposing YOLO inference + optional MLP post-filtering."""

    def __init__(
        self,
        yolo_model: YOLO,
        color_mode: str = "gray",
        conf: float = 0.01,
        iou: float = 0.55,
        max_det: int = 300,
        mlp_model: nn.Module = None,
        mlp_mu: np.ndarray = None,
        mlp_sigma: np.ndarray = None,
        mlp_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self._yolo = yolo_model  # Ultralytics YOLO object
        self.network = yolo_model.model  # underlying torch model (train/eval)
        self.color_mode = color_mode

        # Inference parameters
        self.conf = float(conf)
        self.iou = float(iou)
        self.max_det = int(max_det)

        # MLP parameters
        self.mlp = mlp_model
        self.mlp_mu = mlp_mu
        self.mlp_sigma = mlp_sigma
        self.mlp_threshold = mlp_threshold

    def _density_tensor_to_rgb_uint8(self, t: torch.Tensor) -> np.ndarray:
        """
        Convert a (C,H,W) tensor into an RGB uint8 image for YOLO inference.
        Supports 1-channel density inputs or 3-channel images.
        """
        t = t.detach().cpu()

        if t.ndim != 3:
            raise ValueError(f"Expected tensor (C,H,W), got shape={tuple(t.shape)}")

        C, _, _ = t.shape  # Channels, Height, Width
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
    
    def _mlp_filter(self, gray_u8: np.ndarray, boxes: torch.Tensor) -> torch.Tensor:
        """Returns a CPU boolean mask of size (N,) indicating which boxes to keep."""
        if self.mlp is None or boxes.numel() == 0:
            return torch.ones((boxes.shape[0],), dtype=torch.bool)

        # Device of the MLP
        mlp_dev = next(self.mlp.parameters()).device

        feats = []
        for b in boxes.cpu().numpy():
            box = BoxXYXY(*b.tolist())
            f = extract_features(gray_u8, box)
            if self.mlp_mu is None or self.mlp_sigma is None:
                raise RuntimeError("MLP standardizer (mu/sigma) missing.")
            f = (f - self.mlp_mu.squeeze()) / self.mlp_sigma.squeeze()
            feats.append(f)

        X = torch.from_numpy(np.stack(feats)).to(mlp_dev)

        with torch.no_grad():
            logits = self.mlp(X)
            probs = torch.sigmoid(logits).detach().cpu()  # <- CPU bool tensor for consistency

        keep = probs >= float(self.mlp_threshold)  # CPU bool tensor
        return keep

    def forward(self, x: torch.Tensor) -> list[dict]:
        """
        Run YOLO inference and apply post-filtering.
        Outputs CPU tensors to avoid Codabench device surprises.
        """
        # Support both single tensors and lists of tensors as input
        if isinstance(x, list):
            xs = x
        elif isinstance(x, torch.Tensor) and x.ndim == 3:
            xs = [x]
        else:
            xs = list(x)

        infer_device_str = get_infer_device()
        out_device = torch.device("cpu")

        inputs = [self._density_tensor_to_rgb_uint8(t) for t in xs]

        preds = self._yolo.predict(
            inputs,
            conf=self.conf,
            iou=self.iou,
            max_det=self.max_det,
            device=infer_device_str,
            verbose=False,
        )

        outputs: list[dict] = []
        for k, p in enumerate(preds):
            if len(p.boxes) == 0:
                outputs.append(
                    {
                        "boxes": torch.zeros((0, 4), device=out_device),
                        "scores": torch.zeros((0,), device=out_device),
                        "labels": torch.zeros((0,), dtype=torch.int64, device=out_device),
                    }
                )
                continue

            # Keep everything on CPU for consistency (Codabench-friendly)
            boxes = p.boxes.xyxy.detach().cpu().float()
            scores_yolo = p.boxes.conf.detach().cpu().float()
            labels = p.boxes.cls.detach().cpu().long()
            gray = cv2.cvtColor(inputs[k], cv2.COLOR_RGB2GRAY)

            # Filter boxes/scores/labels based on MLP predictions
            keep = self._mlp_filter(gray, boxes)  # mask CPU
            boxes = boxes[keep]
            scores_yolo = scores_yolo[keep]
            labels = labels[keep]

            outputs.append(
                {
                    "boxes": boxes.to(out_device),
                    "scores": scores_yolo.to(out_device),
                    "labels": labels.to(out_device),
                }
            )

        return outputs

    def train(self, mode: bool = True):
        self.training = mode
        self.network.train(mode)
        if self.mlp is not None:
            self.mlp.train(mode)
        return self

    def eval(self):
        return self.train(False)


# ============================================================
# DATA CONVERSION (H5 + XML -> YOLO PNG + TXT)
# Convert challenge data into a standard YOLO directory:
#   /tmp/yolo_env/images/train/*.png
#   /tmp/yolo_env/labels/train/*.txt
# Plus an unlabeled pool of images:
#   /tmp/yolo_env/images/unlabeled/*.png
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
    """
    Convert a density frame to an RGB uint8 image.
    Uses normalization and a chosen color encoding.
    """
    gray_u8 = density_to_uint8_norm(frame)
    rgb = apply_color_mode_opencv(gray_u8, color_mode=color_mode)
    return rgb


def load_h5_video(h5_path: str | Path) -> np.ndarray:
    """
    Load a video array (T,H,W) from a .h5 file.
    Prefers a dataset named 'density' when available.
    """
    with h5py.File(h5_path, "r") as f:
        if "density" in f:
            arr = np.array(f["density"])
            if arr.ndim == 3:
                return arr

        # Fallback: first 3D array in the file
        for k in f.keys():
            arr = np.array(f[k])
            if arr.ndim == 3:
                return arr

        raise ValueError(
            f"No 3D (T,H,W) dataset found in {h5_path}. Keys={list(f.keys())}"
        )


def convert_datasets(training_path: str | Path, yolo_root: str | Path, color_mode: str = "gray") -> None:
    """
    Convert challenge files into a YOLO-like folder with PNG images and TXT labels.
    Also exports unlabeled frames from H5-only files.
    """
    training_path = Path(training_path)
    yolo_root = Path(yolo_root)

    img_dir_train = yolo_root / "images/train"
    lab_dir_train = yolo_root / "labels/train"
    img_dir_unlab = yolo_root / "images/unlabeled"

    # YOLO folder structure
    img_dir_train.mkdir(parents=True, exist_ok=True)
    lab_dir_train.mkdir(parents=True, exist_ok=True)
    img_dir_unlab.mkdir(parents=True, exist_ok=True)

    # Discover files
    xml_files = sorted(training_path.glob("*.xml"))
    h5_files = sorted(training_path.glob("*.h5"))

    xml_by_stem = {p.stem: p for p in xml_files}
    h5_by_stem = {p.stem: p for p in h5_files}

    annotated_stems = sorted(set(xml_by_stem.keys()) & set(h5_by_stem.keys()))
    unlabeled_stems = sorted(set(h5_by_stem.keys()) - set(xml_by_stem.keys()))

    n_annot_frames = 0
    n_unlab_frames = 0
    n_annot_boxes = 0

    # Labeled data: XML + H5
    for stem in annotated_stems:
        xml_path = xml_by_stem[stem]
        h5_path = h5_by_stem[stem]

        tree = ET.parse(xml_path)
        root = tree.getroot()

        # Load the full H5 as (T,H,W)
        video = load_h5_video(h5_path)
        T = video.shape[0]

        skipped = 0
        for img_tag in root.findall("image"):
            # The correct index to access the H5 frame is the XML attribute "id"
            frame_id = int(img_tag.get("id"))

            # "name" is only used for naming the output (e.g. "150.png" -> "150")
            name_noext = img_tag.get("name").replace(".png", "")

            if frame_id >= T:
                skipped += 1
                continue

            w = float(img_tag.get("width"))
            h = float(img_tag.get("height"))

            # Convert density frame to a visible RGB image
            frame = video[frame_id]
            frame_rgb = density_to_rgb(frame, color_mode=color_mode)

            # Prefix stem to avoid collisions between different H5 files
            out_name = f"{stem}_{name_noext}"
            Image.fromarray(frame_rgb).save(img_dir_train / f"{out_name}.png")

            # Convert boxes to YOLO txt format (class xc yc w h in [0,1])
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
            print(
                f"[WARN] {stem}: skipped {skipped} XML frames because frame_id >= T={T}"
            )

    # Unlabeled data: H5 only
    for stem in unlabeled_stems:
        h5_path = h5_by_stem[stem]
        video = load_h5_video(h5_path)

        # Export every frame as an unlabeled image
        for frame_id in range(video.shape[0]):
            frame = video[frame_id]
            frame_rgb = density_to_rgb(frame, color_mode=color_mode)

            out_name = f"{stem}_{frame_id}"
            Image.fromarray(frame_rgb).save(img_dir_unlab / f"{out_name}.png")
            n_unlab_frames += 1

    # Conversion summary (useful for debugging inside Codabench logs)
    print("\n=== Dataset conversion summary ===")
    print(f"Annotated files (xml+h5): {len(annotated_stems)} -> {annotated_stems}")
    print(f"Unlabeled files (h5 only): {len(unlabeled_stems)} -> {unlabeled_stems}")
    print(f"Annotated frames exported: {n_annot_frames}")
    print(f"Annotated boxes exported:  {n_annot_boxes}")
    print(f"Unlabeled frames exported: {n_unlab_frames}")
    print("================================\n")


# ============================================================
# PSEUDO-LABELING UTILITIES
# Used to label the unlabeled data
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
    """
    Run YOLO prediction on a list of image paths.
    Returns dict: stem -> xyxy float32 array (N,4) on CPU.
    """
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


def filter_xyxy_with_mlp(
    img_path: Path,
    boxes_xyxy: np.ndarray,
    mlp_model: nn.Module,
    mlp_mu: np.ndarray,
    mlp_sigma: np.ndarray,
    mlp_threshold: float,
) -> np.ndarray:
    """
    Apply the SAME MLP filtering as in inference.
    Input boxes_xyxy: (N,4) float32
    Output: filtered boxes_xyxy: (K,4)
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy

    # read grayscale uint8 exactly as your feature extractor expects
    gray = read_gray_u8(img_path)  # (H,W) uint8

    # MLP device
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

    keep = probs >= float(mlp_threshold)
    return boxes_xyxy[keep]


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
    """
    Write pseudo YOLO labels into labels_unlabeled_dir for images/unlabeled.
    Returns total number of boxes written.
    """
    images_unlabeled_dir = Path(images_unlabeled_dir)
    labels_unlabeled_dir = Path(labels_unlabeled_dir)
    labels_unlabeled_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for stem, xyxy in boxes_by_stem_xyxy.items():
        # locate image (png by default, but support any ext)
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
        

# ============================================================
# MLP TRAINING UTILITIES
# Classifies whether a candidate box contains a blob or not.
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


def yolo_norm_to_xyxy(
    xc: float, yc: float, bw: float, bh: float, W: int, H: int
) -> BoxXYXY:
    """
    Convert a YOLO-normalized box to absolute XYXY coordinates.
    The resulting box is clipped to image boundaries.
    """
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
    """
    Compute Intersection-over-Union between two boxes.
    Returns 0.0 when union is zero.
    """
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
    """
    Compute the max IoU between a candidate box and a list of ground-truth boxes.
    Returns 0.0 if the list is empty.
    """
    if not gts:
        return 0.0
    return max(iou(candidate, gt) for gt in gts)


def load_yolo_txt(txt_path: str | Path) -> list[tuple[int, float, float, float, float]]:
    """
    Load YOLO labels from a .txt file.
    Returns a list of (class, xc, yc, w, h).
    """
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
    """
    List images that have a matching YOLO label file.
    Returns (image_path, label_path) pairs.
    """
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
    """
    Read an image and return a uint8 grayscale array.
    Used by the feature extractor.
    """
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
        """
        Sample a random shift fraction with a bias towards smaller shifts.
        """
        u = rng.random()
        if u < 0.70:
            return rng.uniform(-0.20, 0.20)
        return rng.uniform(-1.00, 1.00)

    dx_frac = sample_shift_frac()
    dy_frac = sample_shift_frac()

    dx = dx_frac * w * translate_max_frac * 4.0
    dy = dy_frac * h * translate_max_frac * 4.0

    sw = 1.0 + rng.uniform(-scale_jitter, scale_jitter)  # scale width
    sh = 1.0 + rng.uniform(-scale_jitter, scale_jitter)  # scale height

    nw = max(2.0, w * sw)  # new width
    nh = max(2.0, h * sh)  # new height

    ncx = cx + dx  # new center x
    ncy = cy + dy  # new center y

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

    # Try many jittered candidates and keep those that meet IoU criteria until we have enough
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
                pos.append(cand)  # candidate is a positive sample (IoU above pos_iou)
        elif cur_neg_low <= miou <= cur_neg_high:
            if len(neg) < n_neg:
                neg.append(cand)  # candidate is a negative sample (IoU between neg_low and neg_high)

        if len(pos) == n_pos and len(neg) == n_neg:
            return pos, neg

        # Loosen negative IoU bounds if sampling is too hard.
        if (t + 1) % 800 == 0:
            cur_neg_high = min(0.49, cur_neg_high + 0.05)
            cur_neg_low = max(0.0, cur_neg_low - 0.02)

    return pos, neg


def sobel_gradients(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute simple Sobel gradients in x and y (manual implementation)."""
    p2 = np.pad(p, ((1, 1), (1, 1)), mode="edge")

    kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)  # horizontal edges
    ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)  # vertical edges

    H, W = p.shape
    gx = np.zeros((H, W), dtype=np.float32)  # horizontal gradient response
    gy = np.zeros((H, W), dtype=np.float32)  # vertical gradient response

    # Convolve the 3x3 Sobel kernels over the image to compute gradients at each pixel
    for y in range(H):
        for x in range(W):
            patch = p2[y : y + 3, x : x + 3]
            gx[y, x] = float(np.sum(patch * kx))
            gy[y, x] = float(np.sum(patch * ky))

    return gx, gy


def laplacian(p: np.ndarray) -> np.ndarray:
    """Compute a simple Laplacian response (manual implementation)."""
    p2 = np.pad(p, ((1, 1), (1, 1)), mode="edge")
    k = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)  # Laplacian kernel for edge detection
    H, W = p.shape
    out = np.zeros((H, W), dtype=np.float32)
    # Convolve the 3x3 Laplacian kernel over the image to compute the second derivative response at each pixel
    for y in range(H):
        for x in range(W):
            patch = p2[y : y + 3, x : x + 3]
            out[y, x] = float(np.sum(patch * k))
    return out


def entropy_hist(x: np.ndarray, bins: int = 32) -> float:
    """Compute a histogram-based entropy estimate for an array."""
    if x.size == 0:
        return 0.0
    x = np.clip(x, 0, 255).astype(np.uint8).reshape(-1)  # flatten to 1D
    idx = (x.astype(np.int32) * bins) // 256
    hist = np.bincount(idx, minlength=bins).astype(np.float64)
    p = hist / (hist.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def crop_patch(gray: np.ndarray, b: BoxXYXY) -> np.ndarray:
    """
    Crop a grayscale patch corresponding to a box.
    Returns a small empty patch when invalid.
    """
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

    # Outer box coordinates
    xb1 = int(round(cx - 0.5 * nw))
    yb1 = int(round(cy - 0.5 * nh))
    xb2 = int(round(cx + 0.5 * nw))
    yb2 = int(round(cy + 0.5 * nh))

    # Clip to image bounds
    xb1 = max(0, min(xb1, W - 1))
    yb1 = max(0, min(yb1, H - 1))
    xb2 = max(0, min(xb2, W))
    yb2 = max(0, min(yb2, H))

    outer = gray[yb1:yb2, xb1:xb2]
    if outer.size == 0:
        return np.zeros((0,), dtype=np.uint8)

    # Inner box coordinates relative to outer crop
    x1 = int(round(b.x1)) - xb1
    y1 = int(round(b.y1)) - yb1
    x2 = int(round(b.x2)) - xb1
    y2 = int(round(b.y2)) - yb1

    # Clip inner box to outer crop bounds
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

    # Intensity-based features
    mean = float(patch.mean())
    std = float(patch.std())
    vmin = float(patch.min())
    vmax = float(patch.max())
    p10 = float(np.percentile(patch, 10))
    p50 = float(np.percentile(patch, 50))
    p90 = float(np.percentile(patch, 90))
    ent = float(entropy_hist(patch.astype(np.uint8), bins=32))

    # Sobel features
    p_norm = patch / 255.0
    gx, gy = sobel_gradients(p_norm)
    gmag = np.sqrt(gx * gx + gy * gy)
    gmean = float(gmag.mean())
    gstd = float(gmag.std())
    gp90 = float(np.percentile(gmag, 90))

    # Laplacian features
    lap = laplacian(p_norm)
    lap_abs = np.abs(lap)
    lap_var = float(lap_abs.var())
    edge_density = float((gmag > np.percentile(gmag, 80)).mean())

    # Ring features (local background)
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

    # Geometry features (normalized to image size)
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

    # Intensity and texture features
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
    images_dir: str | Path, labels_dir: str | Path, mlp_cfg: dict,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build an MLP dataset by generating synthetic positive/negative boxes per GT.
    Returns features, labels, and group ids for splitting.
    """
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

    # Process each image and its labels
    for img_path, lab_path in pairs:
        gray = read_gray_u8(img_path)
        H, W = gray.shape

        labels = load_yolo_txt(lab_path)
        gts: list[BoxXYXY] = [
            yolo_norm_to_xyxy(xc, yc, bw, bh, W, H) for _, xc, yc, bw, bh in labels
        ]
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

            # Strict requirement for a balanced set of positives and negatives per GT
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

    # Normal case: at least 2 groups to split
    if len(uniq) >= 2:
        n_target = int(round(test_size * n_total))
        test_groups: set[str] = set()
        n_test = 0

        # Add groups until reaching the target test size
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

        # Safety check: keep a valid split.
        if len(train_idx) > 0 and len(test_idx) > 0:
            return train_idx, test_idx

    # Fallback: simple shuffled split if group split fails
    idx = list(range(n_total))
    rng.shuffle(idx)
    n_test = max(1, int(round(test_size * n_total)))
    test_idx = np.array(idx[:n_test], dtype=np.int64)
    train_idx = np.array(idx[n_test:], dtype=np.int64)

    # Last-resort safety
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
    """
    Compute feature-wise mean and std for standardization.
    A small std floor avoids division by zero.
    """
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
        layers.append(nn.Linear(prev, 1))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    # Training setup
    loss_fn = nn.BCEWithLogitsLoss()
    optim = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["wd"],
    )

    # DataLoader
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

        # Iterate over batches and perform training steps
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

        print(f"[MLP] epoch {epoch+1}/{cfg['epochs']} done  avg_loss={avg_loss:.4f}  best={best_loss:.4f}  bad={bad_epochs}/{cfg['patience']}")

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

        gray = read_gray_u8(img_path)  # (H,W) uint8

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
# Codabench calls train_model(training_dir) and expects
# a torch.nn.Module instance returned.
# ============================================================


def train_model(training_dir: str | Path) -> torch.nn.Module:
    """
    Codabench entry point to train and return a wrapped detection model.
    Runs dataset conversion, YOLO training, then trains the MLP post-filter.
    """
    cfg = get_config(CONFIG_ID)

    color_mode = cfg["color_mode"]
    model_ckpt = cfg["model_ckpt"]
    infer = cfg["infer"]
    yolo = cfg["yolo"]

    training_path = Path(training_dir)
    yolo_root = Path("/tmp/yolo_env")

    # Clean temp folder to ensure no stale data from previous runs
    if yolo_root.exists():
        shutil.rmtree(yolo_root, ignore_errors=True)

    (yolo_root / "images").mkdir(parents=True, exist_ok=True)
    (yolo_root / "labels").mkdir(parents=True, exist_ok=True)

    convert_datasets(training_path, yolo_root, color_mode=color_mode)

    # 
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

    # Train YOLO on the 30 labeled images
    yolo_model = YOLO(model_ckpt)

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

    # Ultralytics returns a Results object with save_dir
    run_dir = Path(results.save_dir)
    print(f"[YOLO] Training run dir: {run_dir}")

    w_best = run_dir / "weights" / "best.pt"
    w_last = run_dir / "weights" / "last.pt"
    w_round1 = w_best if w_best.exists() else w_last if w_last.exists() else None

    if w_round1 is None:
        raise RuntimeError(
            f"[YOLO] No weights found in {run_dir}/weights. "
            f"Files={list((run_dir/'weights').glob('*'))}"
        )
    yolo_model = YOLO(str(w_round1))
    print(f"[YOLO] Round-1 weights: {w_round1}")
    
    # Train MLP on synthetic pos/neg boxes generated from the 30 labeled images
    mlp_cfg = MLP_CONFIGS[color_mode]
    images_dir_labeled = yolo_root / "images/train"
    labels_dir_labeled = yolo_root / "labels/train"

    X, y_arr, groups = build_dataset(images_dir_labeled, labels_dir_labeled, mlp_cfg)

    train_idx, _ = group_shuffle_split(groups, test_size=0.2, seed=RANDOM_SEED + 1)
    X_train = X[train_idx]
    y_train = y_arr[train_idx]

    mu, sigma = compute_standardizer(X_train)
    X_train_std = apply_standardizer(X_train, mu, sigma)

    mlp_model = train_mlp(X_train_std, y_train, mlp_cfg)
    mlp_threshold = mlp_cfg["mlp_threshold"]

    # Pseudo-labeling on the 8 unlabeled images using YOLO + MLP top-k filter (by MLP prob)
    images_unlabeled_dir = yolo_root / "images/unlabeled"
    labels_unlabeled_dir = yolo_root / "labels/unlabeled"
    unlab_imgs = list_unlabeled_images(images_unlabeled_dir)
    print(f"[PL] Unlabeled images found: {len(unlab_imgs)}")

    TOP_K = 40  # Max number of pseudo-labels to keep per image

    if len(unlab_imgs) > 0:
        # YOLO predict (xyxy)
        raw_boxes = yolo_predict_xyxy_on_images(
            model=yolo_model,
            image_paths=unlab_imgs,
            conf=infer["conf"],
            iou=infer["iou"],
            max_det=infer["max_det"],
        )

        # MLP top-k per image (rank by MLP probability)
        filt_boxes: dict[str, np.ndarray] = {}
        kept_total = 0

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

            # Keep top-k by probability (descending)
            k = min(TOP_K, probs.shape[0])
            if k <= 0:
                xyxy_top = np.zeros((0, 4), dtype=np.float32)
            else:
                top_idx = np.argsort(-probs)[:k]
                xyxy_top = xyxy[top_idx]

            filt_boxes[stem] = xyxy_top
            kept_total += int(xyxy_top.shape[0])

        # Write pseudo labels (overwrites previous pseudo labels for these stems)
        n_written = write_pseudo_labels_yolo(
            images_unlabeled_dir=images_unlabeled_dir,
            labels_unlabeled_dir=labels_unlabeled_dir,
            boxes_by_stem_xyxy=filt_boxes,
        )
        print(f"[PL] Pseudo labels written: total_boxes={n_written}, kept_total={kept_total}")

        # Retrain YOLO on combined data (30 labeled + 8 pseudo-labeled)
        combined_yaml = yolo_root / "data_pseudo.yaml"
        make_combined_yolo_yaml(yolo_root=yolo_root, out_yaml_path=combined_yaml)

        print("[PL] Starting YOLO round-2 training on labeled + pseudo-labeled data...")

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
                f"[YOLO] No round-2 weights found in {run_dir_pl}/weights. "
                f"Files={list((run_dir_pl/'weights').glob('*'))}"
            )

        print(f"[YOLO] Round-2 weights: {w_round2}")
        yolo_model = YOLO(str(w_round2))
    else:
        print("[PL] No unlabeled images, skipping pseudo-labeling.")

    # Return final wrapper
    return YOLOWrapper(
        yolo_model,
        color_mode=color_mode,
        conf=infer["conf"],
        iou=infer["iou"],
        max_det=infer["max_det"],
        mlp_model=mlp_model,
        mlp_mu=mu,
        mlp_sigma=sigma,
        mlp_threshold=mlp_threshold,
    )