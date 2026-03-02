#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local runner for Tokam2D submission (no TokamDataset).

What it does
- Trains your model by calling `train_model(train/)` from `submission.py`.
- Loads 4 frames from `test/test.h5` and 4 frames from `private_test/private_test.h5`.
- Runs inference with your trained model on those 8 frames.
- Saves a single PNG mosaic in the current directory with predicted boxes in green.

Usage
  python pred.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pickle
import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch


N_PER_SET = 4
OUT_PNG = Path("mosaic_test_private.png")


def _find_frames_key(h5: h5py.File) -> str:
    """Heuristically find the dataset key that contains the video frames."""
    # Common keys seen in h5 challenges; we keep it robust.
    candidates = ["X", "frames", "images", "data", "video"]
    for k in candidates:
        if k in h5 and isinstance(h5[k], h5py.Dataset) and h5[k].ndim >= 3:
            return k
    # Fallback: first dataset with ndim>=3
    for k in h5.keys():
        obj = h5[k]
        if isinstance(obj, h5py.Dataset) and obj.ndim >= 3:
            return k
    raise KeyError("Could not find frames dataset in .h5 (expected a dataset with ndim>=3).")


def _to_chw_float32(frame: np.ndarray) -> torch.Tensor:
    """Convert a frame to torch float32 CHW (1,H,W)."""
    a = frame.astype(np.float32)
    # Accept (H,W) or (H,W,1) or (T,H,W) already indexed before.
    if a.ndim == 3 and a.shape[-1] == 1:
        a = a[..., 0]
    if a.ndim != 2:
        raise ValueError(f"Expected a single 2D frame (H,W), got shape={a.shape}")
    a = np.ascontiguousarray(a)
    return torch.from_numpy(a).unsqueeze(0)  # (1,H,W)


def _tensor_to_rgb_u8(t: torch.Tensor) -> np.ndarray:
    """Convert (C,H,W) tensor to RGB uint8 image for visualization."""
    x = t.detach().cpu()
    if x.ndim != 3:
        raise ValueError(f"Expected (C,H,W), got {tuple(x.shape)}")
    if x.shape[0] > 1:
        x = x[:1]
    a = x[0].numpy().astype(np.float32)
    mn, mx = float(a.min()), float(a.max())
    if mx > mn:
        a = (a - mn) / (mx - mn)
    a = (np.clip(a, 0, 1) * 255.0).astype(np.uint8)
    return np.stack([a, a, a], axis=-1)


def load_some_from_h5(h5_path: Path, n: int) -> List[Tuple[torch.Tensor, int]]:
    """Load the first `n` frames from an H5 file as torch tensors."""
    with h5py.File(str(h5_path), "r") as f:
        key = _find_frames_key(f)
        ds = f[key]
        n = min(n, int(ds.shape[0]))
        out: List[Tuple[torch.Tensor, int]] = []
        for i in range(n):
            frame = ds[i]  # expected (H,W) or (H,W,1)
            out.append((_to_chw_float32(frame), i))
        return out


def predict(model, items: List[Tuple[torch.Tensor, int]], set_name: str) -> List[Dict[str, Any]]:
    """Run model inference on a list of (tensor, frame_index)."""
    model.eval()
    X = [t for (t, _) in items]
    with torch.no_grad():
        y_pred = model(X)  # list[dict] with boxes/scores

    preds: List[Dict[str, Any]] = []
    for (t, fi), p in zip(items, y_pred):
        boxes = p.get("boxes", torch.zeros((0, 4)))
        scores = p.get("scores", torch.zeros((boxes.shape[0],)))
        cnn_probs = p.get("cnn_probs", None)
        preds.append(
            dict(
                set_name=set_name,
                frame_index=int(fi),
                img_rgb=_tensor_to_rgb_u8(t),
                boxes=boxes.detach().cpu().numpy().astype(np.float32) if isinstance(boxes, torch.Tensor) else np.asarray(boxes, np.float32),
                scores=scores.detach().cpu().numpy().astype(np.float32) if isinstance(scores, torch.Tensor) else np.asarray(scores, np.float32),
                cnn_probs=cnn_probs.detach().cpu().numpy().astype(np.float32) if isinstance(cnn_probs, torch.Tensor) else np.zeros((boxes.shape[0],), dtype=np.float32),
            )
        )
    return preds


def save_mosaic(preds: List[Dict[str, Any]], out_png: Path) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.reshape(-1)
    for ax in axes:
        ax.axis("off")

    for i, item in enumerate(preds[:8]):
        img = item["img_rgb"].copy()
        gray_u8 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        boxes = item["boxes"]
        scores = item["scores"]
        cnn_probs = item.get("cnn_probs", None)
        H, W = gray_u8.shape

        for k in range(boxes.shape[0]):
            x1, y1, x2, y2 = boxes[k].astype(int).tolist()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # YOLO score — top-left, green
            yolo_sc = float(scores[k]) if k < scores.size else 0.0
            cv2.putText(img, f"{yolo_sc:.2f}", (x1, min(H - 2, y1 + 12)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)

            # Blackness — bottom-right above CNN prob, orange
            x1i = max(0, min(int(round(boxes[k][0])), W))
            x2i = max(0, min(int(round(boxes[k][2])), W))
            y1i = max(0, min(int(round(boxes[k][1])), H))
            y2i = max(0, min(int(round(boxes[k][3])), H))
            roi = gray_u8[y1i:y2i, x1i:x2i]
            blackness = 255.0 - float(roi.mean()) if roi.size else 0.0
            blk_label = f"b{blackness:.0f}"
            (bw, bh), _ = cv2.getTextSize(blk_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            bx = max(0, x2 - bw - 2)

            # CNN prob — bottom-right, red
            if cnn_probs is not None and k < cnn_probs.size:
                cnn_sc = float(cnn_probs[k])
                cnn_label = f"{cnn_sc:.2f}"
                (tw, th), _ = cv2.getTextSize(cnn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                tx = max(0, x2 - tw - 2)
                ty = min(H - 2, y2 - 2)
                ty = max(th + 1, ty)
                cv2.putText(img, cnn_label, (tx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 1, cv2.LINE_AA)
                # Blackness just above CNN prob
                by = max(bh + 1, ty - th - 2)
            else:
                by = min(H - 2, y2 - 2)
                by = max(bh + 1, by)

            cv2.putText(img, blk_label, (bx, by),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

        ax = axes[i]
        ax.imshow(img)
        n_boxes = int(boxes.shape[0]) if isinstance(boxes, np.ndarray) else int(len(boxes))
        ax.set_title(f"{item['set_name']} | frame {item['frame_index']} | boxes={n_boxes}", fontsize=10)

    fig.suptitle("Tokam2D predictions: test (top) + private_test (bottom)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    """Train model, run inference on both H5 test files, save a PNG mosaic."""
    from submission import train_model  # your local submission.py

    root = Path(".")
    train_dir = root / "train"

    print("[INFO] Training model...")
    model = train_model(train_dir)

    print("[INFO] Loading frames...")
    test_items = load_some_from_h5(root / "test" / "test.h5", N_PER_SET)
    priv_items = load_some_from_h5(root / "private_test" / "private_test.h5", N_PER_SET)

    print("[INFO] Predicting...")
    preds = []
    preds += predict(model, test_items, "test")
    preds += predict(model, priv_items, "private_test")

    save_mosaic(preds, OUT_PNG)
    print(f"[OK] Saved mosaic to: {OUT_PNG.resolve()}")

    # Save raw preds to a pickle for later analysis
    out_pkl = Path("preds.pkl")
    with open(out_pkl, "wb") as f:
        pickle.dump(preds, f)
    print(f"[OK] Saved preds to: {out_pkl.resolve()}")

if __name__ == "__main__":
    main()