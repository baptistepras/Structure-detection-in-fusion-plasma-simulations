#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare two saved preds.pkl files and visualize differences.

- Orange: boxes present in file1 but not in file2
- Blue:   boxes present in file2 but not in file1

Usage:
  python compare_preds.py runA runB
This will load runA.pkl and runB.pkl and save a mosaic PNG.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def load_preds(pkl_stem: str) -> List[Dict[str, Any]]:
    p = Path(pkl_stem).with_suffix(".pkl")
    with open(p, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, list):
        raise TypeError(f"{p}: expected a list, got {type(obj)}")
    return obj


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> float:
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


def greedy_match_by_iou(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
    iou_thr: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy one-to-one matching by IoU.
    Returns:
      - unmatched indices in A
      - unmatched indices in B
    """
    na = int(boxes_a.shape[0])
    nb = int(boxes_b.shape[0])
    if na == 0 and nb == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    if na == 0:
        return np.zeros((0,), dtype=int), np.arange(nb, dtype=int)
    if nb == 0:
        return np.arange(na, dtype=int), np.zeros((0,), dtype=int)

    pairs: List[Tuple[float, int, int]] = []
    for i in range(na):
        for j in range(nb):
            v = iou_xyxy(boxes_a[i], boxes_b[j])
            if v >= iou_thr:
                pairs.append((v, i, j))
    pairs.sort(reverse=True, key=lambda t: t[0])

    matched_a = np.zeros((na,), dtype=bool)
    matched_b = np.zeros((nb,), dtype=bool)

    for v, i, j in pairs:
        if not matched_a[i] and not matched_b[j]:
            matched_a[i] = True
            matched_b[j] = True

    un_a = np.where(~matched_a)[0]
    un_b = np.where(~matched_b)[0]
    return un_a, un_b


def _put_text(
    img_rgb: np.ndarray,
    text: str,
    org: Tuple[int, int],
    bgr: Tuple[int, int, int],
    scale: float = 0.45,
    thickness: int = 1,
) -> None:
    cv2.putText(
        img_rgb,
        text,
        org,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        bgr,
        thickness,
        cv2.LINE_AA,
    )


def draw_boxes_with_scores(
    img_rgb: np.ndarray,
    boxes: np.ndarray,
    scores: np.ndarray,
    cnn_probs: np.ndarray | None,
    color_bgr: Tuple[int, int, int],
) -> None:
    """
    Draw boxes and show YOLO score near top-left (inside box).
    If cnn_probs is provided, show CNN prob in red near bottom-right (inside box).
    """
    H, W = img_rgb.shape[:2]
    for k in range(int(boxes.shape[0])):
        x1, y1, x2, y2 = boxes[k].astype(int).tolist()
        x1 = max(0, min(x1, W - 1))
        y1 = max(0, min(y1, H - 1))
        x2 = max(0, min(x2, W - 1))
        y2 = max(0, min(y2, H - 1))

        cv2.rectangle(img_rgb, (x1, y1), (x2, y2), color_bgr, 2)

        sc = float(scores[k]) if k < scores.size else 0.0
        # score (same color as box), top-left inside
        _put_text(img_rgb, f"{sc:.2f}", (x1, min(H - 2, y1 + 12)), color_bgr)

        if cnn_probs is not None and k < cnn_probs.size:
            cp = float(cnn_probs[k])
            label = f"{cp:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tx = max(0, x2 - tw - 2)
            ty = min(H - 2, y2 - 2)
            ty = max(th + 1, ty)
            # CNN prob in red bottom-right inside
            _put_text(img_rgb, label, (tx, ty), (255, 0, 0))


def save_diff_mosaic(
    preds1: List[Dict[str, Any]],
    preds2: List[Dict[str, Any]],
    name1: str,
    name2: str,
    out_png: Path,
    iou_thr: float = 0.50,
) -> None:
    # Expect 8 = 4 test + 4 private_test (like your runner)
    n = min(len(preds1), len(preds2), 8)
    if n < 1:
        raise ValueError("No items to display (empty preds).")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.reshape(-1)
    for ax in axes:
        ax.axis("off")

    # Colors 
    ORANGE = (255, 165, 0)
    GREEN = (0, 255, 0)

    for i in range(8):
        if i >= n:
            axes[i].set_visible(False)
            continue

        a = preds1[i]
        b = preds2[i]

        img = np.array(a["img_rgb"], copy=True)  # base image from file1
        boxes_a = np.asarray(a.get("boxes", np.zeros((0, 4), np.float32)), np.float32)
        scores_a = np.asarray(a.get("scores", np.zeros((boxes_a.shape[0],), np.float32)), np.float32)
        cnn_a = a.get("cnn_probs", None)
        cnn_a = None if cnn_a is None else np.asarray(cnn_a, np.float32)

        boxes_b = np.asarray(b.get("boxes", np.zeros((0, 4), np.float32)), np.float32)
        scores_b = np.asarray(b.get("scores", np.zeros((boxes_b.shape[0],), np.float32)), np.float32)
        cnn_b = b.get("cnn_probs", None)
        cnn_b = None if cnn_b is None else np.asarray(cnn_b, np.float32)

        # Find differences
        un_a, un_b = greedy_match_by_iou(boxes_a, boxes_b, iou_thr=iou_thr)

        # Draw unique boxes
        if un_a.size > 0:
            draw_boxes_with_scores(
                img_rgb=img,
                boxes=boxes_a[un_a],
                scores=scores_a[un_a] if scores_a.size == boxes_a.shape[0] else scores_a,
                cnn_probs=(cnn_a[un_a] if (cnn_a is not None and cnn_a.size == boxes_a.shape[0]) else None),
                color_bgr=ORANGE,
            )

        if un_b.size > 0:
            draw_boxes_with_scores(
                img_rgb=img,
                boxes=boxes_b[un_b],
                scores=scores_b[un_b] if scores_b.size == boxes_b.shape[0] else scores_b,
                cnn_probs=(cnn_b[un_b] if (cnn_b is not None and cnn_b.size == boxes_b.shape[0]) else None),
                color_bgr=GREEN,
            )

        set_name = a.get("set_name", "?")
        frame_index = a.get("frame_index", "?")
        axes[i].imshow(img)
        axes[i].set_title(
            f"{set_name} | frame {frame_index} | onlyA={int(un_a.size)} onlyB={int(un_b.size)}",
            fontsize=10,
        )

    fig.suptitle("Tokam2D diff: boxes unique to each run", fontsize=14)

    # Legend text (colored)
    fig.text(
        0.01, 0.02, f"{Path(name1).with_suffix('.pkl').name} (only)",
        color="orange", fontsize=12, ha="left", va="bottom",
    )
    fig.text(
        0.01, 0.045, f"{Path(name2).with_suffix('.pkl').name} (only)",
        color="green", fontsize=12, ha="left", va="bottom",
    )

    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("file1", help="First preds filename stem (without .pkl) or path without extension")
    ap.add_argument("file2", help="Second preds filename stem (without .pkl) or path without extension")
    ap.add_argument("--iou", type=float, default=0.50, help="IoU threshold to consider boxes identical")
    ap.add_argument("--out", type=str, default="", help="Output PNG path (default: diff_<file1>__<file2>.png)")
    args = ap.parse_args()

    preds1 = load_preds(args.file1)
    preds2 = load_preds(args.file2)

    out_png = Path(args.out) if args.out else Path(f"diff_{Path(args.file1).stem}__{Path(args.file2).stem}.png")

    save_diff_mosaic(
        preds1=preds1,
        preds2=preds2,
        name1=args.file1,
        name2=args.file2,
        out_png=out_png,
        iou_thr=float(args.iou),
    )

    print(f"[OK] Saved diff mosaic to: {out_png.resolve()}")


if __name__ == "__main__":
    main()