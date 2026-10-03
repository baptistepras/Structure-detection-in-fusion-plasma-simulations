# Implementation

How the pipeline works and what each file does. Commands are in [usage.md](usage.md).

## Data

The challenge provides three sets of density videos in H5 files:

- `blob_dwi`: a small labeled set (boxes in Pascal VOC XML files);
- `turb_i`: unlabeled frames, used for pseudo-labeling;
- `blob_i`: a second labeled set, used only to train the post-filter.

## Training

Everything happens in `train_model()` in `submission.py`.

1. **Conversion.** Frames are exported to grayscale PNG images, and the XML boxes are converted to YOLO labels.
2. **YOLO, round 1.** YOLOv8n is trained on `blob_dwi` with light augmentation.
3. **MLP scorer.** For each ground truth box, jittered copies give positives and negatives according to their IoU. Each box is described by hand-crafted features: geometry, intensity statistics, Sobel gradients, ring contrast around the box, and a small resized patch. A small MLP learns to separate them (BCE loss, early stopping).
4. **Pseudo-labels.** The round 1 detector runs on `turb_i`. The MLP scores every candidate box, and the top-k boxes of each frame become YOLO labels. Top-k worked better than a fixed threshold.
5. **YOLO, round 2.** YOLO is trained again, from the round 1 weights, on `blob_dwi` and the pseudo-labeled `turb_i`.
6. **Post-filter.** The round 2 detector runs on `blob_i`. Its true positives (matched with IoMean) and its missed boxes are positives, its false positives are negatives. A small patch CNN (`BlobCNN`) learns from a 32 x 32 patch of each box and a few physics-inspired features (left and right asymmetry, structure tensor orientation, convexity, aspect ratio, mean intensity).

## Inference

`YOLOWrapper.forward()` runs in three steps.

1. YOLO predicts boxes with a low confidence threshold (0.01) to favor recall.
2. Geometric filters drop right-edge artifacts, boxes that contain other boxes, very dark boxes, and a recurring artifact in the bottom right corner.
3. The CNN vetoes low-confidence boxes on the left side and in the top right and bottom right corners when its probability is below a threshold. High-confidence boxes are always kept.

The wrapper returns boxes, scores, and labels as CPU tensors, plus the CNN probabilities for debugging.

## What did not help

These attempts were tested and left out of the final code:

- larger backbones (YOLOv8s, YOLOv8m, YOLOv8l, YOLOv11n), which gave no gain on so little data;
- self-supervised pretraining (VAE style) and synthetic data (pasting, inpainting, VAE generation);
- the MLP as an inference post-filter, and fusions of the YOLO confidence with the MLP score (average, product, meta-model);
- ridge-map filtering based on structure tensor coherence, as a soft penalty or a hard drop;
- CLAHE, smoothing, other normalizations, and color maps instead of grayscale;
- test-time augmentation with a vertical flip merged by NMS;
- other uses of the three sets, such as training on `blob_i` together with `blob_dwi`.

## Files

| File | Role |
| --- | --- |
| `submission.py` | The whole pipeline in one file, as Codabench requires. `train_model(training_dir)` is the entry point called by Codabench, and `YOLOWrapper` is the model it evaluates. Also holds the conversion, pseudo-labeling, MLP, and CNN utilities. |
| `pred.py` | Trains the pipeline locally, predicts 4 frames of each test set, and saves a mosaic and the raw predictions. |
| `compare.py` | Loads two prediction files, matches their boxes greedily by IoU, and draws only the boxes unique to each run. |
