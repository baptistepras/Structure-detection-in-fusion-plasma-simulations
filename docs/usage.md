# Usage

Setup, then every command with what it does and its outputs. How the pipeline works is in [implementation.md](implementation.md).

## Setup

The project has its own environment, `plasma-blobs`, defined in [`environment.yml`](../environment.yml).

```bash
mamba env create -f environment.yml          # create the environment once
mamba activate plasma-blobs                  # activate it in every new terminal
mamba env update -f environment.yml --prune  # after environment.yml changes
```

For data confidentiality reasons, the challenge data is not included. It is distributed to the participants of the [Codabench challenge](https://www.codabench.org/competitions/11224/). Place it at the project root as `train/`, `test/test.h5`, and `private_test/private_test.h5`. Every command below runs from the project root.

## Commands

| Command | What it does | Outputs |
| --- | --- | --- |
| `python pred.py` | Trains the full pipeline on `train/`, then predicts 4 frames of `test/test.h5` and 4 frames of `private_test/private_test.h5`. | `mosaic_test_private.png` (predicted boxes in green) and `preds.pkl` (raw predictions). YOLO runs are written to `/tmp/runs`. |
| `python compare.py preds_FINAL preds_NOCNN` | Compares two saved runs, for example the final pipeline and the same detector without the post-filter, saved as `preds_FINAL.pkl` and `preds_NOCNN.pkl`. | `diff_preds_FINAL__preds_NOCNN.png`, showing only the boxes unique to each run. |
