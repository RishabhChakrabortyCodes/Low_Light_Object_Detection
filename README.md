# Low-Light Object Detection

Building a **robust object detector for low-light scenes**, based on [YOLO](https://docs.ultralytics.com/) (Ultralytics YOLOv8).

Standard detectors are trained on well-lit images and degrade badly in the dark: low contrast, heavy noise and lost detail hide objects, especially small ones. This project measures that degradation on the [ExDark](https://github.com/cs-chan/Exclusively-Dark-Image-Dataset) dataset, sets up reference baselines, and is the base for improving robustness (e.g. low-light enhancement as preprocessing, illumination-aware augmentation, architectural changes) without giving up real-time speed.

## Status

| Stage | State |
|---|---|
| Dataset download, splits, YOLO conversion | Done |
| Dataset EDA (`notebooks/ExDark_EDA.ipynb`) | Done |
| Zero-shot baseline (COCO-pretrained YOLOv8n) | Done |
| Fine-tuned baseline (YOLOv8n trained on ExDark) | Pipeline written, **not yet run** |
| Illumination robustness sweeps, enhancers (e.g. Zero-DCE), speed benchmark | Planned (config keys exist, `scripts/benchmark.py` is empty) |

## Dataset

**ExDark**: 7,363 low-light images, 12 object classes (Bicycle, Boat, Bottle, Bus, Car, Cat, Chair, Cup, Dog, Motorbike, People, Table), 23,710 bounding boxes.

Findings from the EDA notebook:
- Very dark: mean luminance 0.145; 59% of images are below 0.15 and 88% below 0.25.
- Mild class imbalance (max/min = 1.59 in images); People dominates the boxes.
- Boxes: 54% large, 39% medium, 7% small (COCO buckets at 640 px).
- Images vary a lot in size (0.03-14 MP), and 44.5% of boxes have a label different from their image's folder, because images contain several classes.
- **Known leakage:** 18 byte-identical image pairs exist, 12 of which span two splits. This affects at most ~12 test images but should be fixed by deduplicating before splitting.

Splits are stratified by image folder: 70 / 15 / 15 train / validation / test (seed 42), saved as CSVs in `data/outputs/splits/`.

## Setup

Requires Python 3 to run the setup script; [uv](https://docs.astral.sh/uv/) manages the project environment (Python >= 3.12, installed by uv if needed). Works on Linux, macOS and Windows.

```bash
python setup.py
```

This installs dependencies from `pyproject.toml`/`uv.lock` (installing `uv` via pip if missing), downloads ExDark images and annotations from Google Drive, builds the splits and the YOLO-format dataset, and downloads `yolov8n.pt`. Every step is skipped if already done, so it is safe to re-run. About 1.5 GB of images are downloaded.

## Usage

All commands run from the project root.

**Zero-shot baseline**: evaluate COCO-pretrained YOLOv8n on ExDark without training. The 12 ExDark classes are mapped to their COCO ids and mAP is averaged over those 12.

```bash
uv run python scripts/baseline_eval.py                      # test split, CPU
uv run python scripts/baseline_eval.py --split validation --device 0
```

**Fine-tuned baseline**: fine-tune YOLOv8n on the ExDark train split (12-class head), validate each epoch, then evaluate `best.pt` on the test split. Hyperparameters live in `configs/baseline.yaml`.

```bash
uv run python scripts/baseline_train.py --device 0                       # full 100-epoch run (GPU)
uv run python scripts/baseline_train.py --epochs 1 --fraction 0.05 --name smoke --device cpu   # quick check
uv run python scripts/baseline_train.py --eval-only path/to/best.pt      # evaluate existing weights
uv run python scripts/baseline_train.py --resume                         # resume an interrupted run
```

A full run needs a GPU; the project was developed on a CPU-only machine, so only the zero-shot baseline has been run so far.

Results are written to `outputs/<run_name>/` as a metrics JSON (overall and per-class precision, recall, mAP50, mAP50-95, speed) plus PR curves and confusion matrices.

## Results so far

Zero-shot YOLOv8n (COCO weights, no fine-tuning) on the ExDark **test** split (1,105 images, 3,680 boxes), 640 px, CPU:

| Precision | Recall | mAP50 | mAP50-95 | Inference |
|---|---|---|---|---|
| 0.659 | 0.505 | 0.558 | 0.290 | 13.7 ms/img |

Best classes: bus (mAP50 0.82), person (0.73). Worst: dining table (0.28, recall 0.23), boat (0.41), bottle (0.46). Per-class numbers are in `outputs/baseline_v8n_exdark/test_metrics.json`.

The fine-tuned numbers will be added here once that run completes; they are the reference any robustness improvement should beat.

## Project layout

```
setup.py                  one-shot setup (dependencies, data, weights)
configs/baseline.yaml     hyperparameters, augmentation, eval and benchmark settings
scripts/
  download_exdark*.py     fetch images / annotations
  prepare_dataset.py      class list + stratified train/val/test split CSVs
  baseline_eval.py        zero-shot COCO-pretrained evaluation
  baseline_train.py       fine-tune + evaluate pipeline
  benchmark.py            speed benchmark (empty, planned)
src/low_light/data.py     ExDark -> YOLO format conversion (shared by the scripts)
notebooks/ExDark_EDA.ipynb  dataset exploration
outputs/                  metrics and plots from runs
data/                     downloaded and generated data (gitignored)
```

## Notes

- Annotations are in bbGt format (`label l t w h`, pixel coordinates) and converted to normalised YOLO labels by `src/low_light/data.py`.
- `data/yolo_exdark/` (fine-tuning, ids 0-11) and `data/yolo_coco/` (zero-shot, COCO ids) are generated, symlinked/copied views of the raw data.
- Dataset: Loh & Chan, "Getting to Know Low-light Images with The Exclusively Dark Dataset", CVIU 2019.
