# Swin Faster RCNN for Nodule Detection

This repository contains the Swin Transformer-based Faster RCNN implementation used for pulmonary nodule detection on chest X-rays. The code focuses on preparing the data pipeline, defining the model, training, and visualizing predictions.

## Features

- Swin Transformer backbone + FPN + Faster R-CNN for single-class nodule localization.
- IoU-target shift for training/eval (0.40 pos / 0.30 neg) to match FROC/mAP@0.40 reporting.
- Built‑in FROC (FPPI targets: 0, 0.125, 0.25, 0.5, 1, 2), AUROC, and COCO mAP@0.40.
- Visualization script to overlay predictions and GT, with optional multi-nodule prioritization.

## Setup

1) Create/activate a Python 3.10+ env (example):
```bash
python -m venv .venv
.\\.venv\\Scripts\\activate   # on Windows
```
2) Install deps:
```bash
pip install -r requirements.txt
```
3) Put your COCO JSONs and images where `data.py` expects them (defaults in `proccessed_data/`).

## Training

Example full run (adjust paths for your machine):
```bash
python train.py \
  --train-ann ../proccessed_data/Node21_Nodule_Bbox_NAD_train_3.json \
  --val-ann   ../proccessed_data/Node21_Nodule_Bbox_NAD_test_3.json \
  --img-root  ../proccessed_data/images \
  --batch-size 2 --workers 8 \
  --epochs 24 \
  --iou-thr 0.4 \
  --output ../work_dirs/swin_fpn
```
Notes:
- Default inference thresholds during training: `box_score_thresh=0.40`, `box_detections_per_img=5`, NMS 0.4 (set in `model.py`).
- Log file lands in `--output/log.jsonl`; checkpoints saved each epoch plus `best.pth` by FROC@0.5.

## Evaluation (FROC/AUROC + mAP@0.40)

After training:
```bash
python plot_eval.py \
  --ckpt ../work_dirs/swin_fpn/best.pth \
  --ann  ../proccessed_data/Node21_Nodule_Bbox_NAD_test_3.json \
  --img-root ../proccessed_data/images \
  --iou-thr 0.4 \
  --out-dir ../6th\\ Dec\\ SOL\\ 50\\ epoch/plots_mAP40_tight
```
Outputs:
- `metrics.json` with AUROC, FROC@{0…2}, FROC AUC, and `mAP40`.
- `froc.png`, `roc.png` plots.

## Visualization

Overlay predictions vs GT on positive images:
```bash
python visualize_preds.py \
  --ckpt ../work_dirs/swin_fpn/best.pth \
  --ann  ../proccessed_data/Node21_Nodule_Bbox_NAD_test_3.json \
  --img-root ../proccessed_data/images \
  --score-thr 0.40 \
  --num 12 \
  --prefer-multi \
  --out-dir ../6th\\ Dec\\ SOL\\ 50\\ epoch/vis_val_thresh04
```
- Green = predictions (filtered at `score_thr`), Red = ground truth. `--prefer-multi` samples images with multiple nodules first.

## Dataset

The repo targets the Node21 chest X-ray nodule set (COCO-style). Make sure your JSONs carry `images`, `annotations` (bbox `[x,y,w,h]`, `category_id=1`), and `file_name` paths relative to `--img-root`.

## Tips
- If early FROC stays low, reduce `box_score_thresh` during training (e.g., 0.05) in `model.py`, then tighten at inference.
- For quick smoke tests, subsample COCO JSONs and run 1–3 epochs with `--batch-size 1 --workers 0`.
