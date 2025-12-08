# Swin Faster RCNN for Nodule Detection

This repository contains the Swin Transformer-based Faster RCNN implementation used for pulmonary nodule detection on chest X-rays. The code focuses on preparing the data pipeline, defining the model, training, and visualizing predictions.

## Features

- Swin Transformer backbone integrated with a Faster R-CNN head for localization.
- Training and evaluation scripts with configurable metrics logging.
- Prediction visualization utilities to inspect bounding boxes and confidence scores.

## Setup

1. Create and activate a Python 3.10+ virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Prepare your data splits and annotations to match the expected format in `data.py`.

## Training and Evaluation

- Use `train.py` to kick off model training; configure logging as needed (e.g., set `work_dirs`).
- Run `plot_eval.py` or `metrics.py` after training to aggregate metrics and visualize performance.
- Use `visualize_preds.py` to overlay detected bounding boxes on sample X-rays.

## Dataset

This work leverages the dataset provided by the [NODE21 Grand Challenge](https://node21.grand-challenge.org/), which offers annotated nodules in chest X-rays for research into localization performance. Please cite the challenge if you publish work based on these annotations.
