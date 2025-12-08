import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from data import NoduleDataset
from model import build_detector


def draw_boxes(img: np.ndarray, boxes: np.ndarray, scores: np.ndarray, score_thr: float = 0.3):
    img = img.copy()
    for box, score in zip(boxes, scores):
        if score < score_thr:
            continue
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(img, f"{score:.2f}", (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize predictions on a few samples.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--ann", required=True, help="Annotation JSON")
    parser.add_argument("--img-root", required=True, help="Image root directory")
    parser.add_argument("--num", type=int, default=8, help="Number of images to visualize")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for drawing boxes")
    parser.add_argument("--out-dir", default="vis_preds", help="Output directory for visualizations")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_detector(num_classes=2, pretrained_backbone=False)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    ds = NoduleDataset(args.ann, args.img_root, train=False)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for i in range(min(args.num, len(ds))):
        img_t, _ = ds[i]
        img = (img_t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        with torch.no_grad():
            pred = model([img_t.to(device)])[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        vis = draw_boxes(img, boxes, scores, score_thr=args.score_thr)
        cv2.imwrite(str(out_dir / f"vis_{i:04d}.png"), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
