import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import torch

from data import NoduleDataset
from model import build_detector


def draw_boxes(img: np.ndarray, boxes: np.ndarray, color, scores=None, score_thr: float = 0.3, label: str = ""):
    img = img.copy()
    for idx, box in enumerate(boxes):
        score_ok = True
        if scores is not None:
            score = scores[idx]
            if score < score_thr:
                continue
        else:
            score = None
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f"{label}{score:.2f}" if score is not None else label
        if text:
            cv2.putText(img, text, (x1, max(0, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Visualize predictions vs ground truth on positive cases.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--ann", required=True, help="Annotation JSON")
    parser.add_argument("--img-root", required=True, help="Image root directory")
    parser.add_argument("--num", type=int, default=8, help="Number of positive images to visualize")
    parser.add_argument("--score-thr", type=float, default=0.3, help="Score threshold for drawing boxes")
    parser.add_argument("--out-dir", default="vis_preds", help="Output directory for visualizations")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling positives")
    parser.add_argument("--prefer-multi", action="store_true", help="Prioritize images with >=2 GT boxes")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_detector(num_classes=2, pretrained_backbone=False)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    ds = NoduleDataset(args.ann, args.img_root, train=False)
    # collect indices with at least one GT box
    pos_indices = [i for i in range(len(ds)) if len(ds[i][1]["boxes"]) > 0]
    if args.prefer_multi:
        multi = [i for i in pos_indices if len(ds[i][1]["boxes"]) >= 2]
        single = [i for i in pos_indices if len(ds[i][1]["boxes"]) == 1]
        rng.shuffle(multi)
        rng.shuffle(single)
        selected = (multi + single)[: min(args.num, len(pos_indices))]
    else:
        rng.shuffle(pos_indices)
        selected = pos_indices[: min(args.num, len(pos_indices))]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for idx in selected:
        img_t, tgt = ds[idx]
        img = (img_t.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        pred = model([img_t.to(device)])[0]
        boxes = pred["boxes"].cpu().numpy()
        scores = pred["scores"].cpu().numpy()
        # Draw predictions in green and GT in red
        vis = draw_boxes(img, boxes, color=(0, 200, 0), scores=scores, score_thr=args.score_thr, label="")
        vis = draw_boxes(vis, tgt["boxes"].cpu().numpy(), color=(0, 0, 255), scores=None, label="GT")
        out_path = out_dir / f"vis_{idx:04d}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))


if __name__ == "__main__":
    main()
