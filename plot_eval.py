import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from data import NoduleDataset, collate_fn
from model import build_detector


def box_iou_single(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float32)
    xx1 = np.maximum(box[0], boxes[:, 0])
    yy1 = np.maximum(box[1], boxes[:, 1])
    xx2 = np.minimum(box[2], boxes[:, 2])
    yy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter
    return inter / np.maximum(union, 1e-6)


def compute_froc_curve(
    preds: List[Dict[str, np.ndarray]],
    gts: List[np.ndarray],
    fppi_targets: Sequence[float] = (0, 0.125, 0.25, 0.5, 1, 2),
    iou_thr: float = 0.5,
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    num_images = len(gts)
    num_gt = sum(len(b) for b in gts)
    if num_images == 0 or num_gt == 0:
        metrics = {f"froc@{t}": 0.0 for t in fppi_targets}
        metrics["froc_auc"] = 0.0
        return metrics, np.array([0.0]), np.array([0.0])

    dets: List[Tuple[float, int, np.ndarray]] = []
    for img_idx, pred in enumerate(preds):
        boxes = pred["boxes"]
        scores = pred["scores"]
        for b, s in zip(boxes, scores):
            dets.append((float(s), img_idx, b))
    dets.sort(key=lambda x: x[0], reverse=True)

    matched = [np.zeros(len(b), dtype=bool) for b in gts]
    tp_list, fp_list = [], []
    tp = fp = 0
    for _, img_idx, box in dets:
        gt_boxes = gts[img_idx]
        hit = False
        if gt_boxes.size > 0:
            ious = box_iou_single(box, gt_boxes)
            j = int(np.argmax(ious))
            if ious[j] >= iou_thr and not matched[img_idx][j]:
                matched[img_idx][j] = True
                hit = True
        if hit:
            tp += 1
        else:
            fp += 1
        tp_list.append(tp)
        fp_list.append(fp)

    fppi_curve = np.concatenate([[0.0], np.array(fp_list, dtype=float) / float(num_images)])
    sens_curve = np.concatenate([[0.0], np.array(tp_list, dtype=float) / float(num_gt)])

    max_fppi = max(fppi_targets) if len(fppi_targets) > 0 else fppi_curve[-1]
    target_sens = np.interp(fppi_targets, fppi_curve, sens_curve, left=0, right=sens_curve[-1])

    x = fppi_curve
    y = sens_curve
    if max_fppi > x[-1]:
        x = np.concatenate([x, [max_fppi]])
        y = np.concatenate([y, [y[-1]]])
    mask = x <= max_fppi
    froc_auc = float(np.trapz(y[mask], x[mask]) / max(max_fppi, 1e-8))

    metrics = {f"froc@{t}": float(s) for t, s in zip(fppi_targets, target_sens)}
    metrics["froc_auc"] = froc_auc
    return metrics, fppi_curve, sens_curve


def image_roc_curve(preds: List[Dict[str, np.ndarray]], gts: List[np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
    scores = []
    labels = []
    for pred, gt in zip(preds, gts):
        scores.append(float(pred["scores"].max()) if pred["scores"].size > 0 else 0.0)
        labels.append(1 if len(gt) > 0 else 0)
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return 0.5, np.array([0.0, 1.0]), np.array([0.0, 1.0])

    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / float(pos)
    fpr = fp / float(neg)
    tpr_curve = np.concatenate([[0.0], tpr, [1.0]])
    fpr_curve = np.concatenate([[0.0], fpr, [1.0]])
    auroc = float(np.trapz(tpr_curve, fpr_curve))
    return auroc, fpr_curve, tpr_curve


@torch.no_grad()
def gather_predictions(model, loader, device):
    model.eval()
    preds: List[Dict[str, np.ndarray]] = []
    gts: List[np.ndarray] = []
    coco_results = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            preds.append({"boxes": out["boxes"].cpu().numpy(), "scores": out["scores"].cpu().numpy()})
            gts.append(tgt["boxes"].cpu().numpy())
            img_id = int(tgt["image_id"].item())
            for box, score in zip(out["boxes"].cpu().numpy(), out["scores"].cpu().numpy()):
                x1, y1, x2, y2 = box.tolist()
                coco_results.append(
                    {
                        "image_id": img_id,
                        "category_id": 1,
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(score),
                    }
                )
    return preds, gts, coco_results
    return preds, gts


def plot_froc(fppi_curve: np.ndarray, sens_curve: np.ndarray, fppi_targets: Iterable[float], out_path: Path):
    plt.figure()
    plt.plot(fppi_curve, sens_curve, label="FROC")
    plt.scatter(list(fppi_targets), np.interp(list(fppi_targets), fppi_curve, sens_curve, left=0, right=sens_curve[-1]), color="red", marker="x", label="targets")
    plt.xlabel("False positives per image")
    plt.ylabel("Sensitivity")
    plt.grid(True)
    plt.xlim(0, max(fppi_targets))
    plt.ylim(0, 1.05)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_roc(fpr: np.ndarray, tpr: np.ndarray, out_path: Path):
    plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.grid(True)
    plt.xlim(0, 1)
    plt.ylim(0, 1.05)
    plt.legend()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate and plot FROC/ROC curves.")
    parser.add_argument("--ckpt", required=True, help="Path to checkpoint (.pth)")
    parser.add_argument("--ann", required=True, help="Annotation file (COCO JSON)")
    parser.add_argument("--img-root", required=True, help="Image root directory")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iou-thr", type=float, default=0.5)
    parser.add_argument("--out-dir", default="eval_plots")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_detector(num_classes=2, pretrained_backbone=False)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device)

    ds = NoduleDataset(args.ann, args.img_root, train=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, collate_fn=collate_fn)

    preds, gts, coco_results = gather_predictions(model, loader, device)
    fppi_targets = (0, 0.125, 0.25, 0.5, 1, 2)
    froc_metrics, fppi_curve, sens_curve = compute_froc_curve(preds, gts, fppi_targets=fppi_targets, iou_thr=args.iou_thr)
    auroc, fpr_curve, tpr_curve = image_roc_curve(preds, gts)

    out_dir = Path(args.out_dir)
    plot_froc(fppi_curve, sens_curve, fppi_targets, out_dir / "froc.png")
    plot_roc(fpr_curve, tpr_curve, out_dir / "roc.png")

    summary = {"auroc": auroc, **froc_metrics}

    # COCO mAP at IoU 0.40 (single class)
    coco = COCO(args.ann)
    if len(coco_results) == 0:
        summary["mAP40"] = 0.0
    else:
        coco_dt = coco.loadRes(coco_results)
        coco_eval = COCOeval(coco, coco_dt, iouType="bbox")
        coco_eval.params.iouThrs = np.array([0.40])
        coco_eval.params.catIds = [1]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        # stats[0] is AP at IoU=0.5:0.95, but with single IoU it matches 0.40
        summary["mAP40"] = float(coco_eval.stats[0])

    print(json.dumps(summary, indent=2))
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
