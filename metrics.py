from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


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


def compute_froc(
    preds: List[Dict[str, np.ndarray]],
    gts: List[np.ndarray],
    fppi_targets: Sequence[float] = (0, 0.125, 0.25, 0.5, 1, 2),
    iou_thr: float = 0.5,
) -> Dict[str, float]:
    """Compute FROC sensitivity at given FPPI targets and the area under FROC up to max FPPI."""
    num_images = len(gts)
    num_gt = sum(len(b) for b in gts)
    if num_images == 0 or num_gt == 0:
        return {f"froc@{t}": 0.0 for t in fppi_targets} | {"froc_auc": 0.0}

    # Flatten detections
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
    for score, img_idx, box in dets:
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

    # curves
    fppi_curve = np.concatenate([[0.0], np.array(fp_list, dtype=float) / float(num_images)])
    sens_curve = np.concatenate([[0.0], np.array(tp_list, dtype=float) / float(num_gt)])

    max_fppi = max(fppi_targets) if len(fppi_targets) > 0 else fppi_curve[-1]
    target_sens = np.interp(fppi_targets, fppi_curve, sens_curve, left=0, right=sens_curve[-1])

    # area under FROC up to max_fppi
    x = fppi_curve
    y = sens_curve
    if max_fppi > x[-1]:
        x = np.concatenate([x, [max_fppi]])
        y = np.concatenate([y, [y[-1]]])
    mask = x <= max_fppi
    froc_auc = float(np.trapz(y[mask], x[mask]) / max(max_fppi, 1e-8))

    metrics = {f"froc@{t}": float(s) for t, s in zip(fppi_targets, target_sens)}
    metrics["froc_auc"] = froc_auc
    return metrics


def compute_image_auroc(preds: List[Dict[str, np.ndarray]], gts: List[np.ndarray]) -> float:
    """AUROC at image level using max score per image (0 if none)."""
    scores = []
    labels = []
    for pred, gt in zip(preds, gts):
        scores.append(float(pred["scores"].max()) if pred["scores"].size > 0 else 0.0)
        labels.append(1 if len(gt) > 0 else 0)
    scores = np.array(scores, dtype=float)
    labels = np.array(labels, dtype=int)
    pos = (labels == 1).sum()
    neg = (labels == 0).sum()
    if pos == 0 or neg == 0:
        return 0.5
    order = np.argsort(scores)[::-1]
    labels_sorted = labels[order]
    tp = np.cumsum(labels_sorted)
    fp = np.cumsum(1 - labels_sorted)
    tpr = tp / float(pos)
    fpr = fp / float(neg)
    tpr_curve = np.concatenate([[0.0], tpr, [1.0]])
    fpr_curve = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapz(tpr_curve, fpr_curve))
