import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import NoduleDataset, collate_fn
from metrics import compute_froc, compute_image_auroc
from model import build_detector


def parse_args():
    parser = argparse.ArgumentParser(description="Swin-FPN Faster R-CNN for Nodule Localization with FROC/AUROC")
    parser.add_argument("--train-ann", default="../proccessed_data/Node21_Nodule_Bbox_NAD_train_3.json")
    parser.add_argument("--val-ann", default="../proccessed_data/Node21_Nodule_Bbox_NAD_test_3.json")
    parser.add_argument("--img-root", default="../proccessed_data/images")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="work_dirs/swin_fpn")
    parser.add_argument("--iou-thr", type=float, default=0.5)
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, loader, device, iou_thr: float):
    model.eval()
    preds: List[Dict[str, np.ndarray]] = []
    gts: List[np.ndarray] = []
    for images, targets in tqdm(loader, desc="Val", leave=False):
        images = [img.to(device) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            preds.append(
                {
                    "boxes": out["boxes"].cpu().numpy(),
                    "scores": out["scores"].cpu().numpy(),
                }
            )
            gts.append(tgt["boxes"].cpu().numpy())

    froc = compute_froc(preds, gts, iou_thr=iou_thr)
    auroc = compute_image_auroc(preds, gts)
    return froc, auroc


def train_one_epoch(model, loader, optimizer, device, scaler):
    model.train()
    total_loss = 0.0
    for images, targets in tqdm(loader, desc="Train", leave=False):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        total_loss += loss.item() * len(images)
    return total_loss / len(loader.dataset)


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = NoduleDataset(args.train_ann, args.img_root, train=True)
    val_ds = NoduleDataset(args.val_ann, args.img_root, train=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    model = build_detector(num_classes=2, pretrained_backbone=True).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    best_froc = -1.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        froc, auroc = evaluate(model, val_loader, device, iou_thr=args.iou_thr)
        froc_at_05 = froc.get("froc@0.5", 0.0)

        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "auroc": auroc,
            **froc,
        }
        with open(output_dir / "log.jsonl", "a") as f:
            f.write(json.dumps(log) + "\n")

        print(f"[Epoch {epoch}] loss={train_loss:.4f} auroc={auroc:.4f} froc@0.5={froc_at_05:.4f} froc_auc={froc['froc_auc']:.4f}")

        ckpt_path = output_dir / f"epoch_{epoch}.pth"
        torch.save({"model": model.state_dict(), "epoch": epoch, "froc@0.5": froc_at_05}, ckpt_path)
        if froc_at_05 > best_froc:
            best_froc = froc_at_05
            torch.save({"model": model.state_dict(), "epoch": epoch, "froc@0.5": froc_at_05}, output_dir / "best.pth")


if __name__ == "__main__":
    main()
