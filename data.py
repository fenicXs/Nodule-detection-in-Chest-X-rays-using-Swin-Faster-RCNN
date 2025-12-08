import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import albumentations as A
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def _random_resized_crop(size: int = 1024, scale=(0.8, 1.0), ratio=(0.9, 1.1), p: float = 0.7):
    """Create RandomResizedCrop compatible with both albumentations 1.x (height/width)
    and 2.x (size) signatures."""
    try:
        return A.RandomResizedCrop(size=(size, size), scale=scale, ratio=ratio, p=p)
    except TypeError:
        return A.RandomResizedCrop(height=size, width=size, scale=scale, ratio=ratio, p=p)


def build_transforms(train: bool = True) -> A.BasicTransform:
    """Albumentations pipeline tuned for 1024x1024 CXRs with small nodules."""
    if train:
        return A.Compose(
            [
                _random_resized_crop(),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=7, p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.GaussNoise(var_limit=(5.0, 25.0), p=0.2),
            ],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0),
        )
    else:
        return A.Compose(
            [A.LongestMaxSize(max_size=1024), A.PadIfNeeded(min_height=1024, min_width=1024, border_mode=cv2.BORDER_CONSTANT)],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["class_labels"], min_visibility=0.0),
        )


class NoduleDataset(Dataset):
    """COCO-style dataset loader for single-class pulmonary nodules."""

    def __init__(self, ann_file: str, img_root: str, train: bool = True):
        self.ann_file = Path(ann_file)
        self.img_root = Path(img_root)
        with open(self.ann_file, "r") as f:
            coco = json.load(f)
        self.images = {img["id"]: img for img in coco["images"]}
        self.anns_by_img: Dict[int, List[Dict[str, Any]]] = {img_id: [] for img_id in self.images}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", False):
                continue
            self.anns_by_img[ann["image_id"]].append(ann)
        self.ids = list(self.images.keys())
        self.transforms = build_transforms(train=train)

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_id = self.ids[idx]
        info = self.images[img_id]
        img_path = self.img_root / info["file_name"]
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        anns = self.anns_by_img.get(img_id, [])
        bboxes = []
        labels = []
        areas = []
        for ann in anns:
            x, y, w, h = ann["bbox"]
            bboxes.append([x, y, x + w, y + h])
            labels.append(1)  # single foreground class
            areas.append(ann.get("area", w * h))

        if len(bboxes) == 0:
            bboxes = []
            labels = []
            areas = []

        transformed = self.transforms(image=img, bboxes=bboxes, class_labels=labels)
        img_t = torch.from_numpy(transformed["image"].transpose(2, 0, 1)).float() / 255.0

        boxes_np = np.array(transformed["bboxes"], dtype=np.float32) if transformed["bboxes"] else np.zeros((0, 4), dtype=np.float32)
        target: Dict[str, Any] = {
            "boxes": torch.as_tensor(boxes_np, dtype=torch.float32),
            "labels": torch.as_tensor(transformed["class_labels"], dtype=torch.int64) if transformed["class_labels"] else torch.zeros((0,), dtype=torch.int64),
            "image_id": torch.tensor([img_id]),
            "area": torch.as_tensor(areas if areas else np.zeros((0,), dtype=np.float32)),
            "iscrowd": torch.zeros((len(transformed["class_labels"]),), dtype=torch.int64) if transformed["class_labels"] else torch.zeros((0,), dtype=torch.int64),
        }
        return img_t, target


def collate_fn(batch):
    imgs, targets = zip(*batch)
    return list(imgs), list(targets)
