from typing import Dict, List

import timm
import torch
from torch import nn
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.backbone_utils import LastLevelMaxPool
from torchvision.ops import FeaturePyramidNetwork, MultiScaleRoIAlign


class TimmBackboneWithFPN(nn.Module):
    """Wrap a timm backbone with FPN for torchvision detectors."""

    def __init__(self, backbone_name: str = "swin_base_patch4_window7_224", pretrained: bool = True):
        super().__init__()
        # Swin produces 4 stages; use all of them (0-3).
        self.body = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
            img_size=1024,
        )
        in_channels_list: List[int] = self.body.feature_info.channels()
        self.fpn = FeaturePyramidNetwork(in_channels_list, out_channels=256, extra_blocks=LastLevelMaxPool())
        self.out_channels = 256

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        feats = self.body(x)
        feats = [f.permute(0, 3, 1, 2) for f in feats]  # NHWC -> NCHW
        feats = {str(i): f for i, f in enumerate(feats)}
        return self.fpn(feats)


def build_detector(num_classes: int = 2, pretrained_backbone: bool = True) -> FasterRCNN:
    """Construct a Swin-FPN Faster R-CNN model.

    Args:
        num_classes: number of classes INCLUDING background. Use 2 for binary detection.
    """
    backbone = TimmBackboneWithFPN(pretrained=pretrained_backbone)
    anchor_generator = AnchorGenerator(
        sizes=((16,), (32,), (64,), (128,), (256,)),
        aspect_ratios=((0.5, 1.0, 2.0),) * 5,
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3", "pool"], output_size=7, sampling_ratio=2)

    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        box_detections_per_img=200,
        image_mean=[0.485, 0.456, 0.406],
        image_std=[0.229, 0.224, 0.225],
        min_size=1024,
        max_size=1024,
    )
    return model
