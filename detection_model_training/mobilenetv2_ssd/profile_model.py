from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys

import torch
from torch import nn
from torchprofile import profile_macs

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from detection_model_training.mobilenetv2_ssd.mobilenetv2_ssd import mobilenet_v2_ssd


class DetectorCore(nn.Module):
    """Profile only backbone + SSD head, excluding anchors and post-processing."""

    def __init__(self, detector: nn.Module) -> None:
        super().__init__()
        self.backbone = detector.backbone
        self.head = detector.head

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        features = list(features.values())
        outputs = self.head(features)
        return outputs["cls_logits"], outputs["bbox_regression"]


def main() -> None:
    model = mobilenet_v2_ssd(
        num_classes=2,
        size=(320, 240),
        pretrained_backbone=False,
    ).eval()

    total_params = sum(parameter.numel() for parameter in model.parameters())
    trainable_params = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    core = DetectorCore(model).eval()
    sample = torch.randn(1, 3, 240, 320)

    with torch.no_grad():
        macs = profile_macs(core, args=(sample,))

    print("MobilenetV2SSD @ 320x240")
    print(f"Total params: {total_params:,}")
    print(f"Trainable params: {trainable_params:,}")
    print(f"Detector-core MACs: {macs:,}")


if __name__ == "__main__":
    main()
