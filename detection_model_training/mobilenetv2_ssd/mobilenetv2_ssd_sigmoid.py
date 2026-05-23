"""
MobileNetV2 SSD matched sigmoid variant.

Current matching changes relative to the standard PyTorch variant:
1. 1x1 predictor kernels instead of 3x3 predictor kernels.
2. 3 anchors on the first feature map, using the reduced-lowest-layer SSD
   convention from the TensorFlow Object Detection API.
3. Sigmoid-based classification loss and inference scores, with background
   sliced off before NMS.
"""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
import math
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection._utils import retrieve_out_channels
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.models.detection.ssd import SSD
from torchvision.models.mobilenetv2 import mobilenet_v2
from torchvision.ops import boxes as box_ops
from torchvision.ops.misc import Conv2dNormActivation


DEFAULT_INPUT_SIZE = (320, 240)


def _normal_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

# TensorFlow 1.x-style initialization
def _trunc_normal_init(module: nn.Module) -> None:
    for layer in module.modules():
        if isinstance(layer, nn.Conv2d):
            nn.init.trunc_normal_(layer.weight, mean=0.0, std=0.03)
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)


def _ssd_extra_block(
    in_channels: int,
    out_channels: int,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    activation_layer: Callable[..., nn.Module] = nn.ReLU6,
) -> nn.Sequential:
    mid_channels = out_channels // 2
    return nn.Sequential(
        Conv2dNormActivation(
            in_channels,
            mid_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        ),
        Conv2dNormActivation(
            mid_channels,
            out_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer,
        ),
    )


class MobileNetV2SSDFeatureExtractor(nn.Module):
    """SSD feature extractor built from MobileNetV2, with non-lite extra blocks."""

    def __init__(
        self,
        backbone: nn.Sequential,
        c4_pos: int,
        width_mult: float = 1.0,
        min_depth: int = 16,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        init_fn: Callable[[nn.Module], None] = _normal_init,
        activation_layer: Callable[..., nn.Module] = nn.ReLU6,
    ) -> None:
        super().__init__()

        # C4 must be a stride-2 block without residual connection.
        # We split the backbone at this point to obtain the first two SSD
        # feature maps from MobileNetV2.
        if backbone[c4_pos].use_res_connect:
            raise ValueError("backbone[c4_pos].use_res_connect should be False")

        # Backbone portion before the C4 split.
        self.pre_c4_layers = nn.Sequential(*backbone[:c4_pos])
        # First SSD feature map: C4 expansion output (higher spatial resolution).
        self.c4_expansion = backbone[c4_pos].conv[0]
        # Second SSD feature map: remaining C4 ops + all later backbone layers.
        self.post_c4_layers = nn.Sequential(
            backbone[c4_pos].conv[1:],
            *backbone[c4_pos + 1:],
        )

        last_backbone_out_channels = backbone[-1].out_channels
        get_depth = lambda d: max(min_depth, int(d * width_mult))  # noqa: E731

        # Additional SSD feature blocks.
        # These progressively reduce spatial resolution and produce the last
        # four feature maps of the 6-level SSD pyramid.
        extra = nn.ModuleList(
            [
                _ssd_extra_block(
                    last_backbone_out_channels,
                    get_depth(512),
                    norm_layer,
                    activation_layer=activation_layer,
                ),
                _ssd_extra_block(
                    get_depth(512),
                    get_depth(256),
                    norm_layer,
                    activation_layer=activation_layer,
                ),
                _ssd_extra_block(
                    get_depth(256),
                    get_depth(256),
                    norm_layer,
                    activation_layer=activation_layer,
                ),
                _ssd_extra_block(
                    get_depth(256),
                    get_depth(128),
                    norm_layer,
                    activation_layer=activation_layer,
                ),
            ]
        )
        init_fn(extra)
        self.extra = extra

    def forward(self, x: Tensor) -> OrderedDict[str, Tensor]:
        # Collect the 6 SSD feature maps in order.
        output = []

        # Run MobileNetV2 up to the C4 split.
        x = self.pre_c4_layers(x)

        # First SSD feature map: high-resolution C4 expansion output.
        x = self.c4_expansion(x)
        output.append(x)

        # Continue through the rest of MobileNetV2 to obtain the second SSD
        # feature map.
        x = self.post_c4_layers(x)
        output.append(x)

        # Build the remaining SSD feature maps using the extra conv blocks.
        for block in self.extra:
            x = block(x)
            output.append(x)

        # Return an OrderedDict to match torchvision SSD backbone expectations.
        return OrderedDict((str(i), value) for i, value in enumerate(output))


def _mobilenet_v2_extractor(
    backbone: nn.Module,
    width_mult: float = 1.0,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    init_fn: Callable[[nn.Module], None] = _normal_init,
    activation_layer: Callable[..., nn.Module] = nn.ReLU6,
) -> MobileNetV2SSDFeatureExtractor:
    """Wrap a MobileNetV2 backbone into the custom SSD feature extractor."""

    # Discard the MobileNetV2 classifier and keep only convolutional features.
    backbone = backbone.features

    # Stage boundaries are defined by stride-changing blocks.
    # The second-to-last boundary is used as the SSD C4 split point.
    stage_indices = (
        [0]
        + [i for i, block in enumerate(backbone) if getattr(block, "_is_cn", False)]
        + [len(backbone) - 1]
    )

    c4_pos = stage_indices[-2]

    # Build the SSD feature extractor that will output 6 feature maps:
    # 2 from MobileNetV2 and 4 from extra SSD blocks.
    return MobileNetV2SSDFeatureExtractor(
        backbone,
        c4_pos,
        width_mult=width_mult,
        norm_layer=norm_layer,
        init_fn=init_fn,
        activation_layer=activation_layer,
    )


class SSDHead1x1(nn.Module):
    """SSD head using 1x1 predictors instead of the torchvision 3x3 ones."""

    def __init__(
        self,
        in_channels: list[int],
        num_anchors: list[int],
        num_classes: int,
        init_fn: Callable[[nn.Module], None] = _normal_init,
    ):
        super().__init__()
        # Classification head:
        # for each feature map location and anchor, predict class logits.
        self.classification_head = _SSDScoringHead1x1(
            in_channels, num_anchors, num_classes, init_fn=init_fn
        )
        # Regression head:
        # for each feature map location and anchor, predict 4 box offsets.
        self.regression_head = _SSDScoringHead1x1(
            in_channels, num_anchors, 4, init_fn=init_fn
        )

    def forward(self, x: list[Tensor]) -> dict[str, Tensor]:
        # Run both heads on the same SSD feature maps.
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
        }


class _SSDScoringHead1x1(nn.Module):
    def __init__(
        self,
        in_channels: list[int],
        num_anchors: list[int],
        num_columns: int,
        init_fn: Callable[[nn.Module], None] = _normal_init,
    ):
        super().__init__()
        layers = nn.ModuleList()
        # Create one predictor conv per SSD feature map.
        # Output channels = num_anchors_at_this_level * values_per_anchor.
        for channels, anchors in zip(in_channels, num_anchors):
            layers.append(nn.Conv2d(channels, anchors * num_columns, kernel_size=1))
        init_fn(layers)
        self.layers = layers
        self.num_columns = num_columns

    def forward(self, x: list[Tensor]) -> Tensor:
        all_results = []
        for features, layer in zip(x, self.layers):
            # Apply the 1x1 predictor on one feature map.
            results = layer(features)
            n, _, h, w = results.shape

            # Reshape from [N, A*K, H, W] to [N, H*W*A, K], where:
            # - A = number of anchors per spatial location
            # - K = values predicted per anchor
            # This is the format expected by the SSD loss / postprocessing code.
            results = results.view(n, -1, self.num_columns, h, w)
            results = results.permute(0, 3, 4, 1, 2)
            results = results.reshape(n, -1, self.num_columns)
            all_results.append(results)

        # Concatenate predictions from all SSD feature maps into a single tensor.
        return torch.cat(all_results, dim=1)


class ReducedFirstLayerBoxGenerator(DefaultBoxGenerator):
    """
    TensorFlow-style SSD uses 3 anchors on the first feature map when the
    reduced-lowest-layer anchor convention is enabled.

    The first feature map uses the common TF Object Detection API
    reduced-lowest-layer convention:
    - one square box at scale 0.1
    - one box at aspect ratio 2.0 using the first layer scale
    - one box at aspect ratio 0.5 using the first layer scale
    """

    def num_anchors_per_location(self) -> list[int]:
        # First feature map:
        # use the reduced-lowest-layer convention, which creates 3 anchors
        # per spatial location.

        # Remaining feature maps:
        # use the standard SSD rule:
        # 2 square boxes + 2 boxes for each extra aspect ratio.
        return [3] + [2 + 2 * len(ratios) for ratios in self.aspect_ratios[1:]]

    def _generate_wh_pairs(
        self,
        num_outputs: int,
        dtype: torch.dtype = torch.float32,
        device: torch.device = torch.device("cpu"),
    ) -> list[Tensor]:
        wh_pairs: list[Tensor] = []
        for idx in range(num_outputs):
            # First feature map:
            # manually create the 3 reduced anchors instead of using the
            # standard SSD width/height generation rule.
            if idx == 0:
                scale = self.scales[idx]
                wh_pairs.append(
                    torch.as_tensor(
                        [
                            # Small square box
                            [0.1, 0.1],
                            # Wide box with aspect ratio 2.0
                            [scale * math.sqrt(2.0), scale / math.sqrt(2.0)],
                            # Tall box with aspect ratio 0.5
                            [scale / math.sqrt(2.0), scale * math.sqrt(2.0)],
                        ],
                        dtype=dtype,
                        device=device,
                    )
                )
                continue

            # Standard SSD rule for later feature maps:
            # - one square box with scale s_k
            # - one square box with scale sqrt(s_k * s_(k+1))
            s_k = self.scales[idx]
            s_prime_k = math.sqrt(self.scales[idx] * self.scales[idx + 1])
            level_pairs = [[s_k, s_k], [s_prime_k, s_prime_k]]

            # For each extra aspect ratio r, add:
            # - a wide box  (w = s_k * sqrt(r), h = s_k / sqrt(r))
            # - a tall box  (w = s_k / sqrt(r), h = s_k * sqrt(r))
            for aspect_ratio in self.aspect_ratios[idx]:
                sq_aspect_ratio = math.sqrt(aspect_ratio)
                width = self.scales[idx] * sq_aspect_ratio
                height = self.scales[idx] / sq_aspect_ratio
                level_pairs.extend([[width, height], [height, width]])

            # Store the width/height pairs for this feature map level.
            wh_pairs.append(torch.as_tensor(level_pairs, dtype=dtype, device=device))

        # Return one tensor of (width, height) pairs per SSD feature map.
        return wh_pairs


class SigmoidSSD(SSD):
    """
    SSD variant with TF-style sigmoid classification semantics.

    The model still carries an internal background channel, but:
    - unmatched anchors use all-zero class targets
    - foreground labels populate channels 1..C
    - inference applies sigmoid and slices off background before NMS
    """

    def __init__(self, *args: Any, min_negatives_per_image: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.min_negatives_per_image = min_negatives_per_image

    def compute_loss(
        self,
        targets: list[dict[str, Tensor]],
        head_outputs: dict[str, Tensor],
        anchors: list[Tensor],
        matched_idxs: list[Tensor],
    ) -> dict[str, Tensor]:
        """
        Inputs:
            targets:
                list of dictionaries, each containing "boxes" and "labels" tensors
                {"boxes": Tensor, "labels": Tensor}
            head_outputs:
                dictionary containing "bbox_regression" and "cls_logits" tensors
                {"bbox_regression": Tensor, "cls_logits": Tensor}
            anchors:
                list of per-image anchor tensors
                each tensor has shape [num_anchors, 4]
            matched_idxs:
                list of per-image match tensors
                each tensor has shape [num_anchors]
                value >= 0 means matched GT index, -1 means unmatched
        """

        bbox_regression = head_outputs["bbox_regression"]
        cls_logits = head_outputs["cls_logits"]

        num_classes_with_background = cls_logits.size(-1)
        num_foreground_classes = num_classes_with_background - 1

        num_foreground = 0
        bbox_loss = []
        cls_targets = []
        foreground_masks = []

        for (
            targets_per_image,
            bbox_regression_per_image,
            anchors_per_image,
            matched_idxs_per_image,
        ) in zip(targets, bbox_regression, anchors, matched_idxs):
            # Find indices of foreground anchors
            foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]

            # For each foreground anchor, find the corresponding matched GT box index
            foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]

            # Save a boolean mask of matched anchors for this image.
            # Only matched anchors contribute to the box regression loss.
            foreground_masks.append(matched_idxs_per_image >= 0)
            num_foreground += foreground_matched_idxs_per_image.numel()

            # Check whether this image has any matched anchors at all
            if foreground_matched_idxs_per_image.numel() > 0:

                # Take the GT boxes coordinates assigned to the positive anchors
                matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]

                # Keep only the model’s predicted box offsets for positive anchors
                bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]

                # Keep only the anchor boxes corresponding to those positive anchors
                anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]

                # Convert each matched GT box into the target offset relative to its anchor
                target_regression = self.box_coder.encode_single(
                    matched_gt_boxes_per_image,
                    anchors_per_image,
                )
                bbox_loss.append(
                    F.smooth_l1_loss(
                        bbox_regression_per_image,
                        target_regression,
                        reduction="sum",
                    )
                )
            else:
                bbox_loss.append(
                    torch.zeros((), dtype=bbox_regression.dtype, device=bbox_regression.device)
                )

            # Create sigmoid/BCE classification targets for all anchors.
            # Unmatched anchors stay all zeros; positive anchors get a 1.0
            # in their matched foreground class channel.
            # Initialize a zero target matrix of shape
            # (num_anchors, num_classes_with_background).
            gt_classes_target = torch.zeros(
                (cls_logits.size(1), num_classes_with_background),
                dtype=cls_logits.dtype,
                device=cls_logits.device,
            )
            # For each positive anchor, set its matched foreground class
            # channel to 1.0.
            if foreground_matched_idxs_per_image.numel() > 0:
                foreground_labels = targets_per_image["labels"][foreground_matched_idxs_per_image]
                torch._assert(
                    torch.all((foreground_labels >= 1) & (foreground_labels <= num_foreground_classes)),
                    "Expected foreground labels in [1, num_foreground_classes].",
                )
                gt_classes_target[foreground_idxs_per_image, foreground_labels] = 1.0

            cls_targets.append(gt_classes_target)

        bbox_loss = torch.stack(bbox_loss)
        cls_targets = torch.stack(cls_targets)
        foreground_mask = torch.stack(foreground_masks)

        cls_loss = F.binary_cross_entropy_with_logits(
            cls_logits,
            cls_targets,
            reduction="none",
        ).sum(dim=-1)

        # Negative count calculation
        # Given neg_to_pos_ratio = x, num_negative = x * num_foreground
        num_negative = (self.neg_to_pos_ratio * foreground_mask.sum(1, keepdim=True)).to(torch.int64)
        num_negative = torch.clamp_min(num_negative, self.min_negatives_per_image)

        negative_loss = cls_loss.clone()
        # Put to -inf the positive anchors
        negative_loss[foreground_mask] = -float("inf")
        # Select hardest negative examples (higher loss)
        values, idx = negative_loss.sort(1, descending=True)
        background_idxs = torch.logical_and(
            idx.sort(1)[1] < num_negative,
            torch.isfinite(values),
        )

        normalizer = max(1, num_foreground)

        # Final losses:
        # - bbox_regression: summed box regression loss, normalized by num_foreground
        # - classification: summed positive + selected negative BCE loss,
        #   normalized by num_foreground
        return {
            "bbox_regression": bbox_loss.sum() / normalizer,
            "classification": (
                cls_loss[foreground_mask].sum() + cls_loss[background_idxs].sum()
            ) / normalizer,
        }

    def postprocess_detections(
        self,
        head_outputs: dict[str, Tensor],
        image_anchors: list[Tensor],
        image_shapes: list[tuple[int, int]],
    ) -> list[dict[str, Tensor]]:
        """
        Inputs:
            head_outputs:
                dictionary containing "bbox_regression" and "cls_logits" tensors
                {"bbox_regression": Tensor, "cls_logits": Tensor}
            image_anchors:
                list of anchor boxes for each image
                [Tensor, Tensor, ...]
            image_shapes:
                list of per-image shapes as (height, width)
        """
        bbox_regression = head_outputs["bbox_regression"]
        # Inference uses sigmoid activation for class scores
        # Immediately remove background class scores
        pred_scores = torch.sigmoid(head_outputs["cls_logits"])[..., 1:]

        num_classes = pred_scores.size(-1)
        device = pred_scores.device

        detections: list[dict[str, Tensor]] = []

        # Iterate over each image's predictions
        for boxes, scores, anchors, image_shape in zip(
            bbox_regression,
            pred_scores,
            image_anchors,
            image_shapes,
        ):
            # Decode bounding box predictions
            boxes = self.box_coder.decode_single(boxes, anchors)
            boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

            image_boxes = []
            image_scores = []
            image_labels = []
            # Iterate over each class
            for class_idx in range(num_classes):
                score = scores[:, class_idx]

                # Filter out low-confidence predictions
                keep_idxs = score > self.score_thresh
                score = score[keep_idxs]
                box = boxes[keep_idxs]

                num_topk = det_utils._topk_min(score, self.topk_candidates, 0)
                score, idxs = score.topk(num_topk)
                box = box[idxs]

                image_boxes.append(box)
                image_scores.append(score)
                image_labels.append(
                    torch.full_like(score, fill_value=class_idx + 1, dtype=torch.int64, device=device)
                )

            image_boxes = torch.cat(image_boxes, dim=0)
            image_scores = torch.cat(image_scores, dim=0)
            image_labels = torch.cat(image_labels, dim=0)

            # Apply NMS separately for each class label
            keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
            keep = keep[: self.detections_per_img]

            detections.append(
                {
                    "boxes": image_boxes[keep],
                    "scores": image_scores[keep],
                    "labels": image_labels[keep],
                }
            )
        return detections


def mobilenet_v2_ssd_sigmoid(
    num_classes: int,
    size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
    width_mult: float = 1.0,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    match_init: bool = True,
    relu6: bool = True,
    **kwargs: Any,
) -> SSD:
    """
    Construct a MobileNetV2 SSD sigmoid variant with TF-style semantics.

    Args:
        num_classes: Number of foreground classes. This builder adds an
            internal background channel automatically.
    """
    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # Choose the initialization rule:
    # - TF-style truncated normal when match_init=True
    # - standard normal otherwise
    init_fn = _trunc_normal_init if match_init else _normal_init
    activation_layer = nn.ReLU6 if relu6 else nn.ReLU

    # Build a MobileNetV2 backbone from scratch and initialize its weights.
    backbone_model = mobilenet_v2(weights=None, width_mult=width_mult)
    init_fn(backbone_model)

    # Wrap MobileNetV2 into the SSD feature extractor that produces
    # the 6 feature maps used by SSD.
    backbone = _mobilenet_v2_extractor(
        backbone_model,
        width_mult=width_mult,
        norm_layer=norm_layer,
        init_fn=init_fn,
        activation_layer=activation_layer,
    )

    # Build the SSD anchor generator.
    # The first feature map uses the reduced 3-anchor TF-style convention.
    anchor_generator = ReducedFirstLayerBoxGenerator(
        [[2, 3] for _ in range(6)],
        min_ratio=0.2,
        max_ratio=0.95,
    )

    # Infer how many channels each SSD feature map produces.
    out_channels = retrieve_out_channels(backbone, size)
    # Infer how many anchors each feature map level uses.
    num_anchors = anchor_generator.num_anchors_per_location()

    if len(out_channels) != len(num_anchors):
        raise ValueError(
            f"Backbone produces {len(out_channels)} feature maps but anchor generator "
            f"expects {len(num_anchors)}"
        )

    # This sigmoid-matched variant treats num_classes as foreground classes.
    # Add one internal background channel to match the TF-style setup.
    num_classes_with_background = num_classes + 1

    # Build the SSD prediction head:
    # - 1x1 classification predictors
    # - 1x1 box regression predictors
    head = SSDHead1x1(
        out_channels,
        num_anchors,
        num_classes_with_background,
        init_fn=init_fn,
    )

    # Post-processing defaults chosen to stay close to the TF SSD config.
    defaults = {
        # Closest torchvision equivalents to the TF1 SSD post-processing config.
        # Note: torchvision SSD has no separate max_detections_per_class setting.
        "score_thresh": 1e-5,
        "nms_thresh": 0.5,
        "detections_per_img": 100,
        "topk_candidates": 100,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}

    # Assemble the final detector:
    # - custom sigmoid-style SSD loss / inference behavior
    # - custom MobileNetV2 SSD backbone
    # - custom reduced-first-layer anchor generator
    # - custom 1x1 prediction head
    return SigmoidSSD(
        backbone,
        anchor_generator,
        size,
        num_classes_with_background,
        head=head,
        **kwargs,
    )

MobilenetV2SSD_Sigmoid = mobilenet_v2_ssd_sigmoid
build_model = mobilenet_v2_ssd_sigmoid
