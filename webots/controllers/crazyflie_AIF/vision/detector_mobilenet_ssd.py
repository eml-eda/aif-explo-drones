from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch


class DetectorOutput:
    def __init__(self, detections=None, embedding=None, raw=None):
        self.detections = detections or []
        self.embedding = embedding
        self.raw = raw


def _repo_root() -> Path:
    # detector_mobilenet_ssd.py
    #   vision/ -> crazyflie_AIF/ -> controllers/ -> webots/ -> repo root
    return Path(__file__).resolve().parents[4]


def _detection_training_root() -> Path:
    return _repo_root() / "detection_model_training"



def _import_model_builder():
    training_root = _detection_training_root()

    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))

    from mobilenetv2_ssd.mobilenetv2_ssd_sigmoid import (
        mobilenet_v2_ssd_sigmoid,
    )

    return mobilenet_v2_ssd_sigmoid


def _torch_load(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(
            path,
            map_location=map_location,
            weights_only=False,
        )
    except TypeError:
        return torch.load(
            path,
            map_location=map_location,
        )


def _normalize_class_names(value) -> Optional[list[str]]:
    """
    Normalize checkpoint/controller class-name representations.

    Training checkpoints produced by train_mobilenetv2_ssd_sigmoid.py store
    class_names as an ordered list, but accepting a dict here makes the runtime
    slightly more robust to older/custom checkpoints.
    """
    if value is None:
        return None

    if isinstance(value, dict):
        try:
            return [
                str(value[k])
                for k in sorted(value.keys(), key=lambda x: int(x))
            ]
        except Exception:
            return [str(v) for v in value.values()]

    return [str(v) for v in value]


def _resolve_device(device: str) -> torch.device:
    requested = str(device).strip()
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(
            "[MBV2-SSD] CUDA requested but unavailable; falling back to CPU",
            flush=True,
        )
        requested = "cpu"
    return torch.device(requested)



class Detector:
    """
    MobileNetV2-SSD detector used by the Webots controller.

    Output format matches the detector interface expected by crazyflie_AIF.py:

        {
            "bbox_xyxy": [x1, y1, x2, y2],
            "confidence": float,

            # Controller/YOLO-style foreground class id:
            #   0 = target_bottle
            #   1 = target_can
            #   2 = desk        (medium/hard models)
            "class_id": int,

            "label": str,
            "class_name": str,
        }

    Label convention
    ----------------
    Training uses torchvision-style SSD labels:

        0     = background
        1..C  = foreground classes

    The deployment/controller convention is zero-based foreground ids:

        SSD label 1 -> class_id 0
        SSD label 2 -> class_id 1
        SSD label 3 -> class_id 2
        ...

    The checkpoint's class_names list defines the semantic meaning and order
    of those foreground labels.
    """

    def __init__(
        self,
        model_name=None,
        device="cuda:0",
        conf=0.25,
        iou=0.45,
        img_size=320,
        class_names=None,
        expected_model_size=None,
    ):
        # ------------------------------------------------------------------
        # Resolve checkpoint path and device
        # ------------------------------------------------------------------
        if not model_name:
            raise ValueError(
                "MobileNetV2-SSD checkpoint path was not supplied. Select it "
                "through configuration/model_registry.py or AIF_MODEL_PATH."
            )

        self.model_name = str(model_name)
        self.device = _resolve_device(device)
        self.conf = float(conf)
        self.iou = float(iou)
        self.detections_per_img = 100

        if not os.path.exists(self.model_name):
            raise FileNotFoundError(
                f"MobileNetV2-SSD checkpoint not found: {self.model_name}"
            )

        # ------------------------------------------------------------------
        # Load checkpoint
        # ------------------------------------------------------------------
        ckpt = _torch_load(
            self.model_name,
            map_location=self.device,
        )

        if not isinstance(ckpt, dict) or "model_state" not in ckpt:
            raise KeyError(
                f"Checkpoint {self.model_name} does not contain 'model_state'. "
                f"Found keys: "
                f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
            )

        ckpt_args = ckpt.get("args", {}) or {}

        # ------------------------------------------------------------------
        # Resolve classes
        #
        # The trainer stores class_names directly in the checkpoint.
        # Those names are authoritative because they correspond exactly to the
        # output-head ordering learned during training.
        #
        # crazyflie_AIF.py also supplies the expected environment classes.
        # If both are present, require them to match rather than silently
        # assigning the wrong semantic labels to model outputs.
        # ------------------------------------------------------------------
        checkpoint_class_names = _normalize_class_names(
            ckpt.get("class_names", None)
        )
        requested_class_names = _normalize_class_names(class_names)

        if (
            checkpoint_class_names is not None
            and requested_class_names is not None
            and checkpoint_class_names != requested_class_names
        ):
            raise ValueError(
                "MobileNetV2-SSD class mismatch.\n"
                f"  checkpoint: {checkpoint_class_names}\n"
                f"  controller: {requested_class_names}\n"
                f"  checkpoint path: {self.model_name}\n"
                "This usually means the wrong environment-specific weights "
                "were selected in config.py."
            )

        if checkpoint_class_names is not None:
            self.class_names = checkpoint_class_names
        elif requested_class_names is not None:
            self.class_names = requested_class_names
        else:
            # Backwards-compatible fallback for older easy checkpoints.
            self.class_names = [
                "target_bottle",
                "target_can",
            ]

        self.num_classes = len(self.class_names)

        if self.num_classes <= 0:
            raise ValueError(
                "MobileNetV2-SSD resolved zero foreground classes."
            )

        # ------------------------------------------------------------------
        # Reconstruct the exact architecture used by training.
        #
        # train_mobilenetv2_ssd_sigmoid.py stores these arguments in the
        # checkpoint. Both medium and hard models were trained at 320x240.
        # ------------------------------------------------------------------
        image_width = int(
            ckpt_args.get("image_width", img_size)
        )
        image_height = int(
            ckpt_args.get("image_height", 240)
        )
        width_mult = float(
            ckpt_args.get("width_mult", 1.0)
        )
        relu6 = bool(
            ckpt_args.get("relu6", True)
        )
        match_init = bool(
            ckpt_args.get("match_init", True)
        )

        self.image_width = image_width
        self.image_height = image_height
        self.width_mult = width_mult

        if expected_model_size is not None:
            try:
                expected_width_mult = float(expected_model_size)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid expected MobileNet model size: {expected_model_size!r}"
                ) from exc
            if abs(float(width_mult) - expected_width_mult) > 1e-9:
                raise ValueError(
                    "MobileNetV2-SSD model-size mismatch.\n"
                    f"  expected width_mult: {expected_width_mult}\n"
                    f"  checkpoint width_mult: {width_mult}\n"
                    f"  checkpoint path: {self.model_name}"
                )

        mobilenet_v2_ssd_sigmoid = _import_model_builder()

        self.model = mobilenet_v2_ssd_sigmoid(
            num_classes=self.num_classes,
            size=(image_width, image_height),
            width_mult=width_mult,
            relu6=relu6,
            match_init=match_init,
            score_thresh=self.conf,
            nms_thresh=self.iou,
            detections_per_img=self.detections_per_img,
        )

        # strict=True is intentional. A medium/hard checkpoint with three
        # foreground classes must not silently load into a two-class head,
        # or vice versa.
        try:
            self.model.load_state_dict(
                ckpt["model_state"],
                strict=True,
            )
        except RuntimeError as exc:
            raise RuntimeError(
                "Failed to load MobileNetV2-SSD checkpoint architecture.\n"
                f"  checkpoint: {self.model_name}\n"
                f"  classes: {self.class_names}\n"
                f"  image size: ({image_width}, {image_height})\n"
                f"  width_mult: {width_mult}\n"
                f"Original error:\n{exc}"
            ) from exc

        self.model.to(self.device)
        self.model.eval()

        # torchvision-style SSD attributes.
        self.model.score_thresh = self.conf
        self.model.nms_thresh = self.iou
        self.model.detections_per_img = self.detections_per_img

        # ------------------------------------------------------------------
        # Startup diagnostics
        # ------------------------------------------------------------------
        print(
            f"[MBV2-SSD] model={self.model_name}",
            flush=True,
        )
        print(
            f"[MBV2-SSD] device={self.device}",
            flush=True,
        )
        print(
            f"[MBV2-SSD] classes={dict(enumerate(self.class_names))}",
            flush=True,
        )
        print(
            "[MBV2-SSD] label mapping: "
            "SSD 0=background, SSD 1..C -> controller class_id 0..C-1",
            flush=True,
        )
        print(
            f"[MBV2-SSD] size=({image_width},{image_height}) "
            f"width_mult={width_mult} "
            f"relu6={relu6} "
            f"match_init={match_init} "
            f"conf={self.conf:.3f} "
            f"iou={self.iou:.3f} "
            f"detections_per_img={self.detections_per_img}",
            flush=True,
        )

        if (image_width, image_height) != (320, 240):
            print(
                f"[MBV2-SSD] WARNING: checkpoint input is "
                f"{image_width}x{image_height}; expected deployment camera "
                "configuration is 320x240.",
                flush=True,
            )

    # ----------------------------------------------------------------------
    # Input conversion
    # ----------------------------------------------------------------------

    def _image_from_webots(
        self,
        image,
        width,
        height,
    ) -> Optional[np.ndarray]:
        if image is None or width is None or height is None:
            return None

        width = int(width)
        height = int(height)

        frame = np.frombuffer(
            image,
            dtype=np.uint8,
        )

        expected = int(width * height * 4)

        if frame.size != expected:
            print(
                f"[MBV2-SSD] invalid Webots image buffer: "
                f"got {frame.size}, expected {expected}",
                flush=True,
            )
            return None

        # Webots camera buffer is BGRA.
        bgra = frame.reshape(
            (height, width, 4)
        )

        # Training reads images using:
        #     Image.open(...).convert("RGB")
        #
        # Convert Webots BGRA to the same RGB ordering.
        rgb = bgra[:, :, [2, 1, 0]].copy()

        return rgb

    def _tensor_from_rgb(
        self,
        rgb: np.ndarray,
    ) -> torch.Tensor:
        """
        Training/evaluation convert RGB pixels to CHW float tensors in [0, 1].
        Reproduce exactly the same representation here.
        """
        tensor = torch.from_numpy(rgb)
        tensor = tensor.to(
            self.device,
            non_blocking=True,
        )

        tensor = (
            tensor
            .permute(2, 0, 1)
            .contiguous()
            .float()
            / 255.0
        )

        return tensor

    # ----------------------------------------------------------------------
    # Output conversion
    # ----------------------------------------------------------------------

    def _label_to_output_class(
        self,
        label: int,
    ) -> tuple[int, str]:
        """
        Convert one-based SSD foreground labels back to the zero-based
        controller/YOLO convention used everywhere else in the project.
        """
        label = int(label)

        # 0 is reserved for background and should not normally be emitted by
        # the detector's post-processing.
        if label <= 0:
            return -1, "background"

        class_id = label - 1

        if 0 <= class_id < len(self.class_names):
            return (
                class_id,
                str(self.class_names[class_id]),
            )

        return (
            class_id,
            f"class_{class_id}",
        )

    # ----------------------------------------------------------------------
    # Inference
    # ----------------------------------------------------------------------

    def process(
        self,
        image,
        width=None,
        height=None,
        timestamp=None,
    ) -> DetectorOutput:
        rgb = self._image_from_webots(
            image,
            width,
            height,
        )

        if rgb is None:
            return DetectorOutput(
                detections=[],
                embedding=None,
                raw=None,
            )

        img_tensor = self._tensor_from_rgb(rgb)

        with torch.no_grad():
            # The custom model follows the torchvision detection API and expects
            # a list of CHW image tensors.
            outputs = self.model([img_tensor])

        if not outputs:
            return DetectorOutput(
                detections=[],
                embedding=None,
                raw=outputs,
            )

        output = outputs[0]

        boxes = output.get(
            "boxes",
            torch.empty(
                (0, 4),
                dtype=torch.float32,
                device=self.device,
            ),
        )

        scores = output.get(
            "scores",
            torch.empty(
                (0,),
                dtype=torch.float32,
                device=self.device,
            ),
        )

        labels = output.get(
            "labels",
            torch.empty(
                (0,),
                dtype=torch.int64,
                device=self.device,
            ),
        )

        boxes = (
            boxes
            .detach()
            .cpu()
            .numpy()
        )

        scores = (
            scores
            .detach()
            .cpu()
            .numpy()
        )

        labels = (
            labels
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        detections = []

        for box, score, label in zip(
            boxes,
            scores,
            labels,
        ):
            score = float(score)

            # Normally already enforced by the SSD post-processing, but retain
            # the explicit deployment-side check.
            if score < self.conf:
                continue

            class_id, class_name = (
                self._label_to_output_class(label)
            )

            # Ignore background/invalid model labels.
            if class_id < 0:
                continue

            x1, y1, x2, y2 = [
                float(v)
                for v in box
            ]

            detections.append(
                {
                    "bbox_xyxy": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                    "confidence": score,
                    "class_id": int(class_id),

                    # Provide both fields explicitly. The controller currently
                    # accepts either, but this makes MobileNet output identical
                    # in semantics to the YOLO/table detector pipeline.
                    "label": class_name,
                    "class_name": class_name,
                }
            )

        return DetectorOutput(
            detections=detections,
            embedding=None,
            raw=outputs,
        )