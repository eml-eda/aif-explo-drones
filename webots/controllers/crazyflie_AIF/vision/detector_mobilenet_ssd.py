from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

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


def _default_model_path() -> Path:
    return (
        _detection_training_root()
        / "mobilenetv2_ssd_all"
        / "runs"
        / "mbv2ssd_sigmoid_bottle_can_v1"
        / "best.pt"
    )


def _import_model_builder():
    training_root = _detection_training_root()

    if str(training_root) not in sys.path:
        sys.path.insert(0, str(training_root))

    from mobilenetv2_ssd.mobilenetv2_ssd_sigmoid import mobilenet_v2_ssd_sigmoid

    return mobilenet_v2_ssd_sigmoid


def _torch_load(path: str | Path, map_location: torch.device) -> dict[str, Any]:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


class Detector:
    """
    Drop-in replacement for vision.detector_yolo.Detector.

    Returns detections in the same format expected by crazyflie_AIF.py:

        {
            "bbox_xyxy": [x1, y1, x2, y2],
            "confidence": float,
            "class_id": int,       # YOLO-style foreground id: 0, 1, ...
            "label": str,
        }

    The trained MobileNetV2-SSD uses torchvision-style labels:
        0 = background
        1 = YOLO class 0
        2 = YOLO class 1
    """

    def __init__(
        self,
        model_name=None,
        device="cuda:0",
        conf=0.25,
        iou=0.45,
        img_size=320,
        class_names=None,
    ):
        if model_name is None:
            model_name = os.environ.get("AIF_MBV2SSD_MODEL", str(_default_model_path()))

        if device.startswith("cuda") and not torch.cuda.is_available():
            print("[MBV2-SSD] CUDA requested but unavailable; falling back to CPU", flush=True)
            device = "cpu"

        self.model_name = str(model_name)
        self.device = torch.device(os.environ.get("AIF_MBV2SSD_DEVICE", device))
        self.conf = float(os.environ.get("AIF_MBV2SSD_CONF", conf))
        self.iou = float(os.environ.get("AIF_MBV2SSD_IOU", iou))

        if not os.path.exists(self.model_name):
            raise FileNotFoundError(f"MobileNetV2-SSD checkpoint not found: {self.model_name}")

        ckpt = _torch_load(self.model_name, map_location=self.device)

        if "model_state" not in ckpt:
            raise KeyError(
                f"Checkpoint {self.model_name} does not contain 'model_state'. "
                f"Found keys: {list(ckpt.keys())}"
            )

        ckpt_args = ckpt.get("args", {}) or {}

        ckpt_class_names = ckpt.get("class_names", None)
        if class_names is None:
            if ckpt_class_names is None:
                class_names = ["target_bottle", "target_can"]
            else:
                class_names = list(ckpt_class_names)

        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)

        image_width = int(ckpt_args.get("image_width", img_size))
        image_height = int(ckpt_args.get("image_height", 240))
        width_mult = float(ckpt_args.get("width_mult", 1.0))
        relu6 = bool(ckpt_args.get("relu6", True))
        match_init = bool(ckpt_args.get("match_init", True))

        mobilenet_v2_ssd_sigmoid = _import_model_builder()

        self.model = mobilenet_v2_ssd_sigmoid(
            num_classes=self.num_classes,
            size=(image_width, image_height),
            width_mult=width_mult,
            relu6=relu6,
            match_init=match_init,
            score_thresh=self.conf,
            nms_thresh=self.iou,
        )

        self.model.load_state_dict(ckpt["model_state"], strict=True)
        self.model.to(self.device)
        self.model.eval()

        # These exist on torchvision SSD and are useful if the builder defaults
        # are overwritten or if env vars change.
        self.model.score_thresh = self.conf
        self.model.nms_thresh = self.iou

        print(f"[MBV2-SSD] model={self.model_name}", flush=True)
        print(f"[MBV2-SSD] device={self.device}", flush=True)
        print(
            f"[MBV2-SSD] classes={dict(enumerate(self.class_names))} "
            f"foreground labels start at 1",
            flush=True,
        )
        print(
            f"[MBV2-SSD] size=({image_width},{image_height}) "
            f"width_mult={width_mult} relu6={relu6} match_init={match_init} "
            f"conf={self.conf:.3f} iou={self.iou:.3f}",
            flush=True,
        )

    def _image_from_webots(self, image, width, height):
        if image is None or width is None or height is None:
            return None

        frame = np.frombuffer(image, dtype=np.uint8)
        expected = int(width * height * 4)
        if frame.size != expected:
            print(
                f"[MBV2-SSD] invalid Webots image buffer: got {frame.size}, expected {expected}",
                flush=True,
            )
            return None

        # Webots camera buffer is BGRA.
        bgra = frame.reshape((height, width, 4))

        # Training used PIL.Image.convert("RGB"), so use RGB here.
        rgb = bgra[:, :, [2, 1, 0]].copy()
        return rgb

    def _tensor_from_rgb(self, rgb: np.ndarray) -> torch.Tensor:
        # Torchvision detection models expect CHW float tensor in [0, 1].
        tensor = torch.from_numpy(rgb).to(self.device)
        tensor = tensor.permute(2, 0, 1).contiguous().float() / 255.0
        return tensor

    def _label_to_output_class(self, label: int) -> tuple[int, str]:
        # Model label 0 is background; foreground labels are 1..C.
        class_id = int(label) - 1

        if 0 <= class_id < len(self.class_names):
            return class_id, str(self.class_names[class_id])

        return class_id, f"class_{class_id}"

    def process(self, image, width=None, height=None, timestamp=None):
        rgb = self._image_from_webots(image, width, height)
        if rgb is None:
            return DetectorOutput(detections=[], embedding=None, raw=None)

        img_tensor = self._tensor_from_rgb(rgb)

        with torch.no_grad():
            # Important: your model is torchvision-style and expects a list.
            outputs = self.model([img_tensor])

        detections = []

        if not outputs:
            return DetectorOutput(detections=[], embedding=None, raw=outputs)

        output = outputs[0]

        boxes = output.get("boxes", torch.empty((0, 4), device=self.device))
        scores = output.get("scores", torch.empty((0,), device=self.device))
        labels = output.get("labels", torch.empty((0,), dtype=torch.int64, device=self.device))

        boxes = boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy().astype(int)

        for box, score, label in zip(boxes, scores, labels):
            score = float(score)
            if score < self.conf:
                continue

            class_id, class_name = self._label_to_output_class(int(label))

            x1, y1, x2, y2 = [float(v) for v in box]

            detections.append(
                {
                    "bbox_xyxy": [x1, y1, x2, y2],
                    "confidence": score,
                    "class_id": int(class_id),
                    "label": class_name,
                }
            )

        return DetectorOutput(detections=detections, embedding=None, raw=outputs)