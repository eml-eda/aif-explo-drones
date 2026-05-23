# crazyflie_AIF/vision/detector_yolo.py

import os

import numpy as np
from ultralytics import YOLO
from config import AIF_YOLO_MODEL

class DetectorOutput:
    def __init__(self, detections=None, embedding=None, raw=None):
        self.detections = detections or []
        self.embedding = embedding
        self.raw = raw


class Detector:
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
            model_name = os.environ.get(
                "AIF_YOLO_MODEL",
                AIF_YOLO_MODEL,
            )
        self.model_name = model_name

        self.device = device
        self.conf = float(os.environ.get("AIF_YOLO_CONF", conf))
        self.iou = float(os.environ.get("AIF_YOLO_IOU", iou))
        self.img_size = int(os.environ.get("AIF_YOLO_IMGSZ", img_size))

        self.model = YOLO(self.model_name)

        if class_names is None:
            class_names = ["target_bottle", "target_can"]

        name_to_id = {v: k for k, v in self.model.names.items()}
        self.classes = [name_to_id[n] for n in class_names if n in name_to_id]

        # If none matched, disable filtering rather than silently dropping all detections
        if not self.classes:
            self.classes = None

        print(f"[YOLO] model={self.model_name}")
        print("YOLO classes example:", list(self.model.names.items())[:10])
        print("Filtering classes:", self.classes)

    def _image_from_webots(self, image, width, height):
        if image is None or width is None or height is None:
            return None

        frame = np.frombuffer(image, dtype=np.uint8)
        expected = int(width * height * 4)
        if frame.size != expected:
            return None

        frame = frame.reshape((height, width, 4))
        frame = frame[:, :, :3]  # Webots camera buffer is BGRA -> keep BGR for YOLO
        return frame

    def process(self, image, width=None, height=None, timestamp=None):
        frame = self._image_from_webots(image, width, height)
        if frame is None:
            return DetectorOutput(detections=[], embedding=None, raw=None)

        results = self.model.predict(
            source=frame,
            imgsz=self.img_size,
            conf=self.conf,
            iou=self.iou,
            device=self.device,
            classes=self.classes,
            verbose=False,
        )

        detections = []
        if results:
            res = results[0]
            boxes = res.boxes
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy()
                confs = boxes.conf.detach().cpu().numpy()
                clss = boxes.cls.detach().cpu().numpy().astype(int)

                for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
                    label = res.names.get(int(cls_id), str(int(cls_id)))
                    detections.append(
                        {
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(score),
                            "class_id": int(cls_id),
                            "label": str(label),
                        }
                    )

        return DetectorOutput(detections=detections, embedding=None, raw=results)