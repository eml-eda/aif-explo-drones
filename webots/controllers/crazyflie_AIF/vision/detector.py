import os

import numpy as np

from ultralytics import YOLO


class DetectorOutput:
    def __init__(self, detections=None, embedding=None, raw=None):
        self.detections = detections or []
        self.embedding = embedding
        self.raw = raw


class Detector:
    def __init__(self, model_name=None, device=None, conf=0.25, iou=0.45, img_size=320):
        if model_name is None:
            model_name = os.environ.get("AIF_YOLO_MODEL", "yolo11n.pt")
        self.model_name = model_name
        self.device = device or os.environ.get("AIF_YOLO_DEVICE", None)
        self.conf = float(os.environ.get("AIF_YOLO_CONF", conf))
        self.iou = float(os.environ.get("AIF_YOLO_IOU", iou))
        self.img_size = int(os.environ.get("AIF_YOLO_IMGSZ", img_size))

        self.model = YOLO(self.model_name)

    def _image_from_webots(self, image, width, height):
        if image is None or width is None or height is None:
            return None
        frame = np.frombuffer(image, dtype=np.uint8)
        expected = int(width * height * 4)
        if frame.size != expected:
            return None
        frame = frame.reshape((height, width, 4))
        frame = frame[:, :, :3]  # BGRA -> BGR
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
            verbose=False,
        )

        detections = []
        if results:
            res = results[0]
            boxes = res.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), score, cls_id in zip(xyxy, confs, clss):
                    label = res.names.get(int(cls_id), str(int(cls_id)))
                    detections.append(
                        {
                            "bbox_xyxy": [float(x1), float(y1), float(x2), float(y2)],
                            "confidence": float(score),
                            "class_id": int(cls_id),
                            "label": label,
                        }
                    )

        return DetectorOutput(detections=detections, embedding=None, raw=results)
