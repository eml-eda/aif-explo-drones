class DetectorOutput:
    def __init__(self, detections=None, embedding=None, raw=None):
        self.detections = detections or []
        self.embedding = embedding
        self.raw = raw


class Detector:
    def __init__(self, model_name="stub"):
        self.model_name = model_name

    def process(self, image, width=None, height=None, timestamp=None):
        return DetectorOutput(detections=[], embedding=None, raw=None)
