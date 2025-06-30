# src/detector.py
from ultralytics import YOLO

class PlayerDetector:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
    
    def detect_players(self, frame):
        results = self.model(frame)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy()  # class id

            for box, conf, cls in zip(boxes, confs, classes):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                detections.append([[x1, y1, w, h], conf, int(cls)])  # DEEPSORT expects this format

        return detections
