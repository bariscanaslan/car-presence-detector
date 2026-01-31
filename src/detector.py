# src/detector.py

from ultralytics import YOLO

MODEL_PATH = "models/yolo_car.pt"
DEVICE = "cpu"


class YOLODetector:
    def __init__(self):
        print("Loading YOLO model...")
        self.model = YOLO(MODEL_PATH)
        self.model.to(DEVICE)

    def detect(self, image):
        print("Running inference...")
        results = self.model(image, verbose=False)

        detections = []

        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])

                detections.append({
                    "class_id": class_id,
                    "confidence": confidence,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)]
                })

        return detections
