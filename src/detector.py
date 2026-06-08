from ultralytics import YOLO

class VehicleDetector:

    def __init__(self, model_path="yolov8n.pt", conf=0.4):
        self.model = YOLO(model_path)
        self.conf = conf

        self.vehicle_classes = {
            2: "car",
            3: "motorcycle",
            5: "bus",
            7: "truck"
        }

    def detect(self, frame):

        results = self.model(
            frame,
            conf=self.conf,
            verbose=False
        )[0]

        detections = []

        for box in results.boxes:

            cls_id = int(box.cls[0])

            if cls_id in self.vehicle_classes:

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0]
                )

                detections.append({
                    "box": (x1, y1, x2, y2),
                    "confidence": float(box.conf[0]),
                    "yolo_class": self.vehicle_classes[cls_id]
                })

        return detections