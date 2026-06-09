import cv2
import json

from src.detector import VehicleDetector
from src.classifier import VehicleClassifier

with open("artifacts/class_indices.json", "r") as f:
    class_indices = json.load(f)

idx_to_class = {
    v: k
    for k, v in class_indices.items()
}

detector = VehicleDetector(
    model_path="yolov8n.pt",
    conf=0.4
)

classifier = VehicleClassifier(
    model_path="models/vehicle_classifier_best.keras",
    idx_to_class=idx_to_class
)


cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Could not open camera")
    exit()

window_name = "Real-Time Vehicle Classification"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN
)

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    detections = detector.detect(frame)

    for detection in detections:

        x1, y1, x2, y2 = detection["box"]
        yolo_class = detection["yolo_class"]
        yolo_conf = detection["confidence"]

        crop = frame[y1:y2, x1:x2]

        cnn_class, cnn_conf = classifier.predict(crop)

        label = (
            f"YOLO:{yolo_class} ({yolo_conf:.2f}) | "
            f"CNN:{cnn_class} ({cnn_conf:.2f})"
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    cv2.imshow(
        window_name,
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()