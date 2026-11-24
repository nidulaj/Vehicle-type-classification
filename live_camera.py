import cv2

from src.pipeline import VehiclePipeline

# Initialize the two-stage pipeline from centralized configuration
pipeline = VehiclePipeline.from_config()

# Camera initialization (try camera 1 first, fallback to camera 0 if not opened)
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Could not open camera")
    exit()

window_name = "Real-Time Vehicle Classification"

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL,
)

cv2.setWindowProperty(
    window_name,
    cv2.WND_PROP_FULLSCREEN,
    cv2.WINDOW_FULLSCREEN,
)

while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to read frame")
        break

    # Process frame through two-stage pipeline (detection -> safe cropping -> MobileNetV2)
    predictions = pipeline.process_frame(frame)

    for pred in predictions:
        x1, y1, x2, y2 = pred.box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            frame,
            pred.display_label,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    cv2.imshow(
        window_name,
        frame,
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
