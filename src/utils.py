import cv2

def draw_detection(frame, detection):
    x1, y1, x2, y2 = detection["box"]
    class_name = detection["class_name"]
    confidence = detection["confidence"]
    label = f"{class_name} {confidence:.2f}"

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
        0.7,
        (0, 255, 0),
        2
    )

    return frame