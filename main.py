import cv2
from lane_detection import detect_lanes
from car_detection import detect_cars, model
from utils import draw_boxes, draw_center_path

cap = cv2.VideoCapture("videos/test.mp4")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (1280, 720))

    # 🔥 Lane detection (returns frame + lines)
    lane_frame, lines = detect_lanes(frame)

    # 🔥 Draw center guidance arrows
    if lines is not None and len(lines) == 2:
        lane_frame = draw_center_path(lane_frame, lines[0], lines[1])

    # 🔥 Car detection
    results = detect_cars(frame)

    # 🔥 Draw bounding boxes + distance
    output = draw_boxes(lane_frame, results, model.names)

    # Display
    cv2.imshow("ADAS Lane + Car Detection", output)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()