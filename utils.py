import cv2
import numpy as np
from distance_estimator import estimate_distance


# 🔥 Color based on distance (ADAS style)
def get_color(distance):
    if distance < 15:
        return (0, 0, 255)   # RED (very close)
    elif distance < 30:
        return (0, 165, 255) # ORANGE (medium)
    else:
        return (0, 255, 0)   # GREEN (safe)


# 🔥 Draw car bounding boxes with distance coloring
def draw_boxes(frame, results, model_names):
    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])

        if model_names[cls] == "car" and conf > 0.5:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            width = x2 - x1
            distance = estimate_distance(width)

            color = get_color(distance)

            label = f"car {conf:.2f}"
            dist_label = f"{distance} m"

            # Bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Top label
            cv2.putText(frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Distance label (below box)
            cv2.putText(frame, dist_label, (x1, y2 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame


# 🔥 Draw center path arrows (ADAS guidance)
def draw_center_path(frame, left_line, right_line):
    if left_line is None or right_line is None:
        return frame

    h = frame.shape[0]

    # Midpoints
    mid_bottom = (
        (left_line[0] + right_line[0]) // 2,
        h
    )
    mid_top = (
        (left_line[2] + right_line[2]) // 2,
        int(h * 0.6)
    )

    # Draw arrows along path
    for i in range(12):
        y = int(h - i * 40)

        if y < mid_top[1]:
            break

        x = int(np.interp(
            y,
            [mid_top[1], mid_bottom[1]],
            [mid_top[0], mid_bottom[0]]
        ))

        pts = np.array([
            [x, y],
            [x - 12, y + 25],
            [x + 12, y + 25]
        ], np.int32)

        cv2.fillPoly(frame, [pts], (0, 255, 0))

    return frame