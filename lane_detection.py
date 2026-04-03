import cv2
import numpy as np

# Global memory for previous lanes
prev_left = None
prev_right = None


def region_of_interest(img):
    height, width = img.shape
    mask = np.zeros_like(img)

    polygon = np.array([[
        (0, height),
        (width, height),
        (int(width * 0.55), int(height * 0.6)),
        (int(width * 0.45), int(height * 0.6))
    ]], np.int32)

    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(img, mask)


def make_coordinates(image, line_params):
    slope, intercept = line_params
    y1 = image.shape[0]
    y2 = int(y1 * 0.6)

    if slope == 0:
        return None

    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)

    return np.array([x1, y1, x2, y2])


def average_slope_intercept(image, lines):
    global prev_left, prev_right

    left_fit = []
    right_fit = []

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)

            if x1 == x2:
                continue

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            if abs(slope) < 0.5:
                continue

            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))

    left_line = np.mean(left_fit, axis=0) if left_fit else None
    right_line = np.mean(right_fit, axis=0) if right_fit else None

    # 🔥 Smoothing
    alpha = 0.8

    if prev_left is not None and left_line is not None:
        left_line = alpha * prev_left + (1 - alpha) * left_line

    if prev_right is not None and right_line is not None:
        right_line = alpha * prev_right + (1 - alpha) * right_line

    # 🔥 Handle missing (dashed gaps)
    if left_line is None and prev_left is not None:
        left_line = prev_left

    if right_line is None and prev_right is not None:
        right_line = prev_right

    if left_line is not None:
        prev_left = left_line

    if right_line is not None:
        prev_right = right_line

    lines_out = []

    if left_line is not None:
        coords = make_coordinates(image, left_line)
        if coords is not None:
            lines_out.append(coords)

    if right_line is not None:
        coords = make_coordinates(image, right_line)
        if coords is not None:
            lines_out.append(coords)

    return lines_out


def detect_lanes(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    edges = cv2.Canny(blur, 50, 150)

    # 🔥 Fill dashed gaps
    kernel = np.ones((5, 5), np.uint8)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

    cropped = region_of_interest(edges)

    lines = cv2.HoughLinesP(
        cropped,
        rho=2,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=50,
        maxLineGap=80
    )

    averaged_lines = average_slope_intercept(frame, lines)

    line_image = np.zeros_like(frame)

    # Draw lane lines
    for line in averaged_lines:
        x1, y1, x2, y2 = line
        cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 8)

    # 🔥 Transparent lane fill (ADAS style)
    overlay = frame.copy()

    if len(averaged_lines) == 2:
        left = averaged_lines[0]
        right = averaged_lines[1]

        pts = np.array([[
            (left[0], left[1]),
            (left[2], left[3]),
            (right[2], right[3]),
            (right[0], right[1])
        ]], dtype=np.int32)

        cv2.fillPoly(overlay, pts, (0, 255, 0))

        # transparency
        frame = cv2.addWeighted(overlay, 0.25, frame, 0.75, 0)

    # Combine lines + frame
    final = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # 🔥 IMPORTANT: return both
    return final, averaged_lines