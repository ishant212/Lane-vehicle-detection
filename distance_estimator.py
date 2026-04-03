def estimate_distance(bbox_width):
    focal_length = 800   # tune this
    known_width = 2.0    # avg car width in meters

    if bbox_width == 0:
        return 0

    distance = (known_width * focal_length) / bbox_width
    return round(distance, 2)