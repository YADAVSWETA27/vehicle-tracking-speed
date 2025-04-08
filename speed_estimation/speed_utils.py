import math

def get_centroid(box):
    """Get the centroid (x, y) of a bounding box."""
    x1, y1, x2, y2 = box
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)
    return (cx, cy)

def calculate_speed(prev_pos, curr_pos, fps, ppm):
    """Calculate speed given previous and current positions."""
    dx = curr_pos[0] - prev_pos[0]
    dy = curr_pos[1] - prev_pos[1]
    pixel_distance = math.hypot(dx, dy)  # Euclidean distance

    meters = pixel_distance / ppm
    seconds = 1 / fps
    speed_mps = meters / seconds
    speed_kmph = speed_mps * 3.6

    return round(speed_kmph, 2)
    