from ultralytics import YOLO
import cv2
import numpy as np
import os
import csv
from tracking.sort import Tracker
from speed_estimation.speed_utils import get_centroid, calculate_speed

# Load YOLOv8 model
model = YOLO('models/yolov8x.pt')

# Initialize video and tracker
cap = cv2.VideoCapture('videos/traffic.mp4')
fps = cap.get(cv2.CAP_PROP_FPS)
ppm = 10  # pixels per meter (adjust based on video calibration)

# Setup output video writer
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('results/output_video.mp4', fourcc, fps, (width, height))

# Tracker setup
trackers = []
tracker_positions = {}  # {ID: (prev_centroid, frame_count)}

# Prepare results folder and CSV logging
os.makedirs("results", exist_ok=True)
csv_file = open("results/vehicle_log.csv", mode='w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["Vehicle ID", "Frame", "Video Time", "Speed (km/h)"])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)
    detections = results[0].boxes.xyxy.cpu().numpy()

    current_boxes = [np.array([int(x1), int(y1), int(x2), int(y2)]) for x1, y1, x2, y2, *_ in detections]

    new_trackers = []
    for bbox in current_boxes:
        found = False
        for t in trackers:
            pred = t.predict()
            pred_box = pred[0]
            iou_w = max(0, min(bbox[2], pred_box[2]) - max(bbox[0], pred_box[0]))
            iou_h = max(0, min(bbox[3], pred_box[3]) - max(bbox[1], pred_box[1]))
            iou = iou_w * iou_h

            if iou > 0:
                t.update(bbox)
                new_trackers.append(t)
                found = True
                break

        if not found:
            new_trackers.append(Tracker(np.array(bbox)))

    trackers = new_trackers

    total_speed = 0
    vehicle_count = 0
    frame_num = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    timestamp = round(frame_num / fps, 2)  # video time in seconds

    for t in trackers:
        x1, y1, x2, y2 = map(int, t.kf.x[:4].flatten())
        cx, cy = get_centroid((x1, y1, x2, y2))

        if t.id in tracker_positions:
            prev_cx, prev_cy, prev_frame = tracker_positions[t.id]
            speed = calculate_speed((prev_cx, prev_cy), (cx, cy), fps, ppm)
        else:
            speed = 0

        tracker_positions[t.id] = (cx, cy, frame_num)

        # Log to CSV
        csv_writer.writerow([t.id, frame_num, f"{timestamp}s", round(speed, 2)])

        # Draw on frame
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'ID:{t.id} Speed:{round(speed, 1)} km/h', (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        total_speed += speed
        vehicle_count += 1

    # Stats on screen
    avg_speed = total_speed / vehicle_count if vehicle_count > 0 else 0
    cv2.putText(frame, f'Total Vehicles: {vehicle_count}', (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(frame, f'Average Speed: {round(avg_speed, 2)} km/h', (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Save and show
    out.write(frame)
    cv2.imshow("Vehicle Tracking & Speed Estimation", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
out.release()
csv_file.close()
cv2.destroyAllWindows()
