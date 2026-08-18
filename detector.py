import cv2
import time
import numpy as np
from ultralytics import YOLO

model = YOLO('yolov8n.pt')

VEHICLE_CLASSES = {2: 'Car', 3: 'Motorcycle', 5: 'Bus', 7: 'Truck'}

MIN_GREEN = 5
MAX_GREEN = 30

BOX_COLORS = {
    2: (0, 255, 180),
    3: (255, 170, 0),
    5: (0, 170, 255),
    7: (255, 60, 0),
}
AMBULANCE_COLOR = (0, 0, 255)


def detect_emergency_lights(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 150])
    upper_blue = np.array([130, 255, 255])
    blue_mask  = cv2.inRange(hsv, lower_blue, upper_blue)

    lower_red1 = np.array([0,   150, 150])
    upper_red1 = np.array([10,  255, 255])
    lower_red2 = np.array([170, 150, 150])
    upper_red2 = np.array([180, 255, 255])
    red_mask   = cv2.inRange(hsv, lower_red1, upper_red1) + \
                 cv2.inRange(hsv, lower_red2, upper_red2)

    blue_pixels = cv2.countNonZero(blue_mask)
    red_pixels  = cv2.countNonZero(red_mask)

    h, w        = frame.shape[:2]
    total       = h * w
    blue_ratio  = blue_pixels / total
    red_ratio   = red_pixels  / total

    if blue_ratio > 0.003 or red_ratio > 0.005:
        return True
    return False


def detect_vehicles(frame):
    results = model(frame, verbose=False)[0]

    count              = 0
    ambulance_detected = False

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label  = model.names[cls_id].lower()

        if cls_id in VEHICLE_CLASSES:
            count += 1

        if 'ambulance' in label:
            ambulance_detected = True

    if not ambulance_detected:
        ambulance_detected = detect_emergency_lights(frame)

    return count, ambulance_detected, results


def draw_boxes(frame, results):
    annotated = frame.copy()

    for box in results.boxes:
        cls_id = int(box.cls[0])
        conf   = float(box.conf[0])
        label  = model.names[cls_id].lower()

        is_vehicle   = cls_id in VEHICLE_CLASSES
        is_ambulance = 'ambulance' in label

        if not (is_vehicle or is_ambulance):
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if is_ambulance:
            color = AMBULANCE_COLOR
            tag   = f'AMBULANCE {conf:.2f}'
        else:
            color = BOX_COLORS.get(cls_id, (0, 255, 180))
            tag   = f'{VEHICLE_CLASSES[cls_id]} {conf:.2f}'

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        corner = 10
        cv2.line(annotated, (x1, y1), (x1+corner, y1), color, 2)
        cv2.line(annotated, (x1, y1), (x1, y1+corner), color, 2)
        cv2.line(annotated, (x2, y1), (x2-corner, y1), color, 2)
        cv2.line(annotated, (x2, y1), (x2, y1+corner), color, 2)
        cv2.line(annotated, (x1, y2), (x1+corner, y2), color, 2)
        cv2.line(annotated, (x1, y2), (x1, y2-corner), color, 2)
        cv2.line(annotated, (x2, y2), (x2-corner, y2), color, 2)
        cv2.line(annotated, (x2, y2), (x2, y2-corner), color, 2)

        (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1-th-6), (x1+tw+6, y1), color, -1)
        cv2.putText(annotated, tag, (x1+3, y1-4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1, cv2.LINE_AA)

    return annotated


def draw_lane_overlay(frame, lane_num, count, signal, ambulance):
    h, w    = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, h-28), (w, h), (6, 15, 30), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

    sig_colors = {
        'green': (0, 255, 180),
        'red':   (0, 45, 255),
        'amber': (0, 170, 255),
    }
    sig_color = sig_colors.get(signal, (100, 100, 100))

    info = f'LANE {lane_num}  |  VEHICLES: {count:02d}  |  SIGNAL: {signal.upper()}'
    if ambulance:
        info     += '  |  !! AMBULANCE DETECTED !!'
        sig_color = (0, 0, 255)

    cv2.putText(frame, info, (10, h-10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, sig_color, 1, cv2.LINE_AA)
    return frame


def calculate_green_times(counts):
    total       = sum(counts.values())
    green_times = {}

    if total == 0:
        for lane in counts:
            green_times[lane] = MIN_GREEN
    else:
        for lane, count in counts.items():
            proportion        = count / total
            green_times[lane] = int(
                MIN_GREEN + proportion * (MAX_GREEN - MIN_GREEN)
            )
    return green_times


def process_all_lanes(video_paths, lane_data, latest_frames=None):
    caps = {}
    for lane, path in video_paths.items():
        if path is not None:
            cap = cv2.VideoCapture(path)
            if cap.isOpened():
                caps[lane] = cap
                print(f"Lane {lane} loaded: {path}")

    if not caps:
        print("No videos loaded.")
        return

    frame_skip   = 8
    frame_count  = 0

    current_green_lane       = None
    green_start_time         = 0
    current_green_time       = MIN_GREEN

    # Emergency cooldown — stops ambulance from locking green forever
    emergency_cooldown       = False
    emergency_cooldown_start = 0
    EMERGENCY_COOLDOWN_TIME  = 30  # seconds

    print("Detection started!")

    while True:
        frame_count += 1
        any_frame    = False
        counts       = {}
        ambulances   = {}
        frames       = {}

        for lane, cap in caps.items():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = cap.read()

            if ret:
                any_frame      = True
                frame          = cv2.resize(frame, (640, 360))
                frames[lane]   = frame

        if not any_frame:
            break

        if frame_count % frame_skip == 0:

            # Reset emergency cooldown after 30 seconds
            if emergency_cooldown:
                if time.time() - emergency_cooldown_start > EMERGENCY_COOLDOWN_TIME:
                    emergency_cooldown = False
                    print("Emergency cooldown ended — normal control resumed")

            for lane, frame in frames.items():
                count, ambulance, results = detect_vehicles(frame)
                counts[lane]             = count
                ambulances[lane]         = ambulance

                lane_data[lane]['count']     = count
                lane_data[lane]['ambulance'] = ambulance

                annotated = draw_boxes(frame, results)
                annotated = draw_lane_overlay(
                    annotated, lane, count,
                    lane_data[lane]['signal'], ambulance
                )

                if latest_frames is not None:
                    latest_frames[lane] = annotated

            # ── Signal control ────────────────────────────────
            emergency_lane = next(
                (l for l, a in ambulances.items() if a), None
            )

            if emergency_lane and not emergency_cooldown:
                # Emergency override
                print(f"AMBULANCE in Lane {emergency_lane}! Override!")
                for lane in lane_data:
                    lane_data[lane]['signal']     = 'red'
                    lane_data[lane]['green_time'] = 0
                lane_data[emergency_lane]['signal']     = 'green'
                lane_data[emergency_lane]['green_time'] = 15
                current_green_lane       = emergency_lane
                green_start_time         = time.time()
                current_green_time       = 15
                emergency_cooldown       = True
                emergency_cooldown_start = time.time()

            else:
                now     = time.time()
                elapsed = now - green_start_time

                if current_green_lane is None or elapsed >= current_green_time:
                    if counts:
                        green_times  = calculate_green_times(counts)
                        sorted_lanes = sorted(
                            counts.keys(),
                            key=lambda l: counts[l],
                            reverse=True
                        )

                        next_lane = sorted_lanes[0]
                        if next_lane == current_green_lane and len(sorted_lanes) > 1:
                            next_lane = sorted_lanes[1]

                        for lane in lane_data:
                            if lane == next_lane:
                                lane_data[lane]['signal']     = 'green'
                                lane_data[lane]['green_time'] = green_times[next_lane]
                            elif lane == current_green_lane:
                                lane_data[lane]['signal']     = 'amber'
                                lane_data[lane]['green_time'] = 0
                            else:
                                lane_data[lane]['signal']     = 'red'
                                lane_data[lane]['green_time'] = 0

                        current_green_lane  = next_lane
                        green_start_time    = now
                        current_green_time  = green_times[next_lane]

                        print(f"Lane {next_lane} GREEN for "
                              f"{green_times[next_lane]}s | counts: {counts}")

        else:
            for lane, frame in frames.items():
                if latest_frames is not None and latest_frames[lane] is None:
                    latest_frames[lane] = frame

        time.sleep(0.01)

    for cap in caps.values():
        cap.release()
    print("Processing complete.")