from ultralytics import YOLO
import cv2

model = YOLO('yolov8n.pt')
cap = cv2.VideoCapture('uploads/lane1.mp4')
ret, frame = cap.read()

if ret:
    results = model(frame, verbose=False)[0]
    count = sum(1 for box in results.boxes if int(box.cls[0]) in [2,3,5,7])
    print(f'SUCCESS! Lane 1: detected {count} vehicles in first frame')
else:
    print('ERROR: Could not read video')

cap.release()