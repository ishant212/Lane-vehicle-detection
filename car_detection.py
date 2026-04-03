from ultralytics import YOLO

model = YOLO("weights/yolov8n.pt")

def detect_cars(frame):
    results = model(frame)[0]
    return results