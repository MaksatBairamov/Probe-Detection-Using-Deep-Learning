from ultralytics import YOLO
import os

model = YOLO('results/yolov8_probe/weights/best.pt')

metrics = model.val(
    data='dataset/probe_dataset/probe_labels.json',
    imgsz=640,
    device=0
)

print(metrics)
