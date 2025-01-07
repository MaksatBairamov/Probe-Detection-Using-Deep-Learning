from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('models/yolov8n.pt')

    model.train(
        data='dataset/probe_dataset/probe_dataset.yaml',
        epochs=50,
        batch=8,
        imgsz=640,
        device=0,
        workers=0, 
        project='results',
        name='yolov8_probe4',
        pretrained=True,
        deterministic=True
    )
