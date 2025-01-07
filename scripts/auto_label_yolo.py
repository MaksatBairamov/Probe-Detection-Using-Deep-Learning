import os
import json
from ultralytics import YOLO
import torch

MODEL_PATH = "yolov8n.pt"
IMAGES_DIR = "dataset/probe_dataset/probe_images"
OUTPUT_JSON = "dataset/probe_dataset/probe_labels_autogen_yolo.json"

CONF_THRESHOLD = 0.25  

def pseudo_label_with_yolo(model_path, images_dir, conf_thr=0.25):

    model = YOLO(model_path)

    image_files = [f for f in os.listdir(images_dir) if f.lower().endswith('.jpg')]
    annotations = []

    for img_file in image_files:
        full_path = os.path.join(images_dir, img_file)
        results = model(full_path, conf=conf_thr)

        best_conf = 0.0
        best_bbox = [0, 0, 0, 0]
        for box in results[0].boxes:
            conf = float(box.conf[0].cpu().numpy())
            if conf > best_conf:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                best_conf = conf
                best_bbox = [int(x1), int(y1), int(x2), int(y2)]

        image_id = os.path.splitext(img_file)[0]
        annotations.append({
            "image_id": image_id,
            "bbox": best_bbox
        })

    return annotations

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"✅ Using device: {device}")

    auto_annotations = pseudo_label_with_yolo(MODEL_PATH, IMAGES_DIR, CONF_THRESHOLD)

    out_data = {"annotations": auto_annotations}

    with open(OUTPUT_JSON, "w") as f:
        json.dump(out_data, f, indent=4)

    print(f"✅ Pseudo-labeling done! Result saved to {OUTPUT_JSON}")