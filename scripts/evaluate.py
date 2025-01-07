import torch
from torch import nn
import os
import json
from tqdm import tqdm
import argparse
from PIL import Image, UnidentifiedImageError
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.metrics import precision_score, recall_score, f1_score

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")


class YOLOCustomModel(nn.Module):
    def __init__(self):
        super(YOLOCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 224 * 224, 128)
        self.fc2 = nn.Linear(128, 4)  # [x1, y1, x2, y2]

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def load_model(model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"❌ Model file not found at {model_path}")
    
    model = YOLOCustomModel()
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print("✅ Model loaded successfully.")
    return model


from torchvision import transforms

def preprocess_image(image_path, resize_size=(224, 224)):
    transform = transforms.Compose([
        transforms.Resize(resize_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    try:
        image = Image.open(image_path).convert('RGB')
        tensor = transform(image).unsqueeze(0).to(device)
        return tensor, image.size, image
    except (FileNotFoundError, UnidentifiedImageError) as e:
        raise RuntimeError(f"❌ Error processing image {image_path}: {e}")


def scale_bbox(bbox_224, orig_size):
    width, height = orig_size
    x1 = int(bbox_224[0] * width / 224)
    y1 = int(bbox_224[1] * height / 224)
    x2 = int(bbox_224[2] * width / 224)
    y2 = int(bbox_224[3] * height / 224)
    return [x1, y1, x2, y2]


def calculate_iou(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def show_image_with_bbox(image, image_id, pred_bbox=None, true_bbox=None):
    _, ax = plt.subplots(1, figsize=(8, 8))
    ax.imshow(image)
    
    if pred_bbox:
        rect_pred = patches.Rectangle(
            (pred_bbox[0], pred_bbox[1]),
            pred_bbox[2] - pred_bbox[0],
            pred_bbox[3] - pred_bbox[1],
            linewidth=2, edgecolor='red', facecolor='none', label='Predicted'
        )
        ax.add_patch(rect_pred)
    
    if true_bbox:
        rect_true = patches.Rectangle(
            (true_bbox[0], true_bbox[1]),
            true_bbox[2] - true_bbox[0],
            true_bbox[3] - true_bbox[1],
            linewidth=2, edgecolor='green', facecolor='none', label='Ground Truth'
        )
        ax.add_patch(rect_true)
    
    plt.legend()
    output_path = os.path.join("results", f"{image_id}_result.png")
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True)
    parser.add_argument('--annotations_path', required=True)
    parser.add_argument('--images_path', required=True)
    args = parser.parse_args()
    
    model = load_model(args.model_path)
    with open(args.annotations_path, 'r') as f:
        annotations_data = json.load(f)
    
    iou_scores = []
    for annotation in tqdm(annotations_data['annotations']):
        image_id = annotation['image_id']
        image_info = next(img for img in annotations_data['images'] if img['id'] == image_id)
        image_path = os.path.join(args.images_path, image_info['file_name'])
        gt_bbox = annotation['bbox']
        
        tensor, orig_size, image = preprocess_image(image_path)
        with torch.no_grad():
            outputs = model(tensor).cpu().numpy()[0]
        pred_bbox = scale_bbox(outputs, orig_size)
        true_bbox = scale_bbox(gt_bbox, orig_size)
        
        iou = calculate_iou(pred_bbox, true_bbox)
        iou_scores.append(iou)
        show_image_with_bbox(image, pred_bbox, true_bbox)
    
    print(f'Average IoU: {np.mean(iou_scores):.4f}')