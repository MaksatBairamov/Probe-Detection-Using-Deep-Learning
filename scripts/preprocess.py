import os
import json
import torch
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

IMAGES_PATH = "dataset/probe_dataset/probe_images"
LABELS_PATH = "dataset/probe_dataset/probe_labels.json"
PREPROCESSED_DATA_PATH = "dataset/preprocessed_data.pt"

try:
    with open(LABELS_PATH, 'r') as f:
        data = json.load(f)
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Labels file not found at {LABELS_PATH}")
except json.JSONDecodeError as e:
    raise ValueError(f"❌ Error decoding JSON: {e}")

annotations = data.get('annotations', [])
if not annotations:
    raise ValueError("❌ No annotations found in the JSON file.")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def preprocess_image(image_path):

    try:
        img = Image.open(image_path).convert('RGB')
        return transform(img)
    except (FileNotFoundError, UnidentifiedImageError) as e:
        print(f"❌ Error processing {image_path}: {e}")
        return None



X, y = [], []

for ann in tqdm(annotations, desc="🚀 Processing images"):
    
    image_info = next(
        (img for img in data['images'] if img['id'] == ann['image_id']), 
        None
    )
    
    if image_info is None:
        print(f"⚠️ Image ID {ann['image_id']} not found in images metadata.")
        continue
    
    image_path = os.path.join(IMAGES_PATH, image_info['file_name'])
    image = preprocess_image(image_path)
    
    if image is not None:
        X.append(image)
        y.append(torch.tensor(ann['bbox'], dtype=torch.float32))
    else:
        print(f"⚠️ Skipping {image_path} due to preprocessing error.")


if X and y:
    torch.save((torch.stack(X), torch.stack(y)), PREPROCESSED_DATA_PATH)
    print(f"✅ Data saved to {PREPROCESSED_DATA_PATH}")
else:
    raise RuntimeError("❌ No valid data to save. Please check the image paths and annotations.")
