import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


class ProbeDataset(Dataset):
    def __init__(self, image_dir, label_path, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        
        with open(label_path, "r") as f:
            data = json.load(f)
        
        self.image_metadata = {img["id"]: img["file_name"] for img in data["images"]}
        self.annotations = {ann["image_id"]: ann["bbox"] for ann in data["annotations"]}
        
        self.valid_images = []
        for img_id, file_name in self.image_metadata.items():
            image_path = os.path.join(self.image_dir, file_name)
            if os.path.exists(image_path):
                self.valid_images.append((img_id, image_path))
            else:
                print(f"[WARNING] Missing image file: {file_name}")
        
        if not self.valid_images:
            raise ValueError("Dataset is empty! Check image paths and labels file.")

    def __len__(self):
        return len(self.valid_images)

    def __getitem__(self, idx):
        img_id, img_path = self.valid_images[idx]
        image = Image.open(img_path).convert("RGB")
        bbox = self.annotations.get(img_id, [0, 0, 0, 0])
        
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(bbox, dtype=torch.float32)



class YOLOCustomModel(nn.Module):
    def __init__(self):
        super(YOLOCustomModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(32 * 224 * 224, 128)
        self.fc2 = nn.Linear(128, 4) 

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train():
    image_dir = "dataset/probe_dataset/probe_images"
    label_path = "dataset/probe_dataset/probe_labels.json"
    batch_size = 8
    num_epochs = 5
    learning_rate = 0.001
    weight_decay = 0.01  

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])


    dataset = ProbeDataset(image_dir, label_path, transform)
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,  
        pin_memory=True
    )

    model = YOLOCustomModel().to(device)
    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for i, (images, targets) in enumerate(data_loader):
            images, targets = images.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        avg_loss = running_loss / len(data_loader)
        print(f"[INFO] Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    # Save Model
    torch.save(model.state_dict(), "models/model.pth")
    print("[INFO] Training complete. Model saved as 'models/model.pth'")



if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"[ERROR] {e}")
