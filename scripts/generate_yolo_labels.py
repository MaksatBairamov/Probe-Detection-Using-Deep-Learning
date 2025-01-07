import json
import os


json_file = 'D:/proj/dataset/probe_dataset/probe_labels.json'
train_labels_folder = 'D:/proj/dataset/probe_dataset/labels/train'
val_labels_folder = 'D:/proj/dataset/probe_dataset/labels/val'
train_images_folder = 'D:/proj/dataset/probe_dataset/images/train'
val_images_folder = 'D:/proj/dataset/probe_dataset/images/val'


os.makedirs(train_labels_folder, exist_ok=True)
os.makedirs(val_labels_folder, exist_ok=True)


with open(json_file, 'r') as f:
    data = json.load(f)


for image_name, annotations in data.items():
    if os.path.exists(os.path.join(train_images_folder, image_name)):
        txt_file = os.path.join(train_labels_folder, image_name.replace('.jpg', '.txt'))
    elif os.path.exists(os.path.join(val_images_folder, image_name)):
        txt_file = os.path.join(val_labels_folder, image_name.replace('.jpg', '.txt'))
    else:
        print(f"⚠️ Зображення {image_name} не знайдено у train/val!")
        continue

    
    with open(txt_file, 'w') as f:
        for annotation in annotations:
            
            class_id = annotation.get('class_id') or annotation.get('label')
            bbox = annotation.get('bbox') or annotation.get('coordinates')

            if class_id is not None and bbox is not None:
                x_center, y_center, width, height = bbox
                f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
            else:
                print(f"⚠️ Пропущено анотацію для {image_name}: {annotation}")
