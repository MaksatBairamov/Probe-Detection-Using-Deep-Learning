import json
import os

ANNOTATIONS_FILE = 'dataset/probe_dataset/probe_labels.json'
IMAGES_DIR = 'dataset/probe_dataset/probe_images'

def fix_annotations():
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    images = data.get('images', [])
    
    valid_image_ids = {img['id'] for img in images}
    
    for annotation in annotations:
        if 'category_id' not in annotation:
            annotation['category_id'] = 1  
        
        if annotation['image_id'] not in valid_image_ids:
            print(f"⚠️ Попередження: image_id {annotation['image_id']} не знайдено у списку зображень!")
            continue
    
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("✅ Annotations refreshed!")

if __name__ == '__main__':
    fix_annotations()
