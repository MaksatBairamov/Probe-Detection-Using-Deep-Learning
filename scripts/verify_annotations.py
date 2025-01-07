import json

ANNOTATIONS_FILE = 'dataset/probe_dataset/probe_labels.json'

def verify_annotations():
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)
    
    annotations = data.get('annotations', [])
    images = {img['id']: img for img in data.get('images', [])}
    categories = {cat['id']: cat for cat in data.get('categories', [])}

    valid_annotations = []
    for annotation in annotations:
        if annotation['image_id'] in images and annotation['category_id'] in categories:
            valid_annotations.append(annotation)
        else:
            print(f"❌ Некоректна анотація: {annotation}")

    data['annotations'] = valid_annotations
    
    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    
    print("✅ Анотації перевірено та оновлено!")

if __name__ == '__main__':
    verify_annotations()
