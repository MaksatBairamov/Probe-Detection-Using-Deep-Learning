import json

ANNOTATIONS_FILE = 'dataset/probe_dataset/probe_labels.json'

def fix_category_ids():
    with open(ANNOTATIONS_FILE, 'r') as f:
        data = json.load(f)

    valid_categories = {cat['id'] for cat in data.get('categories', [])}

    for annotation in data.get('annotations', []):
        if annotation['category_id'] not in valid_categories:
            annotation['category_id'] = list(valid_categories)[0]  
            print(f"🛠️ Changed category_id for image_id: {annotation['image_id']}")

    with open(ANNOTATIONS_FILE, 'w') as f:
        json.dump(data, f, indent=4)

    print("✅ All incorrect category_id were refreshed to a catalog dataset.")

if __name__ == '__main__':
    fix_category_ids()
