import json
import os


DATASET_PATH = 'dataset/probe_dataset'
LABELS_PATH = os.path.join(DATASET_PATH, 'probe_labels.json')
IMAGES_PATH = os.path.join(DATASET_PATH, 'probe_images')


def validate_labels():
    with open(LABELS_PATH, 'r') as f:
        labels = json.load(f)

    valid_labels = {}
    for image_name, bbox in labels.items():
        image_path = os.path.join(IMAGES_PATH, image_name)
        if os.path.exists(image_path):
            if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                valid_labels[image_name] = bbox
            else:
                print(f"Invalid bbox for {image_name}: {bbox}")
        else:
            print(f"Missing image file: {image_path}")

    with open(LABELS_PATH, 'w') as f:
        json.dump(valid_labels, f, indent=4)
    print("Annotation sync complete. Valid labels updated.")

if __name__ == '__main__':
    validate_labels()
