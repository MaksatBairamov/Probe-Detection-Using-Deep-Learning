Probe Detection Using Deep Learning

Project Overview:

This project applies YOLOv8 to detect probes in images with precision and efficiency. It includes dataset preparation, training scripts, evaluation metrics, and result visualization.

Installation: Clone the repository:

git clone https://github.com/MaksatBairamov/probe-detection.git
cd probe-detection
Install dependencies:

pip install -r requirements.txt
How to Run? Train the model:

  yolo train model=yolov8n.pt data=probe_dataset.yaml epochs=50
Run inference:

  yolo predict model=best.pt source=dataset/images/val
Results: Object detection accuracy visualized in results/. Metrics include precision, recall, and inference speed.