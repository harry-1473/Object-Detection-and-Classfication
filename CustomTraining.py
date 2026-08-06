import os
import random
import shutil
import torch
from ultralytics import YOLO

print("🖥️ Device:", "GPU" if torch.cuda.is_available() else "CPU")

dataset_folder = 'recycle_dataset'
classes = ['plastic', 'glass', 'metal']
train_ratio = 0.8

# Clean previous dataset if needed
if os.path.exists(dataset_folder):
    shutil.rmtree(dataset_folder)

os.makedirs(f'{dataset_folder}/images/train', exist_ok=True)
os.makedirs(f'{dataset_folder}/images/val', exist_ok=True)
os.makedirs(f'{dataset_folder}/labels/train', exist_ok=True)
os.makedirs(f'{dataset_folder}/labels/val', exist_ok=True)

raw_images_dir = 'raw_images'
raw_labels_dir = 'raw_labels'

images = [f for f in os.listdir(raw_images_dir) if f.lower().endswith('.jpg')]
random.shuffle(images)

train_count = int(len(images) * train_ratio)
train_images = images[:train_count]
val_images = images[train_count:]

for img in train_images:
    shutil.copy(f'{raw_images_dir}/{img}', f'{dataset_folder}/images/train/{img}')
    label_file = img.replace('.jpg', '.txt')
    shutil.copy(f'{raw_labels_dir}/{label_file}', f'{dataset_folder}/labels/train/{label_file}')

for img in val_images:
    shutil.copy(f'{raw_images_dir}/{img}', f'{dataset_folder}/images/val/{img}')
    label_file = img.replace('.jpg', '.txt')
    shutil.copy(f'{raw_labels_dir}/{label_file}', f'{dataset_folder}/labels/val/{label_file}')

print("✅ Files copied and dataset structured.")

yaml_content = f"""
path: D:/python/ObjectDetectionModel/recycle_dataset

train: images/train
val: images/val

names:
"""
for idx, class_name in enumerate(classes):
    yaml_content += f"  {idx}: {class_name}\n"

with open(f'{dataset_folder}/data.yaml', 'w') as f:
    f.write(yaml_content.strip())

print("✅ data.yaml created.")

# Use YOLOv8 Nano model (lightweight)
model = YOLO('yolov8n.pt')

model.train(
    data='recycle_dataset/data.yaml',
    epochs=80,
    imgsz=512,
    batch=4,
    patience=20,
    degrees=10,
    flipud=0.2,
    hsv_h=0.015,
    hsv_s=0.6,
    hsv_v=0.4
)

print("🏁 Training started!")
