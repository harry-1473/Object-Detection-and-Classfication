from ultralytics import YOLO

model = YOLO('runs/detect/train10/weights/best.pt')
results = model('raw_images/IMG_20251010_231701.jpg', conf=0.1)  # Use any image you want
results[0].show()
