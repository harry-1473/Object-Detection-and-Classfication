import cv2
import time
import threading
import serial
from ultralytics import YOLO
import tkinter as tk
from tkinter import Label
from collections import Counter
import qrcode
from PIL import Image, ImageTk
import json

# --- Globals ---
running = False
arduino = None
model = None
cap = None
interval = 5
last_detection_time = 0
annotated_frame = None
detections_log = []  # ✅ store detections


# --- YOLO Detection Thread ---
def run_detection():
    global running, arduino, model, cap, last_detection_time, annotated_frame, detections_log

    detections_log.clear()  # clear previous session logs

    while running:
        ret, frame = cap.read()
        if not ret:
            continue

        current_time = time.time()

        if current_time - last_detection_time >= interval:
            results = model.predict(frame, stream=False, verbose=False)
            last_detection_time = current_time

            for r in results:
                annotated_frame = r.plot()
                boxes = r.boxes

                if boxes is not None and boxes.cls.numel() > 0:
                    confidences = boxes.conf.cpu().numpy()
                    best_idx = confidences.argmax()
                    class_id = int(boxes.cls[best_idx])
                    class_name = model.names[class_id]
                    confidence = confidences[best_idx]

                    print(f"Detected: {class_name} ({confidence:.2f})")

                    # ✅ Record detection
                    detections_log.append(class_name)

                    # ✅ Send to Arduino
                    if arduino:
                        arduino.write(f"{class_name}\n".encode())
                        print(f"Sent to Arduino: {class_name}")
        else:
            if annotated_frame is None:
                annotated_frame = frame.copy()

        if annotated_frame is not None:
            cv2.imshow("Waste Type Classification", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False
            break

    cap.release()
    if arduino:
        arduino.close()
    cv2.destroyAllWindows()

    # --- After detection ends ---
    generate_qr_summary()


# --- QR Code Generator (embed in GUI) ---
def generate_qr_summary():
    global detections_log

    if not detections_log:
        print("⚠️ No detections recorded. QR not generated.")
        label_qr.config(text="⚠️ No detections recorded.", image="", compound="center")
        return

    print("\n📦 Session Summary:")
    count_summary = Counter(detections_log)
    for k, v in count_summary.items():
        print(f"{k}: {v}")

    # Prepare QR text (JSON-like single-line text for scanner readability)
    qr_text = json.dumps(count_summary, separators=(",", ":"))

    # Generate QR code
    qr_img = qrcode.make(qr_text)
    qr_img = qr_img.resize((200, 200))
    qr_photo = ImageTk.PhotoImage(qr_img)

    # Display in GUI
    label_qr.config(image=qr_photo, text="")
    label_qr.image = qr_photo

    print("\n✅ QR code generated and displayed in GUI.")


# --- GUI Callbacks ---
def start():
    global running, arduino, model, cap, last_detection_time
    if not running:
        running = True
        print("🚀 Starting YOLO detection...")
        arduino = serial.Serial('COM5', 9600)
        time.sleep(2)
        model = YOLO("runs/detect/train10/weights/best.pt")
        cap = cv2.VideoCapture(1)
        last_detection_time = 0
        threading.Thread(target=run_detection, daemon=True).start()


def end():
    global running
    if running:
        print("🛑 Stopping YOLO detection...")
        running = False


# --- Tkinter GUI ---
root = tk.Tk()
root.title("YOLO Waste Classifier")

image = Image.open("logo.png")
tk_image = ImageTk.PhotoImage(image)
label = Label(root, image=tk_image)
label.pack(pady=20)

tk.Label(root, text= "EcoKyats", font=("Arial", 16, "bold") ).pack(pady=10)

tk.Label(root, text="", font=("Arial", 16, "bold")).pack(pady=10)

start_button = tk.Button(root, text="Start Detection", width=25, height=2, bg="#4CAF50", fg="white", command=start)
start_button.pack(pady=10)

end_button = tk.Button(root, text="End & Generate QR", width=25, height=2, bg="#f44336", fg="white", command=end)
end_button.pack(pady=10)

tk.Label(root, text="Press 'End' after detection to create QR code", font=("Arial", 10, "italic")).pack(pady=10)

# --- QR Display Area ---
label_qr = Label(root, text="QR code will appear here", font=("Arial", 12))
label_qr.pack(pady=20)

root.mainloop()
