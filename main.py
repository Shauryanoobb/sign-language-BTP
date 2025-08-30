import tensorflow as tf
import cv2
import numpy as np
import os
import time

# Path to your trained model
MODEL_PATH = "tinyvit_student_final_synthetic.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
num_classes = model.output_shape[-1]
print("✅ Model loaded. Num classes:", num_classes)

# ASL class labels (26 letters)
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z'
]

# Preprocess frame for model
def preprocess_frame(frame, target_size=(128,128)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    #img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Create output folder
OUTPUT_DIR = "output_frames"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Video file path (replace with your webcam later if needed)
VIDEO_PATH = "test.mp4"
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error opening video file {VIDEO_PATH}")

frame_count = 0
predict_every_n_frames = 5  # throttle predictions to every 5 frames

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict only every N frames
    if frame_count % predict_every_n_frames == 0:
        img_input = preprocess_frame(frame)
        pred = model.predict(img_input, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_label = CLASS_NAMES[pred_class]
    else:
        pred_label = None  # keep last prediction if needed

    # Draw bounding box and label
    if pred_label:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (10,10), (w-10, h-10), (0,255,0), 2)
        cv2.putText(frame, f"Prediction: {pred_label}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    # Save frame
    frame_filename = os.path.join(OUTPUT_DIR, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_filename, frame)

    frame_count += 1

cap.release()
print(f"✅ Video processing complete. Frames saved in {OUTPUT_DIR}")
