import tensorflow as tf
import cv2
import numpy as np
import os

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
    img = img.astype("float32") #check whether /255.0 is needed
    img = np.expand_dims(img, axis=0)
    return img

# Video file path
VIDEO_PATH = "test1.mp4"  # or 0 for webcam
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error opening video file {VIDEO_PATH}")

# VideoWriter setup
fps = cap.get(cv2.CAP_PROP_FPS) or 20  # fallback if FPS not detected
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_video_path = "output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # compatible with macOS
out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

frame_count = 0
predict_every_n_frames = 5
last_pred_label = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Predict every N frames
    if frame_count % predict_every_n_frames == 0:
        img_input = preprocess_frame(frame)
        pred = model.predict(img_input, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        pred_label = CLASS_NAMES[pred_class]
        if pred_label != last_pred_label:
            last_pred_label = pred_label
    else:
        pred_label = last_pred_label

    # Draw bounding box + label
    if pred_label:
        h, w, _ = frame.shape
        cv2.rectangle(frame, (10,10), (w-10, h-10), (0,255,0), 2)
        cv2.putText(frame, f"Prediction: {pred_label}", (20,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)

    # Write frame to video
    out.write(frame)
    frame_count += 1

cap.release()
out.release()
print(f"✅ Video processing complete. Output saved at {output_video_path}")
