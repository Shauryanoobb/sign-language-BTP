import tensorflow as tf
import cv2
import numpy as np

# Path to your trained model
MODEL_PATH = "tinyvit_student_final_synthetic.keras"

# Load model
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
num_classes = model.output_shape[-1]
print("✅ Model loaded. Num classes:", num_classes)

# ASL class labels (29 total)
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z',
]

# Preprocess frame
def preprocess_frame(frame, target_size=(128,128)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Video file path
VIDEO_PATH = "test.mp4"

cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise ValueError(f"Error opening video file {VIDEO_PATH}")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess & predict
    img = preprocess_frame(frame)
    pred = model.predict(img, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    pred_label = CLASS_NAMES[pred_class]

    # Print predictions
    print("Predicted class:", pred_label)

cap.release()
print("✅ Video processing complete.")
