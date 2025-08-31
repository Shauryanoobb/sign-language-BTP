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

# ASL class labels
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z',
]

# Preprocessing function for webcam frame
def preprocess_frame(frame, target_size=(128, 128)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR->RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)  # (1,128,128,3)
    return img

# Start webcam
cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("❌ Error: Could not open webcam.")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to grab frame.")
        break

    # Preprocess frame
    img = preprocess_frame(frame)

    # Model prediction
    pred = model.predict(img, verbose=0)
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    # Display prediction on frame
    text = f"{CLASS_NAMES[pred_class]} ({confidence:.2f})"
    cv2.putText(frame, text, (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("ASL Prediction", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam & close windows
cap.release()
cv2.destroyAllWindows()
