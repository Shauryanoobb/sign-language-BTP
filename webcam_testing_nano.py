import cv2
import numpy as np
import tensorflow as tf

# -----------------------------
# Load your trained model
# -----------------------------
MODEL_PATH = "mobilenetv2_mendeley_13signs.keras"
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
num_classes = model.output_shape[-1]

print("✅ Model loaded. Num classes:", num_classes)

# ASL class labels (make sure order matches training)
CLASS_NAMES = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'R', 'X']

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_frame(frame, target_size=(128, 128)):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   # Convert to RGB
    img = cv2.resize(img, target_size)             # Resize to model input
    img = img.astype("float32")                    # Convert to float
    img = np.expand_dims(img, axis=0)              # Add batch dim
    return img

# -----------------------------
# Webcam live testing
# -----------------------------
cap = cv2.VideoCapture(0)  # Open webcam (0 = default camera)

# Define bounding box (center of the frame)
box_size = 224
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
x1 = frame_w // 2 - box_size // 2
y1 = frame_h // 2 - box_size // 2
x2 = x1 + box_size
y2 = y1 + box_size

print("📷 Webcam started. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Draw bounding box
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Crop hand region
    hand_region = frame[y1:y2, x1:x2]

    if hand_region.shape[0] > 0 and hand_region.shape[1] > 0:
        # Preprocess
        img = preprocess_frame(hand_region, target_size=(128, 128))

        # Predict
        pred = model.predict(img, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)

        # Display prediction
        text = f"{CLASS_NAMES[pred_class]} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame
    cv2.imshow("ASL Live Testing", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
