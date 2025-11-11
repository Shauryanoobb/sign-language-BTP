import cv2
import numpy as np
import tensorflow as tf
import string
from keras_vision.MobileViT_v1 import build_MobileViT_v1
from tensorflow.keras import layers, Model

# -----------------------------
# Load your trained model
# -----------------------------
# 1️⃣ Rebuild the architecture exactly as before
# 1️⃣ Build MobileViT backbone (no classification head)
backbone = build_MobileViT_v1(
    model_type="XXS",
    pretrained=False,
    include_top=False,
    num_classes=0
)

# 2️⃣ Define normalization and augmentation
normalization_layer = layers.Rescaling(1./255)

data_augmentation = tf.keras.Sequential([
    layers.RandomTranslation(0.05, 0.05),  # hand not centered
    layers.RandomRotation(0.05),           # slight tilt ±5°
])

# 3️⃣ Build the full model with augmentation + normalization
inputs = layers.Input(shape=(256, 256, 3))
x = normalization_layer(inputs)
x = data_augmentation(x)
x = backbone(x)  # feed into MobileViT backbone
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(26, activation='softmax')(x)  # 26 ASL classes
model = Model(inputs, outputs=x, name="MobileViT_ASL")

# 4️⃣ Build the model by passing a dummy input (important before loading weights)
dummy_input = tf.random.uniform((1, 256, 256, 3))
model(dummy_input)  # ensures all layers are initialized

# 5️⃣ Load weights
MODEL_PATH = "MobileVit-XXS-ASL-Augmented-TRUE-Mendeley.keras"
model.load_weights(MODEL_PATH)


# model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
num_classes = model.output_shape[-1]

print("✅ Model loaded. Num classes:", num_classes)

# ASL class labels (make sure order matches training)
# CLASS_NAMES = ['A', 'B', 'C', 'D', 'E','F', 'G', 'H', 'I', 'K', 'L', 'M','N','O','P','Q','R','S','T','U','V','W','X','Y']
CLASS_NAMES = list(string.ascii_uppercase)

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_frame(frame, target_size=(256, 256)):
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
        img = preprocess_frame(hand_region, target_size=(256, 256))

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
