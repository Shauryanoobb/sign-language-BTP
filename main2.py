import tensorflow as tf
import cv2
import numpy as np
import os
from keras_vision.MobileViT_v1 import build_MobileViT_v1
from tensorflow.keras import layers, Model

# Path to your trained model
MODEL_PATH = "mobilevit_signlang_synthetic.keras"
TEST_FOLDER = "live_testing"

# Load model
# # 1️⃣ Rebuild the architecture exactly as before
# backbone = build_MobileViT_v1(model_type="XS", pretrained=False, include_top=False, num_classes=0)

# x = backbone.output
# x = layers.GlobalAveragePooling2D()(x)
# x = layers.Dense(26, activation='softmax')(x)  # 26 ASL classes

# model = Model(inputs=backbone.input, outputs=x)

# # 2️⃣ Build the model by passing a dummy input
# dummy_input = tf.random.uniform((1, 256, 256, 3))
# model(dummy_input)  # ensures all layers have weights

# # 3️⃣ Load weights from your trained file
# model.load_weights("MobileVit-XS-ASL.keras")
model = tf.keras.models.load_model(MODEL_PATH)
model.trainable = False
num_classes = model.output_shape[-1]
print("✅ Model loaded. Num classes:", num_classes)

# ASL class labels (29 total)
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z',
]

# Preprocessing
def preprocess_image(image_path, target_size=(128, 128)):
# def preprocess_image(image_path, target_size=(256, 256)):
    img = cv2.imread(image_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32")#/ 255.0
    img = np.expand_dims(img, axis=0)  # (1,128,128,3)
    return img

# Run prediction on all images in folder
for fname in os.listdir(TEST_FOLDER):
    if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        image_path = os.path.join(TEST_FOLDER, fname)
        img = preprocess_image(image_path)
        pred = model.predict(img, verbose=0)
        pred_class = np.argmax(pred, axis=1)[0]
        print(f"Image: {fname}")
        print("Predicted class:", CLASS_NAMES[pred_class])
        print("Raw prediction scores:", pred)
        print("-" * 40)