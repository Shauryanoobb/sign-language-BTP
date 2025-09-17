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

# Preprocessing
def preprocess_image(image_path, target_size=(128, 128)):
    img = cv2.imread(image_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32") #/ 255.0
    img = np.expand_dims(img, axis=0)  # (1,128,128,3)
    return img

# Test prediction
image_path = "Ytest.png"  # replace with one test image
img = preprocess_image(image_path)
pred = model.predict(img, verbose=0)

pred_class = np.argmax(pred, axis=1)[0]
print("Predicted class:", CLASS_NAMES[pred_class])
print("Raw prediction scores:", pred)
