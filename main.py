import cv2
import numpy as np

# Try to load TFLite runtime (for Raspberry Pi)
try:
    import tflite_runtime.interpreter as tflite
    backend = "tflite-runtime"
except ImportError:
    # Fallback to TensorFlow Lite (for laptops/desktops with full TF)
    import tensorflow.lite as tflite
    backend = "tensorflow.lite"

print(f"✅ Using backend: {backend}")

# Path to TFLite model
MODEL_PATH = "asl_model.tflite"

# Load model
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

# Get input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ASL class labels
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z',
]

def preprocess_image(image_path, target_size=(128,128)):
    img = cv2.imread(image_path)  # BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Test prediction
image_path = "A_test.jpg"
img = preprocess_image(image_path)

# Run inference
interpreter.set_tensor(input_details[0]['index'], img)
interpreter.invoke()
output_data = interpreter.get_tensor(output_details[0]['index'])

pred_class = np.argmax(output_data, axis=1)[0]
print("Predicted class:", CLASS_NAMES[pred_class])
print("Raw prediction scores:", output_data)
