import cv2
import numpy as np
import tflite_runtime.interpreter as tflite

# -------------------------
# Load TFLite model
# -------------------------
MODEL_PATH = "mobilenetv2_signlang_synthetic_14_signs.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape'][1:3]  # (height, width)

# Define your labels (update if needed)
CLASS_NAMES = [
    'A','B','C','D','F','G','H','I','J','K',
    'L','O','R','X'
]

# -------------------------
# Preprocess function
# -------------------------
def preprocess_frame(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, tuple(input_shape))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)  # (1, h, w, 3)
    return img

# -------------------------
# Capture stream from Mac
# -------------------------
cap = cv2.VideoCapture("udp://0.0.0.0:1234")

if not cap.isOpened():
    print("❌ Could not open UDP stream.")
    exit()

print("✅ Listening for video stream...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ No frame received")
        continue

    # Preprocess frame
    img = preprocess_frame(frame)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])

    # Prediction
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    print(f"Prediction: {CLASS_NAMES[pred_class]} ({confidence:.2f})")

    # Optional: show video feed
    cv2.imshow("Pi Stream", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()