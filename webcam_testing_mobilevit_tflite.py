import cv2
import numpy as np
import tensorflow as tf
import string

# -----------------------------
# Load TFLite model
# -----------------------------
TFLITE_MODEL_PATH = "mobilevit_asl_dynamic.tflite"  # change filename as needed

interpreter = tf.lite.Interpreter(model_path=TFLITE_MODEL_PATH)
interpreter.allocate_tensors()

# Get input & output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded")

# ASL class labels (A-Z)
CLASS_NAMES = list(string.ascii_uppercase)

# -----------------------------
# Preprocessing function
# -----------------------------
def preprocess_frame(frame, target_size=(256, 256)):  # use same input size as MobileViT
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
    img = cv2.resize(img, target_size)            
    img = img.astype("float32") #/ 255.0           # normalize if trained that way
    img = np.expand_dims(img, axis=0)             
    return img

# -----------------------------
# Webcam live testing
# -----------------------------
cap = cv2.VideoCapture(0)

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

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    hand_region = frame[y1:y2, x1:x2]

    if hand_region.shape[0] > 0 and hand_region.shape[1] > 0:
        img = preprocess_frame(hand_region, target_size=(256, 256))

        # Run inference
        interpreter.set_tensor(input_details[0]['index'], img)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]['index'])

        pred_class = np.argmax(pred, axis=1)[0]
        confidence = np.max(pred)

        text = f"{CLASS_NAMES[pred_class]} ({confidence:.2f})"
        cv2.putText(frame, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Live Testing - TFLite", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
