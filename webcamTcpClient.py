import cv2
import numpy as np
import socket
import struct
import tflite_runtime.interpreter as tflite

# -------------------------
# Model setup
# -------------------------
MODEL_PATH = "models/mobilenetv2_mendeley_24signs_augmented.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (H, W)

CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# -------------------------
# Sockets
# -------------------------
VIDEO_PORT = 6000
PRED_PORT  = 5005

sock_video = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_video.bind(("0.0.0.0", VIDEO_PORT))
sock_video.listen(1)

print("📹 Waiting for Mac video stream...")
conn, addr = sock_video.accept()
print("✅ Connected to Mac")

sock_pred = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    # Receive 4-byte length prefix
    raw_len = conn.recv(4)
    if not raw_len:
        break
    frame_len = struct.unpack(">L", raw_len)[0]

    # Receive full frame
    data = b""
    while len(data) < frame_len:
        packet = conn.recv(frame_len - len(data))
        if not packet:
            break
        data += packet

    # Decode frame (already ROI from Mac)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        continue

    # Preprocess
    img = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), tuple(input_shape))
    img = img.astype("float32")
    img = np.expand_dims(img, axis=0)

    # Inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)

    pred_text = f"{CLASS_NAMES[pred_class]} ({confidence:.2f})"
    print(pred_text)

    # Send prediction back
    sock_pred.sendto(pred_text.encode(), (addr[0], PRED_PORT))

conn.close()
