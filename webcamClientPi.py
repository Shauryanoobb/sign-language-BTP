import cv2
import numpy as np
import socket
import tflite_runtime.interpreter as tflite
import time

# -------------------------
# Model setup
# -------------------------
MODEL_PATH = "mobilenetv2_mendeley_26signs_augmented.tflite"
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (height, width)
CLASS_NAMES = [
    'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',
    'P','Q','R','S','T','U','V','W','X','Y','Z',
]

# -------------------------
# UDP sockets
# -------------------------
VIDEO_PORT = 6000
PRED_PORT  = 5005

sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.bind(("0.0.0.0", VIDEO_PORT))

sock_pred = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# -------------------------
# Frame assembly variables
# -------------------------
buffers = {}          # frame_id -> {chunk_id: bytes}
frame_times = {}      # frame_id -> first arrival time
MAX_FRAME_AGE = 1.0   # seconds

print("📹 Waiting for Mac video stream...")

while True:
    try:
        packet, addr = sock_video.recvfrom(65535)
        if len(packet) < 6:
            continue  # ignore malformed packets

        # Extract frame_id and chunk_id
        frame_id = int.from_bytes(packet[:4], 'big')
        chunk_id = int.from_bytes(packet[4:6], 'big')
        payload = packet[6:]

        if frame_id not in buffers:
            buffers[frame_id] = {}
            frame_times[frame_id] = time.time()

        if chunk_id == 65535:
            # End marker
            if buffers[frame_id]:
                try:
                    data = b"".join(buffers[frame_id][i] for i in sorted(buffers[frame_id]))
                except KeyError:
                    # Missing chunk, skip frame
                    buffers.pop(frame_id, None)
                    frame_times.pop(frame_id, None)
                    continue

                buffers.pop(frame_id, None)
                frame_times.pop(frame_id, None)

                # Decode JPEG
                frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    continue

                # Preprocess for model
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

                # Send prediction back to Mac
                sock_pred.sendto(pred_text.encode(), (addr[0], PRED_PORT))
        else:
            buffers[frame_id][chunk_id] = payload

        # Cleanup old frames
        now = time.time()
        for fid in list(frame_times.keys()):
            if now - frame_times[fid] > MAX_FRAME_AGE:
                buffers.pop(fid, None)
                frame_times.pop(fid, None)

    except Exception as e:
        print("⚠️ Error:", e)
