import cv2
import socket
import threading
import time
import numpy as np

PI_IP   = "10.151.32.192"  # Raspberry Pi IP
PI_PORT = 6000             # video frames
MAC_PORT = 5005            # receive predictions

MAX_CHUNK = 60000          # chunk size for UDP
FPS_LIMIT = 10             # send at 10 FPS max

sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Receive predictions from Pi
prediction_text = "Waiting..."
def listen_predictions():
    global prediction_text
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", MAC_PORT))
    while True:
        data, _ = sock_recv.recvfrom(1024)
        prediction_text = data.decode()
threading.Thread(target=listen_predictions, daemon=True).start()

# Capture webcam
cap = cv2.VideoCapture(0)
last_send = 0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Limit FPS
    if time.time() - last_send < 1.0 / FPS_LIMIT:
        continue
    last_send = time.time()

    # Resize to reduce UDP payload
    frame_small = cv2.resize(frame, (320, 240))
    _, encoded = cv2.imencode(".jpg", frame_small, [cv2.IMWRITE_JPEG_QUALITY, 40])
    data = encoded.tobytes()

    # Send in chunks
    frame_id = int(time.time() * 1000) % 100000  # unique frame id
    for i in range(0, len(data), MAX_CHUNK):
        chunk = data[i:i+MAX_CHUNK]
        header = frame_id.to_bytes(4, 'big') + (i // MAX_CHUNK).to_bytes(2, 'big')
        sock_send.sendto(header + chunk, (PI_IP, PI_PORT))

    # End marker
    sock_send.sendto(frame_id.to_bytes(4,'big') + (65535).to_bytes(2,'big'), (PI_IP, PI_PORT))

    # Show locally
    cv2.putText(frame, prediction_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
