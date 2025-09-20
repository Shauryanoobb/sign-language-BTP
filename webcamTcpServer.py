import cv2
import socket
import struct
import threading
import time

PI_IP   = "10.17.37.192"  # change according to location
PI_PORT = 6000
MAC_PORT = 5005

sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_send.connect((PI_IP, PI_PORT))

# -------------------------
# Prediction receiver
# -------------------------
prediction_text = "Waiting..."
def listen_predictions():
    global prediction_text
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", MAC_PORT))
    while True:
        data, _ = sock_recv.recvfrom(1024)
        prediction_text = data.decode()
threading.Thread(target=listen_predictions, daemon=True).start()

# -------------------------
# Bounding box
# -------------------------
# ROI = (150, 50, 300, 300)  # must match Pi side


FRAME_SKIP = 5  #this number was found optimal for latency

cap = cv2.VideoCapture(0)
frame_count = 0
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
roi_size = int(0.28 * frame_w)   # enough for hand
x = (frame_w - roi_size) // 2   # left aligned to center
y = (frame_h - roi_size) // 2   # top aligned to center
ROI = (x, y, roi_size, roi_size)
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    frame_count += 1

    # Draw bounding box on local preview
    x, y, w, h = ROI
    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Frame skipping
    if frame_count % FRAME_SKIP == 0:
        roi = frame[y:y+h, x:x+w]
        _, encoded = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 50]) #enough quality for our model
        data = encoded.tobytes()

        # Send with length prefix
        sock_send.sendall(struct.pack(">L", len(data)) + data)

    # Show local feed with prediction overlay
    cv2.putText(frame, prediction_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Webcam (Mac)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock_send.close()
cv2.destroyAllWindows()
