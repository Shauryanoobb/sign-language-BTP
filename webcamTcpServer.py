import cv2
import socket
import struct
import threading
import time

PI_IP   = "10.72.225.192"  # change according to location
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
FRAME_SKIP = 5  # this number was found optimal for latency

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
        _, encoded = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, 50])  # enough quality for our model
        data = encoded.tobytes()

        # Send with length prefix
        sock_send.sendall(struct.pack(">L", len(data)) + data)

    # -------------------------
    # Show prediction above bounding box
    # -------------------------
    text = prediction_text
    font_scale = 1.2   # larger font
    thickness = 3
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size to center it above the ROI
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    text_x = x + (w - text_w) // 2
    text_y = y - 10  # slightly above bounding box

    # Background rectangle for readability
    cv2.rectangle(frame, (text_x, text_y - text_h - 5),
                  (text_x + text_w, text_y + 5), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (text_x, text_y),
                font, font_scale, (0, 255, 0), thickness)

    # Show local feed
    cv2.imshow("ASL Webcam (Mac)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock_send.close()
cv2.destroyAllWindows()
