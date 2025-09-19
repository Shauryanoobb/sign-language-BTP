import cv2
import socket
import struct
import threading

PI_IP   = "10.151.32.192"   # Pi IP
PI_PORT = 6000              # video frames TCP
MAC_PORT = 5005             # receive predictions UDP (can keep this UDP)

prediction_text = "Waiting..."

def listen_predictions():
    global prediction_text
    sock_recv = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_recv.bind(("0.0.0.0", MAC_PORT))
    while True:
        data, _ = sock_recv.recvfrom(1024)
        prediction_text = data.decode()

threading.Thread(target=listen_predictions, daemon=True).start()

# TCP socket for video
sock_send = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock_send.connect((PI_IP, PI_PORT))

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    # Encode full-res frame
    _, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
    data = encoded.tobytes()

    # First send length of frame, then data
    sock_send.sendall(struct.pack(">L", len(data)) + data)

    # Show webcam locally with predictions
    cv2.putText(frame, prediction_text, (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("ASL Webcam", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
sock_send.close()
cv2.destroyAllWindows()
