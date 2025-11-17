import socket
import cv2
import threading
import struct

# -------------------------
# Configuration
# -------------------------
PI_IP = "10.133.118.191"
VIDEO_PORT = 5005
MAC_PRED_PORT = 6005

FRAME_SKIP = 4          # Send every Nth frame for optimal latency
JPEG_QUALITY = 50       # Balance quality and bandwidth
MAX_PACKET_SIZE = 1472  # MTU-safe UDP payload

ROI_SCALE = 0.28        # ROI size as fraction of frame width

# -------------------------
# Global State
# -------------------------
prediction_text = "Waiting..."
prediction_lock = threading.Lock()

# -------------------------
# UDP Prediction Receiver
# -------------------------
def listen_predictions():
    global prediction_text
    sock_pred = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock_pred.bind(("0.0.0.0", MAC_PRED_PORT))
    print(f"👂 Listening for predictions on UDP:{MAC_PRED_PORT}")

    while True:
        try:
            data, _ = sock_pred.recvfrom(1024)
            with prediction_lock:
                prediction_text = data.decode()
        except Exception as e:
            print(f"⚠️ Prediction error: {e}")

threading.Thread(target=listen_predictions, daemon=True).start()

# -------------------------
# UDP Video Sender
# -------------------------
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("❌ Could not open webcam")

frame_count = 0
frame_id = 0

# ROI setup
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
roi_size = int(ROI_SCALE * frame_w)
x = (frame_w - roi_size) // 2
y = (frame_h - roi_size) // 2
ROI = (x, y, roi_size, roi_size)

print(f"🎥 Webcam: {frame_w}x{frame_h}, ROI: {roi_size}x{roi_size}")
print(f"📤 Sending every {FRAME_SKIP} frames, JPEG quality: {JPEG_QUALITY}")
print("Press 'q' to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Draw ROI bounding box
        x, y, w, h = ROI
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Frame skipping - send ROI to Pi
        if frame_count % FRAME_SKIP == 0:
            roi = frame[y:y+h, x:x+w]
            _, jpeg = cv2.imencode(".jpg", roi, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            frame_bytes = jpeg.tobytes()
            
            # Calculate chunks
            data_per_chunk = MAX_PACKET_SIZE - 8
            total_size = len(frame_bytes)
            total_chunks = (total_size + data_per_chunk - 1) // data_per_chunk
            
            # Send chunks
            for chunk_num in range(total_chunks):
                start = chunk_num * data_per_chunk
                end = min(start + data_per_chunk, total_size)
                chunk_data = frame_bytes[start:end]
                
                # Pack header: frame_id (4) + chunk_num (2) + total_chunks (2)
                header = struct.pack(">IHH", frame_id, chunk_num, total_chunks)
                packet = header + chunk_data
                
                sock_video.sendto(packet, (PI_IP, VIDEO_PORT))
            
            frame_id = (frame_id + 1) % 10000
        
        # -------------------------
        # Display Prediction
        # -------------------------
        with prediction_lock:
            text = prediction_text
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2
        thickness = 3
        
        # Center text above ROI
        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = x + (w - text_w) // 2
        text_y = y - 10
        
        # Background rectangle for readability
        cv2.rectangle(frame, 
                     (text_x - 5, text_y - text_h - 5),
                     (text_x + text_w + 5, text_y + 5),
                     (0, 0, 0), -1)
        
        # Draw prediction text
        cv2.putText(frame, text, (text_x, text_y),
                   font, font_scale, (0, 255, 0), thickness)
        
        # Frame counter
        cv2.putText(frame, f"Frame: {frame_count}", (10, frame_h - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show local feed
        cv2.imshow("Static ASL Webcam (Mac)", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n🛑 Interrupted by user")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    cap.release()
    sock_video.close()
    cv2.destroyAllWindows()
    print("🟢 Client closed")