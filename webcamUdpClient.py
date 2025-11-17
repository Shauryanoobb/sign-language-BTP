import cv2
import numpy as np
import socket
import struct
import tensorflow as tf

# -------------------------
# Configuration
# -------------------------
VIDEO_PORT = 5005      # Pi receives video here (UDP)
PRED_PORT = 6005       # Pi sends predictions back to Mac
MODEL_PATH = "models/mobilevit_asl_dynamic.tflite"
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Chunking parameters
MAX_PACKET_SIZE = 1472  # MTU-safe chunk size

# -------------------------
# Model Setup
# -------------------------
print("📦 Loading model...")
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
input_shape = input_details[0]['shape'][1:3]  # (H, W)
print(f"✅ Model loaded - Input shape: {input_shape}")

# -------------------------
# Preprocessing
# -------------------------
def preprocess(frame):
    """Preprocess frame for static gesture model"""
    img = cv2.resize(frame, tuple(input_shape))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32")  # Keep as is if model expects [0-255], or normalize if needed
    img = np.expand_dims(img, axis=0)
    return img

# -------------------------
# Prediction
# -------------------------
def predict(frame):
    """Run inference on a single frame"""
    img = preprocess(frame)
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    pred_class = np.argmax(pred, axis=1)[0]
    confidence = np.max(pred)
    return CLASS_NAMES[pred_class], float(confidence)

# -------------------------
# UDP Sockets
# -------------------------
sock_video = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_video.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 200000)  # Increase buffer
sock_video.bind(("0.0.0.0", VIDEO_PORT))

sock_pred = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"📡 Listening for video on UDP:{VIDEO_PORT}")

# -------------------------
# Frame Reconstruction Buffer
# -------------------------
frame_chunks = {}
expected_chunks = {}

frame_count = 0
INFERENCE_EVERY = 2  # Run inference every N frames to reduce load
last_prediction = ("Waiting...", 0.0)  # Store last prediction

try:
    while True:
        data, mac_addr = sock_video.recvfrom(65535)
        
        # Parse header: frame_id (4 bytes) + chunk_num (2 bytes) + total_chunks (2 bytes) + data
        if len(data) < 8:
            continue
            
        frame_id = struct.unpack(">I", data[0:4])[0]
        chunk_num = struct.unpack(">H", data[4:6])[0]
        total_chunks = struct.unpack(">H", data[6:8])[0]
        chunk_data = data[8:]
        
        # Initialize frame buffer if new frame
        if frame_id not in frame_chunks:
            frame_chunks[frame_id] = {}
            expected_chunks[frame_id] = total_chunks
        
        # Store chunk
        frame_chunks[frame_id][chunk_num] = chunk_data
        
        # Check if all chunks received
        if len(frame_chunks[frame_id]) == expected_chunks[frame_id]:
            # Reconstruct frame
            sorted_chunks = [frame_chunks[frame_id][i] for i in range(total_chunks)]
            full_data = b"".join(sorted_chunks)
            
            # Decode frame
            np_data = np.frombuffer(full_data, np.uint8)
            frame = cv2.imdecode(np_data, cv2.IMREAD_COLOR)
            
            # Clean up frame data
            del frame_chunks[frame_id]
            del expected_chunks[frame_id]
            
            # Clean up very old frames (keep only last 3)
            if len(frame_chunks) > 3:
                oldest = min(frame_chunks.keys())
                del frame_chunks[oldest]
                if oldest in expected_chunks:
                    del expected_chunks[oldest]
            
            if frame is not None:
                frame_count += 1
                
                # Run inference every N frames to reduce load
                if frame_count % INFERENCE_EVERY == 0:
                    gesture, conf = predict(frame)
                    last_prediction = (gesture, conf)
                    print(f"{gesture} ({conf:.3f})")
                
                # Always send the last prediction (even if we didn't just compute a new one)
                msg = f"{last_prediction[0]} ({last_prediction[1]:.2f})".encode()
                sock_pred.sendto(msg, (mac_addr[0], PRED_PORT))

except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
except Exception as e:
    print(f"❌ Error: {e}")
finally:
    sock_video.close()
    sock_pred.close()
    print("🟢 Server closed")