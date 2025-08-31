import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

# Load model
model = tf.keras.models.load_model("tinyvit_student_final_synthetic.keras")
model.trainable = False
CLASS_NAMES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

def preprocess(img, target_size=(128,128)):
    img = cv2.resize(img, target_size)
    img = img.astype("float32")/255.0
    return np.expand_dims(img, axis=0)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Flip for mirror effect
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    
    # Mediapipe processing
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            xmin, xmax = int(min(x_coords)*w), int(max(x_coords)*w)
            ymin, ymax = int(min(y_coords)*h), int(max(y_coords)*h)
            
            # Crop hand region
            hand_img = frame[ymin:ymax, xmin:xmax]
            if hand_img.size > 0:
                inp = preprocess(hand_img)
                pred = model.predict(inp, verbose=0)
                pred_class = np.argmax(pred)
                label = CLASS_NAMES[pred_class]

                # Draw bounding box + label
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)
                cv2.putText(frame, label, (xmin, ymin-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            1, (0,255,0), 2, cv2.LINE_AA)

            # Draw hand landmarks (optional)
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
