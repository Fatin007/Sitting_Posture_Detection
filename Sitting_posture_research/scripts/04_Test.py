import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import os
from keras.models import load_model

# --- VIDEO FILE CONFIG ---
VIDEO_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'videos', 'test_posture_sample_2.mp4')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'custom_posture_model.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing trained model file at: {MODEL_PATH}")

print("Loading pre-trained neural network...")
model = load_model(MODEL_PATH)

# MediaPipe PoseLandmarker (new Tasks API)
base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.abs(radians*180.0/np.pi)

print(f"Opening video file: {VIDEO_PATH}")
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    print(f"❌ Failed to open video: {VIDEO_PATH}")
    exit()

data_buffer = []

print("Initializing MediaPipe tracking on video. Press 'q' to close.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Video ended.")
        break

    # --- POSTURE ANALYSIS INTERFACE ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    detection_result = detector.detect(mp_image)

    if detection_result.pose_landmarks:
        lm = detection_result.pose_landmarks[0]  # first person
        ear, sh, hip = [lm[7].x, lm[7].y], [lm[11].x, lm[11].y], [lm[23].x, lm[23].y]
        
        f1 = calculate_angle([ear[0], 0], sh, ear)
        f2 = calculate_angle(ear, sh, hip)  
        f3 = calculate_angle([sh[0], 0], hip, sh)
        torso_len = np.linalg.norm(np.array(sh) - np.array(hip)) + 1e-6
        f4 = (ear[0] - sh[0]) / torso_len
        f5 = (sh[1] - ear[1]) / torso_len
        
        data_buffer.append([f1, f2, f3, f4, f5])

        if len(data_buffer) >= 60:
            pred = model.predict(np.expand_dims(data_buffer[-60:], axis=0), verbose=0)
            label = "GOOD" if np.argmax(pred) == 0 else "BAD"
            color = (0, 255, 0) if label == "GOOD" else (0, 0, 255)
            
            cv2.rectangle(frame, (10, 10), (450, 110), (0, 0, 0), -1)
            cv2.putText(frame, f"STATUS: {label}", (30, 50), 1, 2, color, 3)
            cv2.putText(frame, f"Curve Angle: {f2:.1f} deg", (30, 90), 1, 1.2, (255, 255, 255), 2)
            data_buffer.pop(0)

    cv2.imshow('Thesis: Spinal Curve Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()