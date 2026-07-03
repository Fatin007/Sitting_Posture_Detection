import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import os

# Clean, standard import path for MediaPipe solutions
import mediapipe as mp
mp_pose = mp.solutions.pose

# Import loader directly from standard standalone Keras
from keras.models import load_model

# Path configurations
BASE_DIR = r'C:\Users\USERAS\Desktop\Sitting_posture_research'
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'custom_posture_model.h5')
VIDEO_PATH = os.path.join(BASE_DIR, 'videos', 'test_posture_sample_4.mp4')

# Load Model and Initialize MediaPipe Pose Tracker
model = load_model(MODEL_PATH)
pose = mp_pose.Pose(min_detection_confidence=0.7)

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    return np.abs(radians*180.0/np.pi)

cap = cv2.VideoCapture(VIDEO_PATH)
data_buffer = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    if results.pose_landmarks:
        lm = results.pose_landmarks.landmark
        ear, sh, hip = [lm[7].x, lm[7].y], [lm[11].x, lm[11].y], [lm[23].x, lm[23].y]
        
        # Calculate Features
        f1 = calculate_angle([ear[0],0],sh,ear)
        f2 = calculate_angle(ear, sh, hip) # Spine curve angle
        f3 = calculate_angle([sh[0],0],hip,sh)
        torso_len = np.linalg.norm(np.array(sh) - np.array(hip)) + 1e-6
        f4 = (ear[0] - sh[0]) / torso_len
        f5 = (sh[1] - ear[1]) / torso_len
        
        data_buffer.append([f1, f2, f3, f4, f5])

        if len(data_buffer) >= 60:
            pred = model.predict(np.expand_dims(data_buffer[-60:], axis=0), verbose=0)
            label = "GOOD" if np.argmax(pred) == 0 else "BAD"
            color = (0, 255, 0) if label == "GOOD" else (0, 0, 255)
            
            # UI Overlay bounding text box
            cv2.rectangle(frame, (10, 10), (450, 110), (0,0,0), -1)
            cv2.putText(frame, f"STATUS: {label}", (30, 50), 1, 2, color, 3)
            cv2.putText(frame, f"Curve Angle: {f2:.1f} deg", (30, 90), 1, 1.2, (255, 255, 255), 2)
            data_buffer.pop(0)

    cv2.imshow('Thesis: Spinal Curve Recognition', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()