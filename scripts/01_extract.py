import cv2
import mediapipe as mp
import mediapipe.python.solutions.pose as mp_pose  # Fixed import
import numpy as np
import os

# --- CONFIG ---
VIDEO_NAME = 'good_posture.mp4' # Run for both videos! (good_posture.mp4 and bad_posture.mp4)
SAVE_DIR = r'C:\Users\USERAS\Desktop\Sitting_posture_research\data\custom'
VIDEO_PATH = os.path.join(r'C:\Users\USERAS\Desktop\Sitting_posture_research\videos', VIDEO_NAME)

if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

# Initialize pose tracking from explicit solution path
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, min_detection_confidence=0.5)

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
        ear = [lm[7].x, lm[7].y]
        sh = [lm[11].x, lm[11].y]
        hip = [lm[23].x, lm[23].y]
        
        # --- 5 FEATURE GEOMETRY ---
        f1 = calculate_angle([ear[0], 0], sh, ear)
        f2 = calculate_angle(ear, sh, hip) 
        f3 = calculate_angle([sh[0], 0], hip, sh)
        torso_len = np.linalg.norm(np.array(sh) - np.array(hip)) + 1e-6
        f4 = (ear[0] - sh[0]) / torso_len
        f5 = (sh[1] - ear[1]) / torso_len

        data_buffer.append([f1, f2, f3, f4, f5])

cap.release()
if len(data_buffer) > 0:
    out_path = os.path.join(SAVE_DIR, VIDEO_NAME.replace('.mp4', '_data.npy'))
    np.save(out_path, np.array(data_buffer))
    print(f"Saved {len(data_buffer)} frames to {out_path}")