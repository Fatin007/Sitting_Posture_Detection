import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import tensorflow as tf
import os
import time
import requests
from keras.models import load_model

# --- LIVE STREAM CONFIG ---
STREAM_URL = "http://192.168.65.216/stream"

# --- BLYNK IOT CONFIG ---
BLYNK_AUTH_TOKEN = "mAunY5kUSboS3lMjyWtbrYic56W-rk-w"
BLYNK_URL = "https://blynk.cloud/external/api/"
BAD_POSTURE_ALERT_SECONDS = 20  # seconds of continuous bad posture before alert

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'custom_posture_model.h5')

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Missing trained model file at: {MODEL_PATH}")

print("Loading pre-trained neural network...")
model = load_model(MODEL_PATH)

# MediaPipe PoseLandmarker (new Tasks API)
TASK_PATH = os.path.join(BASE_DIR, 'pose_landmarker_lite.task')
base_options = python.BaseOptions(model_asset_path=TASK_PATH)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    min_pose_detection_confidence=0.7
)
detector = vision.PoseLandmarker.create_from_options(options)


def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    return np.abs(radians * 180.0 / np.pi)


def send_blynk_alert():
    """Send push notification and update virtual pin via Blynk IoT HTTP API."""
    try:
        # Push notification via logEvent
        event_url = f"{BLYNK_URL}logEvent?token={BLYNK_AUTH_TOKEN}&code=bad_posture"
        resp = requests.get(event_url, timeout=5)
        print(f"[Blynk] logEvent response: {resp.status_code} - {resp.text}")

        # Update Virtual Pin V0 with status
        update_url = f"{BLYNK_URL}update?token={BLYNK_AUTH_TOKEN}&V0=BAD_POSTURE_DETECTED"
        resp2 = requests.get(update_url, timeout=5)
        print(f"[Blynk] V0 update response: {resp2.status_code}")

        return True
    except Exception as e:
        print(f"[Blynk] ERROR sending alert: {e}")
        return False


# --- BUZZER QUEUE (for main loop to process) ---
buzzer_pending = False              # Set to True when buzzer needs to be triggered


def trigger_buzzer():
    """Signal the main loop to trigger the buzzer on the next iteration.
    The actual buzzer logic runs in the main loop to avoid threading issues
    with the VideoCapture object."""
    global buzzer_pending
    buzzer_pending = True
    print("[Buzzer] Queued for next main loop iteration.")


# --- BLYNK BUZZER BUTTON ---
BLYNK_BUZZER_PIN = "V1"              # Virtual pin for buzzer button in Blynk app
blynk_button_was_pressed = False      # Debounce: track previous button state


def check_blynk_buzzer_button():
    """Poll Blynk virtual pin V1. If button is pressed (value=1), trigger buzzer.
    Uses debounce: button must be released (back to 0) before next trigger."""
    global blynk_button_was_pressed
    try:
        get_url = f"{BLYNK_URL}get?token={BLYNK_AUTH_TOKEN}&{BLYNK_BUZZER_PIN}"
        resp = requests.get(get_url, timeout=5)
        if resp.status_code == 200:
            value = resp.text.strip()
            if value == "1" and not blynk_button_was_pressed:
                blynk_button_was_pressed = True
                print(f"[Blynk] Buzzer button pressed on {BLYNK_BUZZER_PIN}! Triggering buzzer...")
                trigger_buzzer()
                # Reset pin back to 0 so the button widget returns to off state
                reset_url = f"{BLYNK_URL}update?token={BLYNK_AUTH_TOKEN}&{BLYNK_BUZZER_PIN}=0"
                requests.get(reset_url, timeout=5)
            elif value == "0":
                blynk_button_was_pressed = False
    except KeyboardInterrupt:
        raise  # Let Ctrl+C pass through
    except Exception:
        pass  # Silently ignore network errors during polling



print(f"Connecting to live stream: {STREAM_URL}")
cap = cv2.VideoCapture(STREAM_URL)
if not cap.isOpened():
    print(f"❌ Failed to open stream: {STREAM_URL}")
    exit()

data_buffer = []

# --- PREDICTION SMOOTHING ---
CONFIDENCE_THRESHOLD = 0.75       # only trust predictions above this confidence
SMOOTH_WINDOW = 15                 # moving average over last N predictions
pred_history = []                   # stores (label, confidence) tuples
smoothed_label = "WAIT"            # current smoothed label

# --- BAD POSTURE TIMER ---
bad_posture_start_time = None      # timestamp when bad posture first detected
alert_sent = False                 # prevents duplicate alerts per session
last_alert_time = 0                # cooldown timestamp to prevent rapid re-triggering
ALERT_COOLDOWN_SECONDS = 60        # minimum seconds between alerts

# --- BLYNK BUTTON POLLING ---
last_blynk_poll_time = 0           # timestamp of last Blynk button poll
BLYNK_POLL_INTERVAL = 2.0          # poll Blynk every 2 seconds

# Warm up stream
for _ in range(10):
    cap.read()

print("Live posture detection running. Press 'q' to quit.")

while True:
    # --- PROCESS PENDING BUZZER (before reading next frame) ---
    if buzzer_pending:
        buzzer_pending = False
        import socket as _socket
        ip = STREAM_URL.split("//")[1].split("/")[0].split(":")[0]
        print(f"[Buzzer] Releasing stream & triggering buzzer at http://{ip}/buzzer ...")

        # 1. Release stream
        if cap.isOpened():
            cap.release()
        time.sleep(1.0)  # Wait for ESP32 to fully release the connection

        # 2. Send buzzer request via raw socket (fire-and-forget)
        try:
            sock = _socket.socket(_socket.AF_INET, _socket.SOCK_STREAM)
            sock.settimeout(2)
            sock.connect((ip, 80))
            req = f"GET /buzzer HTTP/1.1\r\nHost: {ip}\r\nConnection: close\r\n\r\n"
            sock.sendall(req.encode())
            sock.close()
            print("[Buzzer] HTTP request sent, socket closed.")
        except Exception as e:
            print(f"[Buzzer] Socket error: {e}")

        # 3. Wait for buzzer to finish (3s) then reconnect stream
        time.sleep(3.5)
        cap = cv2.VideoCapture(STREAM_URL)
        if cap.isOpened():
            print("[Buzzer] Stream reconnected.")
            for _ in range(10):
                cap.read()
        else:
            print("[Buzzer] ERROR: Failed to reconnect stream!")
        continue  # Skip this iteration, go read fresh frame

    ret, frame = cap.read()
    if not ret:
        print("⚠️  Frame dropped, reconnecting...")
        cap.release()
        cap = cv2.VideoCapture(STREAM_URL)
        continue

    # Resize frame to target display size
    frame = cv2.resize(frame, (600, 450))

    # --- POLL BLYNK BUZZER BUTTON (every 2 seconds) ---
    if time.time() - last_blynk_poll_time >= BLYNK_POLL_INTERVAL:
        check_blynk_buzzer_button()
        last_blynk_poll_time = time.time()

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
            raw_label = "GOOD" if np.argmax(pred) == 0 else "BAD"
            confidence = float(np.max(pred))

            # Only accept prediction if confidence is high enough
            if confidence >= CONFIDENCE_THRESHOLD:
                pred_history.append((raw_label, confidence))
                if len(pred_history) > SMOOTH_WINDOW:
                    pred_history.pop(0)

                # Majority vote over smoothed window
                good_count = sum(1 for l, _ in pred_history if l == "GOOD")
                bad_count = len(pred_history) - good_count
                smoothed_label = "GOOD" if good_count >= bad_count else "BAD"

            color = (0, 255, 0) if smoothed_label == "GOOD" else (0, 0, 255)

            # --- BAD POSTURE TIMER LOGIC ---
            if smoothed_label == "BAD":
                if bad_posture_start_time is None:
                    bad_posture_start_time = time.time()
                    alert_sent = False

                elapsed = time.time() - bad_posture_start_time
                remaining = max(0, BAD_POSTURE_ALERT_SECONDS - int(elapsed))
                cooldown_remaining = max(0, ALERT_COOLDOWN_SECONDS - int(time.time() - last_alert_time))

                if elapsed >= BAD_POSTURE_ALERT_SECONDS and not alert_sent and cooldown_remaining == 0:
                    print(f"[ALERT] {BAD_POSTURE_ALERT_SECONDS}s of bad posture detected! Sending Blynk notification...")
                    success = send_blynk_alert()
                    alert_sent = True
                    last_alert_time = time.time()
                    if success:
                        print("[ALERT] Blynk notification sent successfully!")
                    else:
                        print("[ALERT] Failed to send Blynk notification.")
                    # Trigger buzzer on ESP32 (fire-and-forget, don't block on result)
                    trigger_buzzer()
            else:
                # Posture is GOOD — reset timer
                bad_posture_start_time = None
                alert_sent = False

            # --- FRAME OVERLAY (compact, top-left corner) ---
            overlay_w, overlay_h = 280, 110
            cv2.rectangle(frame, (8, 8), (8 + overlay_w, 8 + overlay_h), (0, 0, 0), -1)
            cv2.rectangle(frame, (8, 8), (8 + overlay_w, 8 + overlay_h), (80, 80, 80), 1)
            cv2.putText(frame, f"{smoothed_label}", (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)
            cv2.putText(frame, f"conf: {confidence:.0%}", (18, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame, f"angle: {f2:.0f} deg", (18, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # Show bad posture timer / alert status
            if smoothed_label == "BAD":
                if alert_sent and cooldown_remaining > 0:
                    cv2.putText(frame, f"cooldown: {cooldown_remaining}s", (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1)
                elif alert_sent:
                    cv2.putText(frame, "ALERT SENT", (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                else:
                    cv2.putText(frame, f"alert in: {remaining}s", (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)
            data_buffer.pop(0)
    else:
        # Show "no person detected" when no landmarks found
        # Also reset timer when no person is visible
        bad_posture_start_time = None
        alert_sent = False

        overlay_w, overlay_h = 240, 35
        cv2.rectangle(frame, (8, 8), (8 + overlay_w, 8 + overlay_h), (0, 0, 0), -1)
        cv2.rectangle(frame, (8, 8), (8 + overlay_w, 8 + overlay_h), (80, 80, 80), 1)
        cv2.putText(frame, "No person", (18, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 2)

    cv2.imshow('Live Posture Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
