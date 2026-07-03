# Sitting Posture Detection - Command Reference

## Prerequisites
- Python 3.11+ with virtual environment
- PlatformIO CLI (for ESP32-CAM firmware)
- ESP32-CAM (AI-Thinker) board connected via USB

---

## 1. Python Virtual Environment Setup

```bash
# Create virtual environment (from project root)
cd Sitting_Posture_Detection
python -m venv .venv

# Activate virtual environment (Windows)
.venv\Scripts\activate

# Activate virtual environment (macOS/Linux)
source .venv/bin/activate

# Install Python dependencies
pip install opencv-python mediapipe tensorflow keras numpy requests
```

---

## 2. ESP32-CAM Firmware (PlatformIO)

### Build & Upload
```bash
# From project root: Sitting_Posture_Detection/

# Build the firmware
pio run

# Upload to ESP32-CAM (via USB)
pio run --target upload

# Monitor serial output (baud 115200)
pio device monitor --baud 115200

# Build + Upload + Monitor (all-in-one)
pio run --target upload && pio device monitor --baud 115200
```

### After Upload
- Note the IP address printed in serial monitor (e.g., `192.168.x.x`)
- Open `http://<IP>` in a browser to view the camera stream
- The stream endpoint is `http://<IP>/stream`
- The buzzer endpoint is `http://<IP>/buzzer`

---

## 3. Model Training Pipeline

All scripts are in `Sitting_posture_research/scripts/`. Run from the `scripts/` directory.

### Step 1: Extract Features from Videos
```bash
cd Sitting_posture_research/scripts

# Extract good posture features (edit VIDEO_NAME in 01_extract.py to 'good_posture.mp4')
python 01_extract.py

# Extract bad posture features (edit VIDEO_NAME in 01_extract.py to 'bad_posture.mp4')
python 01_extract.py
```
> **Note:** Edit `VIDEO_NAME` and paths inside `01_extract.py` before running.
> Output: `data/custom/good_posture_data.npy` and `data/custom/bad_posture_data.npy`

### Step 2: Slice Data into Windows
```bash
python 02_slice.py
```
> Output: `data/X_final.npy` and `data/y_final.npy`

### Step 3: Train the Model
```bash
python 03_train.py
```
> Output: `models/custom_posture_model.h5`

### Step 4: Test the Model (on a video file)
```bash
python 04_Test.py
```
> Tests the trained model on `videos/test_posture_sample_2.mp4`

---

## 4. Live Posture Detection (Real-time Stream)

```bash
cd Sitting_posture_research/scripts

# Make sure STREAM_URL in 05_live_stream.py matches your ESP32-CAM IP
python 05_live_stream.py
```
> **Controls:** Press `q` to quit the live stream window.
> **Requirements:** ESP32-CAM must be powered on and connected to WiFi.

---

## 5. Quick Reference - File Structure

```
Sitting_Posture_Detection/
├── platformio.ini              # PlatformIO project config (board: esp32cam)
├── src/main.cpp                # ESP32-CAM firmware (camera stream + buzzer)
├── Sitting_posture_research/
│   ├── pose_landmarker_lite.task   # MediaPipe pose model
│   ├── data/
│   │   ├── X_final.npy             # Training features (60-frame windows)
│   │   ├── y_final.npy             # Training labels
│   │   └── custom/
│   │       ├── good_posture_data.npy
│   │       └── bad_posture_data.npy
│   ├── models/
│   │   └── custom_posture_model.h5  # Trained LSTM model
│   ├── scripts/
│   │   ├── 01_extract.py            # Extract pose features from video
│   │   ├── 02_slice.py              # Slice features into 60-frame windows
│   │   ├── 03_train.py              # Train LSTM neural network
│   │   ├── 04_Test.py               # Test model on a video file
│   │   └── 05_live_stream.py        # Real-time posture detection from ESP32-CAM
│   └── videos/                      # Input videos for training/testing
```

---

## 6. Troubleshooting

| Issue | Fix |
|-------|-----|
| Camera init failed | Check USB power (ESP32-CAM needs 5V/2A); re-seat camera ribbon cable |
| Stream laggy | Lower `jpeg_quality` in `main.cpp` (try 8-10) |
| WiFi not connecting | Verify SSID/password in `main.cpp`; check 2.4GHz network |
| Import errors (Python) | Activate `.venv` and re-run `pip install` |
| MediaPipe errors | Ensure `pose_landmarker_lite.task` exists in `Sitting_posture_research/` |
| Blynk 400 error | Create `bad_posture` event in Blynk IoT device template |
