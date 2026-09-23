# Drone CrowdVision

A low-cost drone payload that watches crowded areas and estimates crowd risk in real time. The drone captures live video and a laptop on the ground runs the AI processing.

The goal is to catch early signs of dangerous crowd congestion and raise an alert before things get worse, not after.


## Files

- `main.py` - runs on the laptop (ground station). Connects to the ESP32-CAM video stream and does all the detection, tracking and risk scoring described below
- `cam_stream.ino` - firmware for the ESP32-CAM that streams live video over Wi-Fi

## Hardware used

- ESP32-CAM (video streaming)
- MAX9814 microphone module (for the audio part)
- 3.7V LiPo battery and TP4056 charging module

## What's working right now

**Live video streaming**
The ESP32-CAM streams video over Wi-Fi as MJPEG. `main.py` connects to that stream and reads it frame by frame.

**Person detection (YOLOv8)**
Each frame is passed through YOLOv8n, a deep learning object detection model. It's pretrained on the COCO dataset and can recognize 80 types of objects. The code filters its output to keep only detections classified as "person" with confidence above 0.5, and ignores everything else. This is the actual AI component of the project.

**People counting**
Every person YOLO detects in a frame is counted, giving a live headcount shown on screen.

**Person tracking**
A simple centroid-based tracker assigns each detected person an ID and follows them across frames, so the same person isn't counted as a "new" person every frame. This also makes movement tracking possible.

**Zone-based density**
The frame is split into a 3x3 grid. Each zone is counted separately and marked SAFE, WARNING or CRITICAL based on how many people are in it, and shown as a color overlay on the video.

**Movement speed**
Using the tracked positions, the code calculates how much people are moving frame to frame on average. This is meant as a rough signal for crowd agitation, e.g. people scattering or pushing versus standing still.

**Growth rate / trend**
The code keeps a short rolling history of the people count and calculates how fast it's rising over the last 15 seconds. A crowd building up quickly is more concerning than one that's steady, even at the same headcount.

**Crowd risk score**
Right now this is a simple rule-based formula, not a trained model. It combines density, movement, and growth rate into one 0-100 score with a NORMAL / ELEVATED / CRITICAL label. It's a placeholder to demonstrate the idea of fusing multiple signals into one output. It is not yet learned from real data, so the weights and thresholds are guesses that still need to be tuned or replaced.

**Live on-screen display**
FPS, people count, average movement speed, growth rate, and the risk score are all shown directly on the video feed while it runs.


## Running it

```bash
pip install ultralytics opencv-python numpy requests
python main.py
```

Update the `STREAM_URL` in `main.py` to match your ESP32-CAM's IP address before running.

**Status: work in progress.**