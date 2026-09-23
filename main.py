import cv2
import numpy as np
import requests
import time
from collections import deque
from ultralytics import YOLO

STREAM_URL = "http://10.113.89.176:81/stream"

ROWS, COLS = 3, 3                  # zone grid for density mapping
TREND_WINDOW_SEC = 15              # how far back to look for trend/growth rate
RISK_HISTORY_LEN = 100             # points kept for on-screen trend graph
MAX_TRACK_DISTANCE = 60            # px, max distance to match a detection to an existing track
TRACK_TIMEOUT = 1.0                # seconds a track can go unseen before being dropped
ZONE_WARN_COUNT = 6                # people in a zone -> WARNING
ZONE_CRITICAL_COUNT = 10           # people in a zone -> CRITICAL

model = YOLO("yolov8n.pt")


class CentroidTracker:
    def __init__(self):
        self.next_id = 0
        self.tracks = {}  # id -> {"centroid": (x,y), "last_seen": t, "history": deque}

    def update(self, detections, now):
        # detections: list of (x, y) centroids for this frame
        unmatched = set(range(len(detections)))
        used_ids = set()

        # Try to match each existing track to the nearest unmatched detection
        for tid, track in self.tracks.items():
            if not unmatched:
                break
            best_idx, best_dist = None, MAX_TRACK_DISTANCE
            for idx in unmatched:
                dx = detections[idx][0] - track["centroid"][0]
                dy = detections[idx][1] - track["centroid"][1]
                dist = (dx ** 2 + dy ** 2) ** 0.5
                if dist < best_dist:
                    best_dist, best_idx = dist, idx
            if best_idx is not None:
                cx, cy = detections[best_idx]
                track["history"].append(track["centroid"])
                if len(track["history"]) > 10:
                    track["history"].popleft()
                track["centroid"] = (cx, cy)
                track["last_seen"] = now
                unmatched.discard(best_idx)
                used_ids.add(tid)

        # Remaining detections become new tracks
        for idx in unmatched:
            self.tracks[self.next_id] = {
                "centroid": detections[idx],
                "last_seen": now,
                "history": deque(maxlen=10),
            }
            self.next_id += 1

        # Drop stale tracks
        self.tracks = {
            tid: t for tid, t in self.tracks.items()
            if now - t["last_seen"] <= TRACK_TIMEOUT
        }

        return self.tracks

    def average_speed(self):
        # Average per-track displacement per frame (px) -> proxy for crowd agitation
        speeds = []
        for t in self.tracks.values():
            if len(t["history"]) >= 2:
                (x1, y1), (x2, y2) = t["history"][-2], t["history"][-1]
                speeds.append(((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
        return float(np.mean(speeds)) if speeds else 0.0


def connect_stream():
    print("Connecting to ESP32-CAM...")
    response = requests.get(STREAM_URL, stream=True, timeout=(10, 30))
    response.raise_for_status()
    print("Stream connected")
    return response


def zone_density(centroids, frame_w, frame_h):
    """Return a ROWS x COLS grid of counts and a risk label per zone."""
    grid = np.zeros((ROWS, COLS), dtype=int)
    cell_w = frame_w / COLS
    cell_h = frame_h / ROWS

    for (x, y) in centroids:
        col = min(int(x // cell_w), COLS - 1)
        row = min(int(y // cell_h), ROWS - 1)
        grid[row][col] += 1

    return grid


def zone_status(count):
    if count >= ZONE_CRITICAL_COUNT:
        return "CRITICAL", (0, 0, 255)
    if count >= ZONE_WARN_COUNT:
        return "WARNING", (0, 165, 255)
    return "SAFE", (0, 200, 0)


def draw_zone_overlay(frame, grid):
    h, w = frame.shape[:2]
    cell_w, cell_h = w / COLS, h / ROWS
    overlay = frame.copy()

    for row in range(ROWS):
        for col in range(COLS):
            count = grid[row][col]
            status, color = zone_status(count)
            x1, y1 = int(col * cell_w), int(row * cell_h)
            x2, y2 = int(x1 + cell_w), int(y1 + cell_h)
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
            cv2.putText(frame, f"{count}", (x1 + 8, y1 + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    for row in range(ROWS):
        for col in range(COLS):
            x1, y1 = int(col * cell_w), int(row * cell_h)
            x2, y2 = int(x1 + cell_w), int(y1 + cell_h)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (80, 80, 80), 1)


def compute_risk_score(person_count, avg_speed, growth_rate):
    """
    Fuse density + movement + trend into a single 0-100 risk score.
    Tune the weights/normalizers once you have real footage to calibrate against.
    """
    density_score = min(person_count / 30.0, 1.0)        # normalize: 30 people = saturated
    movement_score = min(avg_speed / 25.0, 1.0)           # normalize: 25 px/frame = saturated
    growth_score = min(max(growth_rate, 0) / 5.0, 1.0)    # normalize: +5 people/sec = saturated

    risk = (0.5 * density_score + 0.25 * movement_score + 0.25 * growth_score) * 100
    return round(risk, 1)


def risk_label(score):
    if score >= 70:
        return "CRITICAL", (0, 0, 255)
    if score >= 40:
        return "ELEVATED", (0, 165, 255)
    return "NORMAL", (0, 200, 0)

cv2.namedWindow("Drone-Crowd-Vision", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Drone-Crowd-Vision", 1280, 720)
tracker = CentroidTracker()
count_history = deque(maxlen=RISK_HISTORY_LEN)   # (timestamp, count) for trend calc
prev_frame_time = time.time()

while True:
    try:
        response = connect_stream()
        buffer = b""

        for chunk in response.iter_content(chunk_size=4096):
            if not chunk:
                continue

            buffer += chunk
            start = buffer.find(b"\xff\xd8")
            end = buffer.find(b"\xff\xd9")

            if start == -1 or end == -1 or end <= start:
                continue

            jpg = buffer[start:end + 2]
            buffer = buffer[end + 2:]

            if not jpg:
                continue

            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is None:
                continue

            now = time.time()
            fps = 1.0 / max(now - prev_frame_time, 1e-6)
            prev_frame_time = now

            h, w = frame.shape[:2]

            # ---- YOLO detection ----
            results = model(frame, verbose=False)
            centroids = []
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    confidence = float(box.conf[0])
                    if class_id == 0 and confidence > 0.5:
                        x1, y1, x2, y2 = box.xyxy[0]
                        cx, cy = float((x1 + x2) / 2), float((y1 + y2) / 2)
                        centroids.append((cx, cy))

            person_count = len(centroids)

            # ---- Tracking (persistent IDs + movement) ----
            tracks = tracker.update(centroids, now)
            avg_speed = tracker.average_speed()

            # ---- Trend / growth rate over the window ----
            count_history.append((now, person_count))
            old = [c for t, c in count_history if now - t <= TREND_WINDOW_SEC]
            if len(old) >= 2:
                growth_rate = (old[-1] - old[0]) / max(now - count_history[0][0], 1e-6)
            else:
                growth_rate = 0.0

            # ---- Zone density grid ----
            grid = zone_density(centroids, w, h)
            draw_zone_overlay(frame, grid)

            # ---- Fused Crowd Risk Score ----
            risk_score = compute_risk_score(person_count, avg_speed, growth_rate)
            label, color = risk_label(risk_score)

            # ---- Draw detections + track IDs ----
            annotated_frame = results[0].plot()
            draw_zone_overlay(annotated_frame, grid)
            for tid, t in tracks.items():
                cx, cy = t["centroid"]
                cv2.putText(annotated_frame, f"ID {tid}", (int(cx), int(cy) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # ---- HUD ----
            cv2.putText(annotated_frame, f"People: {person_count}", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Avg speed: {avg_speed:.1f}px", (20, 75),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"Growth: {growth_rate:+.2f}/s", (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(annotated_frame, f"RISK: {risk_score} ({label})", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

            if label == "CRITICAL":
                print(f"[ALERT] Crowd risk CRITICAL: score={risk_score}, count={person_count}, "
                      f"growth={growth_rate:.2f}/s, avg_speed={avg_speed:.1f}px")

            display_frame = cv2.resize(annotated_frame, (1280, 720))
            cv2.imshow("Drone-Crowd-Vision", display_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                raise KeyboardInterrupt

        response.close()

    except KeyboardInterrupt:
        print("\nProgram stopped.")
        break

    except Exception as error:
        print(f"\nStream interrupted: {error}")
        print("Reconnecting in 3 seconds...")
        time.sleep(3)

cv2.destroyAllWindows()