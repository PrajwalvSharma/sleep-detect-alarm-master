import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import time
import os
import threading
import urllib.request

try:
    import pygame
    pygame.mixer.init()
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False
    print("[WARNING] pygame not found. Install it with:  pip install pygame-ce")

# ── Tunable settings ────────────────────────────────────────────────────────
EAR_THRESHOLD      = 0.22   # Eye openness threshold — lower = less sensitive
EYE_CLOSED_SECONDS = 2.5    # Seconds eyes must stay closed to trigger alarm
ALARM_SOUND_FILE   = r"dragon-studio-censor-beep-3-372460.mp3"
CAMERA_INDEX       = 0      # Change to 1 if webcam shows black screen
# ────────────────────────────────────────────────────────────────────────────

MODEL_FILENAME = "face_landmarker.task"
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), MODEL_FILENAME)
MODEL_URL  = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
)

LEFT_EYE_INDICES  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33,  160, 158, 133, 153, 144]


def download_model_if_needed():
    if os.path.isfile(MODEL_PATH):
        return
    print("[INFO] Downloading face landmarker model (~7 MB) — first run only...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("[INFO] Model saved to:", MODEL_PATH)
    except Exception as e:
        print(f"[ERROR] Could not download model: {e}")
        raise


def eye_aspect_ratio(landmarks, eye_indices, frame_w, frame_h):
    pts = []
    for idx in eye_indices:
        lm = landmarks[idx]
        pts.append(np.array([lm.x * frame_w, lm.y * frame_h]))
    A = np.linalg.norm(pts[1] - pts[5])
    B = np.linalg.norm(pts[2] - pts[4])
    C = np.linalg.norm(pts[0] - pts[3])
    ear = (A + B) / (2.0 * C)
    return ear


def play_alarm():
    if not PYGAME_AVAILABLE:
        print("\a[ALARM] Eyes closed too long! (No sound – pygame not installed)")
        return
    sound_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ALARM_SOUND_FILE)
    if not os.path.isfile(sound_path):
        print(f"[ALARM] Sound file '{ALARM_SOUND_FILE}' not found – playing beep only.")
        print("\a")
        return
    try:
        pygame.mixer.music.load(sound_path)
        pygame.mixer.music.play(-1)
    except Exception as e:
        print(f"[ALARM] Could not play sound: {e}")


def stop_alarm():
    if not PYGAME_AVAILABLE:
        return
    try:
        if pygame.mixer.music.get_busy():
            pygame.mixer.music.stop()
    except Exception:
        pass


def draw_status_overlay(frame, alarm_active, elapsed, ear_avg):
    h, w = frame.shape[:2]
    if alarm_active:
        color    = (0, 0, 220)
        bg_color = (0, 0, 180)
        label    = "SLEEPING! WAKE UP!"
    else:
        color    = (0, 200, 50)
        bg_color = (0, 150, 40)
        label    = "AWAKE"

    if alarm_active:
        pulse   = int(abs(np.sin(time.time() * 4)) * 80)
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.20 + pulse / 1000, frame, 0.80 - pulse / 1000, 0, frame)

    pill_w, pill_h = 340, 54
    pill_x = (w - pill_w) // 2
    pill_y = 14
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), bg_color, -1, cv2.LINE_AA)
    cv2.rectangle(frame, (pill_x, pill_y), (pill_x + pill_w, pill_y + pill_h), color, 2, cv2.LINE_AA)
    cv2.putText(frame, label, (pill_x + 20, pill_y + 36),
                cv2.FONT_HERSHEY_DUPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)

    dot_x = pill_x - 30
    dot_y  = pill_y + pill_h // 2
    cv2.circle(frame, (dot_x, dot_y), 14, color, -1, cv2.LINE_AA)
    cv2.circle(frame, (dot_x, dot_y), 14, (255, 255, 255), 2, cv2.LINE_AA)

    bar_h = 50
    bar_y = h - bar_h
    cv2.rectangle(frame, (0, bar_y), (w, h), (20, 20, 20), -1)
    ear_text    = f"EAR: {ear_avg:.3f}  (thresh: {EAR_THRESHOLD})"
    closed_text = f"Eyes closed: {elapsed:.1f}s / {EYE_CLOSED_SECONDS}s"
    cv2.putText(frame, ear_text,    (14, bar_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(frame, closed_text, (14, bar_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA)

    if elapsed > 0 and not alarm_active:
        progress    = min(elapsed / EYE_CLOSED_SECONDS, 1.0)
        bar_fill_w  = int((w - 28) * progress)
        cv2.rectangle(frame, (14, bar_y + 45), (14 + bar_fill_w, bar_y + 49),
                      (0, 165, 255), -1, cv2.LINE_AA)
    return frame


def draw_face_box(frame, landmarks, color, w, h):
    xs = [int(lm.x * w) for lm in landmarks]
    ys = [int(lm.y * h) for lm in landmarks]
    pad = 10
    x1, y1 = max(0, min(xs) - pad), max(0, min(ys) - pad)
    x2, y2 = min(w, max(xs) + pad), min(h, max(ys) + pad)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    corner = 18
    thick  = 3
    for cx, cy, dx, dy in [(x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)]:
        cv2.line(frame, (cx, cy), (cx + dx * corner, cy), color, thick, cv2.LINE_AA)
        cv2.line(frame, (cx, cy), (cx, cy + dy * corner), color, thick, cv2.LINE_AA)


def main():
    # Download model on first run
    download_model_if_needed()

    # Build FaceLandmarker with the new Tasks API
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        num_faces=1,
        min_face_detection_confidence=0.6,
        min_face_presence_confidence=0.6,
        min_tracking_confidence=0.6,
    )
    detector = vision.FaceLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open webcam (index {CAMERA_INDEX}). "
              f"Try changing CAMERA_INDEX to 1 at the top of the script.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    eye_closed_start = None
    alarm_active     = False
    alarm_thread     = None

    print("=" * 60)
    print("  Sleep Detector — press Q to quit")
    print(f"  Alarm fires after {EYE_CLOSED_SECONDS}s of closed eyes")
    if PYGAME_AVAILABLE:
        print(f"  Alarm sound: {ALARM_SOUND_FILE}")
    print("=" * 60)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        h, w  = frame.shape[:2]

        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        results  = detector.detect(mp_image)

        ear_avg      = 1.0
        elapsed      = 0.0
        face_detected = False

        if results.face_landmarks:
            face_detected = True
            lms   = results.face_landmarks[0]   # list of NormalizedLandmark
            ear_l = eye_aspect_ratio(lms, LEFT_EYE_INDICES,  w, h)
            ear_r = eye_aspect_ratio(lms, RIGHT_EYE_INDICES, w, h)
            ear_avg     = (ear_l + ear_r) / 2.0
            eyes_closed = ear_avg < EAR_THRESHOLD

            if eyes_closed:
                if eye_closed_start is None:
                    eye_closed_start = time.time()
                elapsed = time.time() - eye_closed_start
                if elapsed >= EYE_CLOSED_SECONDS and not alarm_active:
                    alarm_active  = True
                    alarm_thread  = threading.Thread(target=play_alarm, daemon=True)
                    alarm_thread.start()
            else:
                eye_closed_start = None
                if alarm_active:
                    alarm_active = False
                    stop_alarm()

            face_color = (0, 0, 220) if alarm_active else (0, 220, 60)
            draw_face_box(frame, lms, face_color, w, h)
        else:
            eye_closed_start = None
            if alarm_active:
                alarm_active = False
                stop_alarm()

        frame = draw_status_overlay(frame, alarm_active, elapsed, ear_avg)

        if not face_detected:
            cv2.putText(frame, "No face detected",
                        (w // 2 - 130, h // 2),
                        cv2.FONT_HERSHEY_DUPLEX, 1.0, (0, 180, 255), 2, cv2.LINE_AA)

        cv2.imshow("Sleep Alarm Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    stop_alarm()
    cap.release()
    detector.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
