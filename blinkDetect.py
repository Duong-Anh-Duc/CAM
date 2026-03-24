# -*- coding: utf-8 -*-
"""
Phát hiện ngủ gật đơn giản bằng EAR (Eye Aspect Ratio).
Dùng MediaPipe FaceMesh — không cần dlib, dễ cài trên Windows.
"""

import os
import sys
import cv2
import time
import numpy as np
from scipy.spatial import distance as dist
from threading import Thread
import queue

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sound_path = os.path.join(_BASE_DIR, "alarm.wav")

# Audio
try:
    import pygame
    pygame.mixer.init()
    AUDIO_LIB = 'pygame'
except ImportError:
    try:
        from playsound import playsound as _playsound_func
        AUDIO_LIB = 'playsound'
    except ImportError:
        AUDIO_LIB = None

# MediaPipe
import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# MediaPipe eye landmark indices (tương đương dlib 36-47)
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

EAR_THRESH = 0.22
DROWSY_SECS = 1.5
ALARM_ON = False
threadStatusQ = queue.Queue()


def eye_aspect_ratio(pts):
    A = dist.euclidean(pts[1], pts[5])
    B = dist.euclidean(pts[2], pts[4])
    C = dist.euclidean(pts[0], pts[3])
    return (A + B) / (2.0 * max(C, 1e-6))


def get_eye_pts(landmarks, indices, w, h):
    return [(int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices]


def soundAlert(path, statusQ):
    if AUDIO_LIB == 'pygame':
        try:
            sound = pygame.mixer.Sound(path)
            while True:
                if not statusQ.empty():
                    if statusQ.get():
                        pygame.mixer.stop()
                        return
                sound.play()
                while pygame.mixer.get_busy():
                    if not statusQ.empty():
                        if statusQ.get():
                            pygame.mixer.stop()
                            return
                    pygame.time.wait(50)
        except Exception:
            pass
    elif AUDIO_LIB == 'playsound':
        while True:
            if not statusQ.empty():
                if statusQ.get():
                    return
            try:
                _playsound_func(path)
            except Exception:
                break


def main():
    global ALARM_ON

    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không mở được camera!")
        sys.exit(1)

    eyes_closed_since = None
    blink_count = 0
    prev_closed = False
    total_frames = 0
    closed_frames = 0
    win_name = "Blink Detection"

    print("Phát hiện ngủ gật — q/ESC=thoát, r=reset")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        ear = 0.3  # default
        is_closed = False

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            left_pts = get_eye_pts(lms, LEFT_EYE, w, h)
            right_pts = get_eye_pts(lms, RIGHT_EYE, w, h)

            left_ear = eye_aspect_ratio(left_pts)
            right_ear = eye_aspect_ratio(right_pts)
            ear = (left_ear + right_ear) / 2.0

            is_closed = ear < EAR_THRESH

            # Vẽ landmarks mắt
            for pt in left_pts + right_pts:
                cv2.circle(frame, pt, 1, (0, 255, 0), -1)

            # Đếm blink
            if prev_closed and not is_closed:
                blink_count += 1
            prev_closed = is_closed

        # Thống kê %
        total_frames += 1
        if is_closed:
            closed_frames += 1
        pct_closed = (closed_frames / total_frames) * 100
        pct_open = 100 - pct_closed

        # State machine
        now = time.time()
        if is_closed:
            if eyes_closed_since is None:
                eyes_closed_since = now
            dur = now - eyes_closed_since

            if dur >= DROWSY_SECS:
                cv2.putText(frame, f"!!! CANH BAO NGU GAT ({dur:.1f}s) !!!",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                if not ALARM_ON:
                    ALARM_ON = True
                    while not threadStatusQ.empty():
                        threadStatusQ.get()
                    Thread(target=soundAlert, args=(sound_path, threadStatusQ), daemon=True).start()
            else:
                cv2.putText(frame, f"Mat nhep... ({dur:.1f}s)",
                            (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        else:
            eyes_closed_since = None
            if ALARM_ON:
                ALARM_ON = False
                threadStatusQ.put(True)
                if AUDIO_LIB == 'pygame':
                    try:
                        pygame.mixer.stop()
                    except Exception:
                        pass

        # Info
        eye_status = "NHAM" if is_closed else "MO"
        eye_color = (0, 0, 255) if is_closed else (0, 255, 0)
        cv2.putText(frame, f"Mat: {eye_status}  EAR: {ear:.2f}",
                    (30, frame.shape[0] - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, eye_color, 2)
        cv2.putText(frame, f"Blinks: {blink_count}  |  Mo: {pct_open:.1f}%  Nham: {pct_closed:.1f}%",
                    (30, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        cv2.imshow(win_name, frame)

        k = cv2.waitKey(1) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == ord('r'):
            eyes_closed_since = None
            ALARM_ON = False
            blink_count = 0
            total_frames = 0
            closed_frames = 0
            threadStatusQ.put(True)
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    if ALARM_ON:
        threadStatusQ.put(True)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
