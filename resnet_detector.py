#!/usr/bin/env python3
"""
resnet_detector.py
==================
Phát hiện ngủ gật realtime bằng ResNet50 AI.
Pipeline:
  Camera → Dlib (68 landmarks) → Tách vùng mắt → ResNet50 → Phán đoán

Yêu cầu:
  - models/resnet_drowsiness.pth  (chạy train_resnet.py trước)
  - models/shape_predictor_68_face_landmarks.dat
  - pip install torch torchvision dlib opencv-python numpy scipy

Điều khiển:
  q / ESC  — Thoát
  r        — Reset cảnh báo
"""

import os
import sys
import time
import threading

import cv2
import numpy as np

# ── Kiểm tra thư viện ─────────────────────────────────────────────
def _check_deps():
    missing = []
    for lib in ("torch", "torchvision", "dlib"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        # Hiển thị lỗi qua OpenCV window (vì chạy dưới subprocess)
        blank = np.zeros((300, 700, 3), dtype=np.uint8)
        lines = [
            "THIEU THU VIEN:",
            f"  {', '.join(missing)}",
            "",
            "Cai dat:",
            f"  pip install {' '.join(missing)}",
        ]
        for i, line in enumerate(lines):
            cv2.putText(blank, line, (20, 50 + i * 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 255), 2)
        cv2.imshow("ResNet Detector - LOI", blank)
        cv2.waitKey(0)
        sys.exit(1)

_check_deps()

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import dlib

# ── Đường dẫn ─────────────────────────────────────────────────────
_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(_DIR, "models", "resnet_drowsiness.pth")
DLIB_PATH      = os.path.join(_DIR, "models", "shape_predictor_68_face_landmarks.dat")
ALARM_PATH     = os.path.join(_DIR, "alarm.wav")

# ── Hằng số phát hiện ─────────────────────────────────────────────
DROWSY_SECS     = 0.8   # Giây nhắm mắt → cảnh báo DROWSY
MICROSLEEP_SECS = 1.8   # Giây nhắm mắt → MICROSLEEP
IMG_SIZE        = 224
EYE_PADDING     = 15    # Pixel padding xung quanh vùng mắt
RESIZE_HEIGHT   = 460   # Chiều cao resize frame (giống blinkDetect.py)

# Chỉ số landmark mắt (dlib 68 points)
LEFT_EYE_IDX  = list(range(36, 42))
RIGHT_EYE_IDX = list(range(42, 48))


# ── Âm thanh ──────────────────────────────────────────────────────
class AudioManager:
    def __init__(self, sound_path: str):
        self._playing = False
        self._lock    = threading.Lock()
        self._sound   = None
        self._lib     = None

        if not os.path.exists(sound_path):
            return
        try:
            import pygame
            pygame.mixer.init()
            self._sound = pygame.mixer.Sound(sound_path)
            self._lib   = "pygame"
        except Exception:
            pass

    def start(self):
        with self._lock:
            if self._playing or self._sound is None:
                return
            self._playing = True
        def _loop():
            while True:
                with self._lock:
                    if not self._playing:
                        break
                if self._lib == "pygame":
                    self._sound.play()
                    import pygame
                    while pygame.mixer.get_busy():
                        time.sleep(0.05)
                        with self._lock:
                            if not self._playing:
                                self._sound.stop()
                                return
        threading.Thread(target=_loop, daemon=True).start()

    def stop(self):
        with self._lock:
            self._playing = False
        if self._lib == "pygame" and self._sound is not None:
            try:
                self._sound.stop()
            except Exception:
                pass


# ── Load ResNet model ─────────────────────────────────────────────
def load_model(path: str, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Khong tim thay ResNet model: {path}\n"
            f"Hay chay: python train_resnet.py\n"
            f"(Training mat khoang 15-25 phut tren Apple Silicon)"
        )

    # weights_only=False vì checkpoint chứa dict (class_to_idx) ngoài tensor
    ckpt = torch.load(path, map_location=device, weights_only=False)

    idx_to_class = ckpt.get("idx_to_class", {0: "awake", 1: "sleepy"})
    class_to_idx = ckpt.get("class_to_idx", {"awake": 0, "sleepy": 1})

    # Tìm index "mắt nhắm" một cách linh hoạt
    closed_idx = None
    for cls_name, idx in class_to_idx.items():
        if any(kw in cls_name.lower() for kw in ("sleep", "closed", "drowsy", "close", "shut")):
            closed_idx = idx
            break
    if closed_idx is None:
        # Fallback: class có index cao hơn thường là "bất thường"
        closed_idx = max(class_to_idx.values())
        print(f"[WARN] Không nhận ra class 'closed' từ {class_to_idx}, dùng index={closed_idx}")

    closed_label = idx_to_class[closed_idx]
    print(f"[INFO] Classes: {class_to_idx}")
    print(f"[INFO] 'Nhắm mắt' → class '{closed_label}' (index {closed_idx})")
    print(f"[INFO] Val accuracy đã train: {ckpt.get('val_acc', 'N/A'):.2f}%")

    # Xây model
    model = resnet50(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, 2),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, closed_idx, idx_to_class


# ── Preprocessing ─────────────────────────────────────────────────
_transform = T.Compose([
    T.ToPILImage(),
    T.Lambda(lambda img: img.convert("RGB")),
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def predict_eye(model, device, eye_bgr, closed_idx: int):
    """Trả về (is_closed: bool, closed_prob: float)."""
    if eye_bgr is None or eye_bgr.size == 0:
        return False, 0.0
    h, w = eye_bgr.shape[:2]
    if h < 8 or w < 8:
        return False, 0.0
    tensor = _transform(eye_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1)[0]
    closed_prob = probs[closed_idx].item()
    return closed_prob > 0.5, closed_prob


def get_eye_roi(frame, landmarks, eye_indices, padding: int = EYE_PADDING):
    """Cắt vùng mắt từ frame theo landmarks dlib."""
    pts = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in eye_indices])
    x, y, w, h = cv2.boundingRect(pts)
    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(frame.shape[1], x + w + padding)
    y2 = min(frame.shape[0], y + h + padding)
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2 - x1, y2 - y1)


# ── Overlay helpers ───────────────────────────────────────────────
def draw_prob_bar(frame, x, y, w, h, prob, label):
    """Thanh tiến trình hiển thị xác suất."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * prob)
    color = (0, 0, 255) if prob > 0.5 else (0, 200, 0)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    cv2.putText(frame, f"{label}: {prob*100:.1f}%", (x + 4, y + h - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)


# ── Main loop ─────────────────────────────────────────────────────
def main():
    # Device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[INFO] Device: {device}")

    # Load model (hiển thị lỗi qua OpenCV nếu thiếu)
    try:
        model, closed_idx, idx_to_class = load_model(MODEL_PATH, device)
    except FileNotFoundError as e:
        blank = np.zeros((360, 760, 3), dtype=np.uint8)
        for i, line in enumerate(str(e).split("\n")):
            cv2.putText(blank, line, (20, 60 + i * 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        cv2.imshow("ResNet Detector - LOI", blank)
        cv2.waitKey(0)
        sys.exit(1)

    # Load dlib
    if not os.path.exists(DLIB_PATH):
        blank = np.zeros((200, 700, 3), dtype=np.uint8)
        cv2.putText(blank, f"Khong tim thay: {DLIB_PATH}", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 100, 255), 2)
        cv2.putText(blank, "Hay chay: python download_models.py", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
        cv2.imshow("ResNet Detector - LOI", blank)
        cv2.waitKey(0)
        sys.exit(1)

    detector  = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(DLIB_PATH)

    audio = AudioManager(ALARM_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không mở được camera!")
        sys.exit(1)

    # State
    eyes_closed_since = None   # float | None
    alarm_on          = False
    left_prob_ema     = 0.0    # EMA để hiển thị mượt
    right_prob_ema    = 0.0
    EMA_ALPHA         = 0.4

    print("[INFO] Bắt đầu phát hiện... (q/ESC=thoát, r=reset)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize
        h, w = frame.shape[:2]
        scale  = RESIZE_HEIGHT / h
        frame  = cv2.resize(frame, (int(w * scale), RESIZE_HEIGHT))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray, 0)

        # ── Mặc định nếu không có mặt ──────────────────────────────
        status_text  = "Khong phat hien khuon mat"
        status_color = (0, 165, 255)
        both_closed  = False

        for face in faces:
            landmarks = predictor(gray, face)

            # Khung mặt
            x1, y1 = face.left(), face.top()
            x2, y2 = face.right(), face.bottom()
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)

            # Tách vùng mắt
            left_roi,  left_rect  = get_eye_roi(frame, landmarks, LEFT_EYE_IDX)
            right_roi, right_rect = get_eye_roi(frame, landmarks, RIGHT_EYE_IDX)

            l_closed, l_prob = predict_eye(model, device, left_roi,  closed_idx)
            r_closed, r_prob = predict_eye(model, device, right_roi, closed_idx)

            # EMA
            left_prob_ema  = EMA_ALPHA * l_prob + (1 - EMA_ALPHA) * left_prob_ema
            right_prob_ema = EMA_ALPHA * r_prob + (1 - EMA_ALPHA) * right_prob_ema

            # Vẽ khung mắt
            lx, ly, lw, lh = left_rect
            rx, ry, rw, rh = right_rect
            l_col = (0, 0, 255) if l_closed else (0, 220, 0)
            r_col = (0, 0, 255) if r_closed else (0, 220, 0)
            cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh), l_col, 1)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), r_col, 1)

            both_closed = l_closed and r_closed

        # ── State machine thời gian thực ───────────────────────────
        now = time.time()
        if both_closed:
            if eyes_closed_since is None:
                eyes_closed_since = now
            closed_dur = now - eyes_closed_since

            if closed_dur >= MICROSLEEP_SECS:
                status_text  = f"!!! MICROSLEEP ({closed_dur:.1f}s) !!!"
                status_color = (0, 0, 220)
            elif closed_dur >= DROWSY_SECS:
                status_text  = f"!!! CANH BAO NGU GAT ({closed_dur:.1f}s) !!!"
                status_color = (0, 0, 255)
            else:
                status_text  = f"Mat dang nhep ({closed_dur:.1f}s)..."
                status_color = (0, 165, 255)

            if closed_dur >= DROWSY_SECS and not alarm_on:
                audio.start()
                alarm_on = True
        else:
            eyes_closed_since = None
            if len(faces) > 0:
                avg_prob = (left_prob_ema + right_prob_ema) / 2
                status_text  = f"Tinh tao  |  Prob nham: {avg_prob*100:.1f}%"
                status_color = (0, 200, 0)
            if alarm_on:
                audio.stop()
                alarm_on = False

        # ── Overlay ────────────────────────────────────────────────
        # Thanh đen nền phía trên
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        cv2.putText(frame, "ResNet AI - Phat hien Ngu Gat", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)
        cv2.putText(frame, status_text, (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, status_color, 2)

        # Thanh xác suất mắt trái / phải
        bw = int(frame.shape[1] / 2) - 30
        draw_prob_bar(frame, 10, frame.shape[0] - 28, bw, 22,
                      left_prob_ema,  "Mat trai")
        draw_prob_bar(frame, bw + 30, frame.shape[0] - 28, bw, 22,
                      right_prob_ema, "Mat phai")

        cv2.imshow("ResNet AI - Phat hien Ngu Gat  (q=thoat, r=reset)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):   # q hoặc ESC
            break
        elif key == ord("r"):
            eyes_closed_since = None
            alarm_on = False
            audio.stop()

    audio.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
