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
from PIL import Image, ImageDraw, ImageFont


def _find_font(size=18):
    """Tìm font hỗ trợ tiếng Việt trên hệ thống."""
    import platform
    font_paths = []
    if platform.system() == "Windows":
        win_fonts = os.path.join(os.environ.get("WINDIR", "C:\\Windows"), "Fonts")
        font_paths = [
            os.path.join(win_fonts, "arial.ttf"),
            os.path.join(win_fonts, "segoeui.ttf"),
            os.path.join(win_fonts, "msyh.ttc"),       # Microsoft YaHei (hỗ trợ Unicode tốt)
            os.path.join(win_fonts, "malgun.ttf"),      # Malgun Gothic
        ]
    elif platform.system() == "Darwin":
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        ]
    for fp in font_paths:
        if os.path.exists(fp):
            try:
                return ImageFont.truetype(fp, size)
            except Exception:
                pass
    return ImageFont.load_default()


_FONT_CACHE: dict[int, ImageFont.FreeTypeFont] = {}


def _get_font(size=18):
    if size not in _FONT_CACHE:
        _FONT_CACHE[size] = _find_font(size)
    return _FONT_CACHE[size]


def put_vn_text(frame, text, pos, font_size=18, color=(255, 255, 255),
                bg_color=None):
    """Vẽ text Unicode (tiếng Việt) lên frame OpenCV bằng PIL."""
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font = _get_font(font_size)
    if bg_color:
        bbox = draw.textbbox(pos, text, font=font)
        pad = 3
        draw.rectangle(
            [bbox[0] - pad, bbox[1] - pad, bbox[2] + pad, bbox[3] + pad],
            fill=bg_color)
    draw.text(pos, text, font=font, fill=color)
    result = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    np.copyto(frame, result)


# ── Kiểm tra thư viện ─────────────────────────────────────────────
def _check_deps():
    missing = []
    for lib in ("torch", "torchvision", "mediapipe"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        # Hiển thị lỗi qua OpenCV window (vì chạy dưới subprocess)
        blank = np.zeros((300, 700, 3), dtype=np.uint8)
        lines = [
            f"Thiếu thư viện: {', '.join(missing)}",
            "",
            f"Cài đặt: pip install {' '.join(missing)}",
        ]
        for i, line in enumerate(lines):
            put_vn_text(blank, line, (20, 30 + i * 40), font_size=18, color=(100, 100, 255))
        cv2.imshow("ResNet Detector - Lỗi", blank)
        cv2.waitKey(0)
        sys.exit(1)

_check_deps()

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.models import resnet50, ResNet50_Weights
import mediapipe as mp

# ── Đường dẫn ─────────────────────────────────────────────────────
_DIR           = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH     = os.path.join(_DIR, "models", "resnet_drowsiness.pth")
ALARM_PATH     = os.path.join(_DIR, "alarm.wav")

# MediaPipe eye indices
LEFT_EYE_MP  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_MP = [362, 385, 387, 263, 373, 380]

# ── Hằng số phát hiện ─────────────────────────────────────────────
DROWSY_SECS     = 0.8   # Giây nhắm mắt → cảnh báo DROWSY
MICROSLEEP_SECS = 1.8   # Giây nhắm mắt → MICROSLEEP
IMG_SIZE        = 224
EYE_PADDING     = 15    # Pixel padding xung quanh vùng mắt


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
        # Dừng ngay lập tức
        if self._lib == "pygame" and self._sound is not None:
            try:
                self._sound.stop()
            except Exception:
                pass


# ── Load ResNet model ─────────────────────────────────────────────
def load_model(path: str, device):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Không tìm thấy ResNet model: {path}\n"
            f"Hãy chạy: python train_resnet.py\n"
            f"(Training mất khoảng 15-25 phút)"
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
    """Trả về (is_closed: bool, drowsy_score: float 0-1).
    Dùng hiệu logit (sleepy - awake) qua sigmoid để có score biến động mượt."""
    if eye_bgr is None or eye_bgr.size == 0:
        return False, 0.0
    h, w = eye_bgr.shape[:2]
    if h < 8 or w < 8:
        return False, 0.0
    tensor = _transform(eye_bgr).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)[0]
        awake_idx = 1 - closed_idx
        # Hiệu logit chia temperature lớn → score biến động mượt
        diff = (logits[closed_idx] - logits[awake_idx]).item()
        score = 1.0 / (1.0 + np.exp(-diff / 8.0))  # temperature=8
    return score > 0.5, score


def get_eye_roi_mp(frame, lms, eye_indices, w, h, padding: int = EYE_PADDING):
    """Cắt vùng mắt từ frame theo MediaPipe landmarks."""
    pts = np.array([(int(lms[i].x * w), int(lms[i].y * h)) for i in eye_indices])
    bx, by, bw, bh = cv2.boundingRect(pts)
    x1 = max(0, bx - padding)
    y1 = max(0, by - padding)
    x2 = min(w, bx + bw + padding)
    y2 = min(h, by + bh + padding)
    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2 - x1, y2 - y1)


def get_eye_pts_mp(lms, indices, w, h):
    """Lấy tọa độ pixel của eye landmarks từ MediaPipe."""
    return [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]


# ── Overlay helpers ───────────────────────────────────────────────
def draw_prob_bar(frame, x, y, w, h, prob, label):
    """Thanh tiến trình hiển thị xác suất."""
    cv2.rectangle(frame, (x, y), (x + w, y + h), (50, 50, 50), -1)
    fill = int(w * prob)
    color = (0, 0, 255) if prob > 0.5 else (0, 200, 0)
    cv2.rectangle(frame, (x, y), (x + fill, y + h), color, -1)
    cv2.rectangle(frame, (x, y), (x + w, y + h), (200, 200, 200), 1)
    put_vn_text(frame, f"{label}: {prob*100:.1f}%", (x + 4, y + 2),
                font_size=14, color=(255, 255, 255))


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
            put_vn_text(blank, line, (20, 40 + i * 50), font_size=18, color=(100, 100, 255))
        cv2.imshow("ResNet Detector - Lỗi", blank)
        cv2.waitKey(0)
        sys.exit(1)

    # MediaPipe FaceMesh
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1, refine_landmarks=True,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    audio = AudioManager(ALARM_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[LỖI] Không mở được camera!")
        sys.exit(1)

    eyes_closed_since = None
    alarm_on          = False
    left_prob_ema     = 0.0
    right_prob_ema    = 0.0
    EMA_ALPHA         = 0.4

    from scipy.spatial import distance as dist

    def _ear(pts):
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * max(C, 1e-6))

    print("[INFO] Bắt đầu phát hiện... (q/ESC=thoát, r=reset)")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        status_text  = "Không phát hiện khuôn mặt"
        status_color = (0, 165, 255)
        both_closed  = False
        has_face     = False

        if result.multi_face_landmarks:
            lms = result.multi_face_landmarks[0].landmark
            has_face = True

            # Bbox từ landmarks
            xs = [int(lms[i].x * w) for i in range(468)]
            ys = [int(lms[i].y * h) for i in range(468)]
            x1, y1, x2, y2 = min(xs) - 10, min(ys) - 10, max(xs) + 10, max(ys) + 10
            cv2.rectangle(frame, (x1, y1), (x2, y2), (100, 255, 100), 2)

            # EAR
            left_pts = get_eye_pts_mp(lms, LEFT_EYE_MP, w, h)
            right_pts = get_eye_pts_mp(lms, RIGHT_EYE_MP, w, h)
            ear = (_ear(left_pts) + _ear(right_pts)) / 2.0
            ear_closed = ear < 0.22

            # Eye ROI + ResNet
            left_roi, left_rect = get_eye_roi_mp(frame, lms, LEFT_EYE_MP, w, h)
            right_roi, right_rect = get_eye_roi_mp(frame, lms, RIGHT_EYE_MP, w, h)

            l_closed, l_prob = predict_eye(model, device, left_roi, closed_idx)
            r_closed, r_prob = predict_eye(model, device, right_roi, closed_idx)

            # Kết hợp EAR + ResNet
            if ear_closed:
                l_prob = max(l_prob, 0.7)
                r_prob = max(r_prob, 0.7)
                l_closed = True
                r_closed = True

            left_prob_ema  = EMA_ALPHA * l_prob + (1 - EMA_ALPHA) * left_prob_ema
            right_prob_ema = EMA_ALPHA * r_prob + (1 - EMA_ALPHA) * right_prob_ema

            # Vẽ khung mắt
            lx, ly, lw, lh = left_rect
            rx, ry, rw, rh = right_rect
            cv2.rectangle(frame, (lx, ly), (lx + lw, ly + lh),
                          (0, 0, 255) if l_closed else (0, 220, 0), 1)
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                          (0, 0, 255) if r_closed else (0, 220, 0), 1)

            put_vn_text(frame, f"EAR: {ear:.2f}", (x1, y2 + 5),
                        font_size=14, color=(255, 255, 0))

            both_closed = l_closed and r_closed

        # State machine
        now = time.time()
        if both_closed:
            if eyes_closed_since is None:
                eyes_closed_since = now
            closed_dur = now - eyes_closed_since

            if closed_dur >= MICROSLEEP_SECS:
                status_text  = f"!!! MICROSLEEP ({closed_dur:.1f}s) !!!"
                status_color = (0, 0, 220)
            elif closed_dur >= DROWSY_SECS:
                status_text  = f"!!! CẢNH BÁO NGỦ GẬT ({closed_dur:.1f}s) !!!"
                status_color = (0, 0, 255)
            else:
                status_text  = f"Mắt đang nhép ({closed_dur:.1f}s)..."
                status_color = (0, 165, 255)

            if closed_dur >= DROWSY_SECS and not alarm_on:
                audio.start()
                alarm_on = True
        else:
            eyes_closed_since = None
            if has_face:
                avg_prob = (left_prob_ema + right_prob_ema) / 2
                status_text  = f"Tỉnh táo  |  Prob nhắm: {avg_prob*100:.1f}%"
                status_color = (0, 200, 0)
            if alarm_on:
                audio.stop()
                alarm_on = False

        # ── Overlay ────────────────────────────────────────────────
        # Thanh đen nền phía trên
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 80), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)

        put_vn_text(frame, "ResNet AI - Phát hiện Ngủ Gật", (10, 4),
                    font_size=16, color=(200, 200, 200))
        put_vn_text(frame, status_text, (10, 35),
                    font_size=22, color=status_color[::-1])  # BGR→RGB

        # Thanh xác suất mắt trái / phải
        bw = int(frame.shape[1] / 2) - 30
        draw_prob_bar(frame, 10, frame.shape[0] - 28, bw, 22,
                      left_prob_ema,  "Mắt trái")
        draw_prob_bar(frame, bw + 30, frame.shape[0] - 28, bw, 22,
                      right_prob_ema, "Mắt phải")

        win_name = "ResNet AI Detector"
        cv2.imshow(win_name, frame)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord("r"):
            eyes_closed_since = None
            alarm_on = False
            audio.stop()
        # Đóng bằng nút X
        if cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    audio.stop()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
