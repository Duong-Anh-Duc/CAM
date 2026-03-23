"""
=============================================================================
BEHAVIOR DETECTOR – Hệ thống giám sát tập trung học sinh (v2 – Optimised)
=============================================================================
Phát hiện hành vi ngủ gật & mất tập trung bằng:
  • YOLOv8 (ultralytics)  : phát hiện người + điện thoại
  • Dlib 68-landmark      : EAR, MAR, Head Pose
  • MediaPipe FaceMesh    : fallback khi Dlib tắt
  • OpenCV                : xử lý ảnh, overlay

Hành vi phát hiện:
  1. DROWSY       – EAR thấp kéo dài
  2. MICROSLEEP   – Mắt nhắm liên tục > ngưỡng cao
  3. YAWNING      – Miệng mở rộng (MAR)
  4. HEAD_DOWN    – Cúi đầu (pitch)
  5. HEAD_TURN    – Quay đầu ngang (yaw)
  6. PHONE_USE    – Phone gần person (IoU overlap)
  7. DISTRACTED   – Gaze off-center
  8. MULTI_PERSON – Nhiều người trong khung hình
  9. NO_FACE      – Không thấy mặt > N giây
  10.FATIGUE_HIGH – Combo ngáp + mắt nhắm
=============================================================================
"""

import cv2
import numpy as np
import time
import sys
import os
import math
from threading import Thread
from datetime import datetime
from collections import deque
import queue

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── PIL cho vẽ text Unicode (tiếng Việt có dấu) ────────────────────────────
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
            os.path.join(win_fonts, "msyh.ttc"),
            os.path.join(win_fonts, "malgun.ttf"),
        ]
    elif platform.system() == "Darwin":
        font_paths = [
            "/System/Library/Fonts/Supplemental/Arial Unicode.ttf",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/Library/Fonts/Arial Unicode.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
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


# ─── Scipy ───────────────────────────────────────────────────────────────────
from scipy.spatial import distance as dist

# ─── Audio ────────────────────────────────────────────────────────────────────
try:
    import playsound as _playsound
    AUDIO_LIB = 'playsound'
except ImportError:
    try:
        import pygame
        pygame.mixer.init()
        AUDIO_LIB = 'pygame'
    except ImportError:
        AUDIO_LIB = None

# ─── Dlib ────────────────────────────────────────────────────────────────────
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    print("[WARN] dlib không khả dụng – tắt EAR detection")

# ─── MediaPipe ────────────────────────────────────────────────────────────────
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_face_mesh = mp.solutions.face_mesh
        mp_drawing   = mp.solutions.drawing_utils
        MP_AVAILABLE = True
    else:
        MP_AVAILABLE = False
        mp_face_mesh = None
        mp_drawing   = None
        print("[WARN] mediapipe Tasks API detected – FaceMesh không dùng Tasks API")
except ImportError:
    MP_AVAILABLE = False
    mp_face_mesh = None
    mp_drawing   = None
    print("[WARN] mediapipe không khả dụng – dùng Dlib cho facial analysis")

# ─── YOLO (ultralytics) ───────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARN] ultralytics (YOLOv8) không khả dụng – tắt object detection")


# =============================================================================
# CONSTANTS  (v3 – cải tiến toàn diện)
# =============================================================================

# --- EAR ---------------------------------------------------------------
EAR_THRESH_DEFAULT    = 0.22       # ngưỡng mặc định (dùng khi chưa calibrate)
EAR_ADAPTIVE_RATIO    = 0.75       # ngưỡng adaptive = baseline * ratio
EAR_CALIBRATION_SECS  = 4.0        # thời gian calibrate EAR mở mắt bình thường
EAR_BLINK_FRAMES      = 3          # >= 3 frame liên tiếp EAR < thresh = 1 blink

# --- Drowsy / Microsleep: TIME-BASED (không phụ thuộc FPS) -------------
DROWSY_TIME_SECS      = 0.8        # mắt nhắm >= 0.8s = cảnh báo ngủ gật
MICROSLEEP_TIME_SECS  = 1.8        # mắt nhắm >= 1.8s = microsleep

# --- MAR ---------------------------------------------------------------
MAR_THRESH            = 0.45       # hạ từ 0.55 → 0.45: nhạy hơn với ngáp nhẹ
MAR_YAWN_MIN_SECS     = 1.5        # ngáp kéo dài ít nhất 1.5s
MAR_YAWN_MAX_SECS     = 6.0        # ngáp tối đa 6s (lâu hơn = nói chuyện)
MAR_TALK_VARIANCE     = 0.04       # nếu variance MAR > 0.04 trong 1s → đang nói

# --- Head Pose (độ) ---------------------------------------------------
PITCH_DOWN_THRESH     = 18         # cúi đầu > 18°
PITCH_UP_THRESH       = -20        # ngẩng đầu
YAW_THRESH            = 25         # quay đầu ngang > 25°
HEAD_POSE_FRAMES      = 12         # frame liên tiếp mới cảnh báo

# --- Gaze detection (iris-based, MediaPipe) ----------------------------
GAZE_OFF_CENTER_THRESH = 0.28      # iris lệch khỏi center mắt > 28% = mất tập trung
GAZE_CONSEC_SECS       = 1.5       # nhìn lệch >= 1.5s mới cảnh báo

# --- Blink rate analysis -----------------------------------------------
BLINK_RATE_WINDOW_SECS = 60.0      # sliding window 60s
BLINK_RATE_LOW         = 8         # < 8 blinks/phút = buồn ngủ
BLINK_RATE_HIGH        = 28        # > 28 blinks/phút = mệt mỏi / kích thích

# --- Combo detection (sliding window) ----------------------------------
COMBO_WINDOW_SECS      = 300.0     # 5 phút window cho combo yawn + drowsy
COMBO_YAWN_THRESHOLD   = 2         # >= 2 lần ngáp trong window

# --- Misc ---------------------------------------------------------------
NO_FACE_SECONDS       = 3.0
YOLO_PERSON_CONF      = 0.45       # confidence cho person detection
YOLO_PHONE_CONF       = 0.30       # confidence thấp hơn cho phone (khó detect hơn)

# --- EMA smoothing – TÁCH RIÊNG detection vs display -------------------
EMA_ALPHA_DETECT      = 0.6        # detection: phản hồi nhanh
EMA_ALPHA_DISPLAY     = 0.3        # display: hiển thị mượt

# --- Face tracking IOU threshold + grace period -------------------------
FACE_TRACK_IOU_MIN    = 0.25
FACE_TRACK_GRACE_FRAMES = 8        # giữ person state thêm 8 frame sau khi mất (IoU matching)
FACE_TRACK_REID_SECS  = 120.0      # giữ embedding 2 phút để re-identify khi quay lại
FACE_TRACK_EMBED_THRESH = 0.45     # ngưỡng khoảng cách embedding (< 0.45 = cùng người)
FACE_TRACK_REID_DIST  = 0.35       # fallback: ngưỡng khoảng cách center nếu không có embedding

# --- CLAHE (image preprocessing) ---------------------------------------
CLAHE_CLIP_LIMIT      = 2.5
CLAHE_GRID_SIZE       = (8, 8)

# --- Focal length estimation -------------------------------------------
FOCAL_LENGTH_RATIO    = 0.85       # focal ≈ width * 0.85 (webcam 65° FOV typical)

# --- MediaPipe landmark indices ----------------------------------------
MP_LEFT_EYE  = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Outer lip
MP_MOUTH_OUTER = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
                  291, 375, 321, 405, 314, 17, 84, 181, 91, 146]

# YOLO class IDs (COCO)
YOLO_PERSON_ID = 0
YOLO_PHONE_ID  = 67

# Severity levels
SEVERITY_LOW      = "LOW"
SEVERITY_MEDIUM   = "MEDIUM"
SEVERITY_HIGH     = "HIGH"
SEVERITY_CRITICAL = "CRITICAL"

# Colors (BGR)
COLOR_GREEN  = (0, 220, 50)
COLOR_YELLOW = (0, 200, 255)
COLOR_ORANGE = (0, 130, 255)
COLOR_RED    = (0, 0, 255)
COLOR_BLUE   = (255, 100, 0)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)
COLOR_CYAN   = (255, 200, 0)
COLOR_PURPLE = (200, 0, 200)


# =============================================================================
# ALERT EVENT
# =============================================================================
class AlertEvent:
    __slots__ = ('behavior_type', 'severity', 'message', 'person_id',
                 'timestamp', 'active')

    def __init__(self, behavior_type, severity, message, person_id=0):
        self.behavior_type = behavior_type
        self.severity      = severity
        self.message       = message
        self.person_id     = person_id
        self.timestamp     = datetime.now()
        self.active        = True

    def to_dict(self):
        return {
            "type":      self.behavior_type,
            "severity":  self.severity,
            "message":   self.message,
            "person_id": self.person_id,
            "timestamp": self.timestamp.isoformat(),
        }

    def __repr__(self):
        return f"[{self.severity}] {self.behavior_type}: {self.message}"


# =============================================================================
# PERSON STATE  – trạng thái riêng từng học sinh
# =============================================================================
class PersonState:
    """Theo dõi trạng thái của một học sinh (v3: adaptive, time-based)"""

    def __init__(self, person_id: int):
        self.person_id = person_id

        # --- Adaptive EAR calibration -----------------------------------
        self._ear_calibration_samples: list[float] = []
        self._calibration_start = time.time()
        self._calibrated        = False
        self.ear_threshold      = EAR_THRESH_DEFAULT  # sẽ được cập nhật sau calibrate
        self.ear_baseline       = 0.30                # EAR trung bình khi mở mắt

        # --- EAR (time-based) -------------------------------------------
        self.last_ear        = 0.30
        self.smooth_ear      = 0.30    # EMA cho display
        self.detect_ear      = 0.30    # EMA cho detection (phản hồi nhanh)
        self.total_blinks    = 0
        self.eyes_closed_since: float | None = None  # timestamp mắt bắt đầu nhắm

        # --- Blink rate analysis ----------------------------------------
        self.blink_timestamps: deque[float] = deque()  # timestamps của mỗi blink
        self.blink_rate      = 0.0     # blinks / phút hiện tại

        # --- MAR (time-based + variance) --------------------------------
        self.last_mar        = 0.0
        self.smooth_mar      = 0.0
        self.detect_mar      = 0.0
        self.yawn_count      = 0
        self.mouth_open_since: float | None = None    # timestamp miệng bắt đầu mở
        self.mar_history: deque[float] = deque(maxlen=30)  # MAR ~1s gần nhất
        self.yawn_timestamps: deque[float] = deque()  # thời điểm mỗi lần ngáp

        # --- Head pose (with calibration) ---------------------------------
        self.head_down_cnt   = 0
        self.head_turn_cnt   = 0
        self.pitch           = 0.0
        self.yaw             = 0.0
        self.roll            = 0.0
        self.smooth_pitch    = 0.0
        self.smooth_yaw      = 0.0
        # Auto-calibration: vài giây đầu đo pitch/yaw baseline (góc cam)
        self._pose_calibration_samples: list[tuple] = []  # (pitch, yaw)
        self._pose_calibrated = False
        self.pitch_baseline  = 0.0   # pitch "thẳng" (bù góc cam)
        self.yaw_baseline    = 0.0   # yaw "thẳng"

        # --- Gaze detection ---------------------------------------------
        self.gaze_off_since: float | None = None      # timestamp nhìn lệch
        self.gaze_ratio      = 0.0     # 0 = center, 1 = edge

        # --- Face bbox (for tracking) ------------------------------------
        self.bbox            = None

        # --- No-face -----------------------------------------------------
        self.last_face_seen  = time.time()
        self.face_visible    = False

        # --- Alerts ------------------------------------------------------
        self.active_alerts: list[str] = []
        self.alert_history: list[AlertEvent] = []

    def calibrate_ear(self, ear: float):
        """Thu thập EAR samples trong giai đoạn calibration."""
        if self._calibrated:
            return
        elapsed = time.time() - self._calibration_start
        if elapsed <= EAR_CALIBRATION_SECS:
            if ear > 0.15:  # chỉ lấy khi mắt mở (loại bỏ blink)
                self._ear_calibration_samples.append(ear)
        else:
            if len(self._ear_calibration_samples) >= 10:
                # Dùng median để loại outlier
                sorted_samples = sorted(self._ear_calibration_samples)
                n = len(sorted_samples)
                self.ear_baseline = sorted_samples[n // 2]
                self.ear_threshold = self.ear_baseline * EAR_ADAPTIVE_RATIO
                # Clamp để tránh giá trị vô lý
                self.ear_threshold = max(0.15, min(0.30, self.ear_threshold))
            self._calibrated = True

    def calibrate_pose(self, pitch: float, yaw: float):
        """Thu thập pitch/yaw trong giai đoạn calibration để bù góc camera.
        Gọi cùng lúc với calibrate_ear (dùng chung thời gian 4s đầu)."""
        if self._pose_calibrated:
            return
        elapsed = time.time() - self._calibration_start
        if elapsed <= EAR_CALIBRATION_SECS:
            # Chỉ lấy khi pitch/yaw hợp lý (loại outlier lúc đang xoay đầu)
            if abs(pitch) < 40 and abs(yaw) < 40:
                self._pose_calibration_samples.append((pitch, yaw))
        else:
            if len(self._pose_calibration_samples) >= 10:
                pitches = sorted([s[0] for s in self._pose_calibration_samples])
                yaws    = sorted([s[1] for s in self._pose_calibration_samples])
                n = len(pitches)
                self.pitch_baseline = pitches[n // 2]  # median
                self.yaw_baseline   = yaws[n // 2]
            self._pose_calibrated = True

    def update_smooth(self, ear: float, mar: float, pitch: float, yaw: float):
        """Cập nhật EMA kép: detect (nhanh) + display (mượt)"""
        ad = EMA_ALPHA_DETECT
        av = EMA_ALPHA_DISPLAY
        # Detection EMA (phản hồi nhanh)
        self.detect_ear   = ad * ear   + (1 - ad) * self.detect_ear
        self.detect_mar   = ad * mar   + (1 - ad) * self.detect_mar
        # Display EMA (mượt)
        self.smooth_ear   = av * ear   + (1 - av) * self.smooth_ear
        self.smooth_mar   = av * mar   + (1 - av) * self.smooth_mar
        self.smooth_pitch = av * pitch + (1 - av) * self.smooth_pitch
        self.smooth_yaw   = av * yaw   + (1 - av) * self.smooth_yaw
        # Raw values
        self.last_ear = ear
        self.last_mar = mar
        self.pitch    = pitch
        self.yaw      = yaw
        # MAR history cho variance check
        self.mar_history.append(mar)

    def update_blink_rate(self):
        """Tính blink rate trong sliding window."""
        now = time.time()
        # Xóa blink cũ hơn window
        while self.blink_timestamps and (now - self.blink_timestamps[0]) > BLINK_RATE_WINDOW_SECS:
            self.blink_timestamps.popleft()
        self.blink_rate = len(self.blink_timestamps)  # blinks / phút (window = 60s)

    def recent_yawn_count(self) -> int:
        """Đếm số lần ngáp trong COMBO_WINDOW_SECS gần nhất."""
        now = time.time()
        while self.yawn_timestamps and (now - self.yawn_timestamps[0]) > COMBO_WINDOW_SECS:
            self.yawn_timestamps.popleft()
        return len(self.yawn_timestamps)

    def mar_variance(self) -> float:
        """Variance của MAR trong ~1s gần nhất (phân biệt ngáp vs nói)."""
        if len(self.mar_history) < 5:
            return 0.0
        arr = np.array(self.mar_history)
        return float(np.var(arr))


# =============================================================================
# AUDIO MANAGER
# =============================================================================
class AudioManager:
    def __init__(self, sound_path: str = None):
        if sound_path is None:
            sound_path = os.path.join(_BASE_DIR, "alarm.wav")
        self.sound_path  = sound_path
        self._alarm_on   = False
        self._thread     = None
        self._stop_queue = queue.Queue()

    def play(self):
        if self._alarm_on:
            return
        if not os.path.exists(self.sound_path):
            return
        self._alarm_on = True
        self._thread = Thread(target=self._play_loop, daemon=True)
        self._thread.start()

    def stop(self):
        if self._alarm_on:
            self._alarm_on = False
            if AUDIO_LIB == 'pygame':
                try:
                    pygame.mixer.stop()
                except Exception:
                    pass
            self._stop_queue.put(True)

    def _play_loop(self):
        if AUDIO_LIB == 'pygame':
            try:
                sound = pygame.mixer.Sound(self.sound_path)
                while self._alarm_on:
                    if not self._stop_queue.empty():
                        self._stop_queue.get()
                        break
                    sound.play()
                    while pygame.mixer.get_busy() and self._alarm_on:
                        if not self._stop_queue.empty():
                            self._stop_queue.get()
                            pygame.mixer.stop()
                            return
                        pygame.time.Clock().tick(10)
            except Exception as e:
                print(f"[AUDIO] pygame error: {e}")
        elif AUDIO_LIB == 'playsound':
            while self._alarm_on:
                if not self._stop_queue.empty():
                    self._stop_queue.get()
                    break
                try:
                    _playsound.playsound(self.sound_path)
                except Exception as e:
                    print(f"[AUDIO] playsound error: {e}")
                    break


# =============================================================================
# YOLO DETECTOR
# =============================================================================
class YOLODetector:
    def __init__(self, model_name: str = None):
        if model_name is None:
            model_name = os.path.join(_BASE_DIR, "yolov8n.pt")
        self.model     = None
        self.available = YOLO_AVAILABLE
        if self.available:
            try:
                print(f"[YOLO] Đang tải model {model_name}...")
                # PyTorch 2.6+ mặc định weights_only=True → patch tạm
                import torch
                _original_load = torch.load
                def _patched_load(*args, **kwargs):
                    kwargs.setdefault("weights_only", False)
                    return _original_load(*args, **kwargs)
                torch.load = _patched_load
                try:
                    self.model = YOLO(model_name)
                finally:
                    torch.load = _original_load
                print("[YOLO] Model tải thành công")
            except Exception as e:
                print(f"[YOLO] Lỗi tải model: {e}")
                self.available = False

    def detect(self, frame: np.ndarray):
        """Trả về persons, phones, raw_results.
        Dùng confidence thấp hơn (YOLO_PHONE_CONF) rồi filter sau,
        để phone detection tốt hơn mà không ảnh hưởng person."""
        persons, phones = [], []
        if not self.available or self.model is None:
            return persons, phones, None
        # Chạy YOLO với conf thấp nhất (phone), filter person riêng
        min_conf = min(YOLO_PERSON_CONF, YOLO_PHONE_CONF)
        results = self.model(frame, conf=min_conf, verbose=False)
        for r in results:
            for box in r.boxes:
                cls  = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if cls == YOLO_PERSON_ID and conf >= YOLO_PERSON_CONF:
                    persons.append((x1, y1, x2, y2, conf))
                elif cls == YOLO_PHONE_ID and conf >= YOLO_PHONE_CONF:
                    phones.append((x1, y1, x2, y2, conf))
        return persons, phones, results

    def draw_detections(self, frame: np.ndarray,
                        persons: list, phones: list) -> np.ndarray:
        for (x1, y1, x2, y2, conf) in persons:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_BLUE, 2)
            cv2.putText(frame, f"Student {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, COLOR_BLUE, 1, cv2.LINE_AA)
        for (x1, y1, x2, y2, conf) in phones:
            cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_PURPLE, 2)
            cv2.putText(frame, f"PHONE {conf:.2f}",
                        (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, COLOR_PURPLE, 2, cv2.LINE_AA)
        return frame


# =============================================================================
# UTILITY: bbox IoU & overlap
# =============================================================================
def _bbox_iou(a, b):
    """Tính IoU giữa 2 bbox (x1,y1,x2,y2)"""
    ax1, ay1, ax2, ay2 = a[:4]
    bx1, by1, bx2, by2 = b[:4]
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-6)


def _bbox_overlap_ratio(small, big):
    """Tỉ lệ diện tích small nằm trong big"""
    sx1, sy1, sx2, sy2 = small[:4]
    bx1, by1, bx2, by2 = big[:4]
    ix1 = max(sx1, bx1)
    iy1 = max(sy1, by1)
    ix2 = min(sx2, bx2)
    iy2 = min(sy2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    area_small = max((sx2 - sx1) * (sy2 - sy1), 1)
    return inter / area_small


def _bbox_center(bbox):
    return ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)


def _center_distance(a, b):
    cx_a, cy_a = _bbox_center(a)
    cx_b, cy_b = _bbox_center(b)
    return math.hypot(cx_a - cx_b, cy_a - cy_b)


# =============================================================================
# FACIAL ANALYZER   (EAR, MAR, Head Pose)
# =============================================================================
class FacialAnalyzer:
    """Phân tích khuôn mặt: EAR, MAR, Head Pose, Gaze (v3)."""

    # 3D model points cho solvePnP – thứ tự:
    # Nose tip, Chin, Left eye corner, Right eye corner, Left mouth, Right mouth
    MODEL_POINTS = np.array([
        (  0.0,    0.0,    0.0),     # Nose tip
        (  0.0, -330.0,  -65.0),     # Chin
        (-225.0, 170.0, -135.0),     # Left eye corner
        ( 225.0, 170.0, -135.0),     # Right eye corner
        (-150.0,-150.0, -125.0),     # Left mouth corner
        ( 150.0,-150.0, -125.0),     # Right mouth corner
    ], dtype=np.float64)

    # MediaPipe iris landmark indices (chỉ có khi refine_landmarks=True)
    # Left iris: 468-472, Right iris: 473-477
    MP_LEFT_IRIS_CENTER  = 468
    MP_RIGHT_IRIS_CENTER = 473

    def __init__(self, use_dlib: bool = True):
        self.use_dlib       = use_dlib and DLIB_AVAILABLE
        self.use_mp         = MP_AVAILABLE
        self.dlib_detector  = None
        self.dlib_predictor = None
        self.face_mesh      = None
        # CLAHE preprocessor
        self._clahe = cv2.createCLAHE(
            clipLimit=CLAHE_CLIP_LIMIT,
            tileGridSize=CLAHE_GRID_SIZE
        )

        if self.use_dlib:
            try:
                self.dlib_detector = dlib.get_frontal_face_detector()
                dat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "shape_predictor_68_face_landmarks.dat")
                if os.path.exists(dat_path):
                    self.dlib_predictor = dlib.shape_predictor(dat_path)
                else:
                    print("[Dlib] shape_predictor_68_face_landmarks.dat không tìm thấy")
                    self.use_dlib = False
            except Exception as e:
                print(f"[Dlib] Lỗi: {e}")
                self.use_dlib = False

        if self.use_mp:
            self.face_mesh = mp_face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )

    def enhance_frame(self, frame: np.ndarray) -> np.ndarray:
        """Tiền xử lý ảnh bằng CLAHE trên L channel (LAB color space)."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = self._clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def enhance_gray(self, gray: np.ndarray) -> np.ndarray:
        """CLAHE trên ảnh grayscale (cho Dlib)."""
        return self._clahe.apply(gray)

    # ── EAR ──────────────────────────────────────────────────────────────
    @staticmethod
    def _ear(pts):
        """EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)"""
        A = dist.euclidean(pts[1], pts[5])
        B = dist.euclidean(pts[2], pts[4])
        C = dist.euclidean(pts[0], pts[3])
        return (A + B) / (2.0 * max(C, 1e-6))

    # ── MAR (Dlib) ───────────────────────────────────────────────────────
    @staticmethod
    def _mar_dlib(pts_48_to_67):
        """
        MAR từ Dlib 68 landmarks.
        Dùng 3 cặp dọc inner lip (61-67, 62-66, 63-65) / ngang (48-54).
        indices relative to pts[48]:
          inner top: idx 13(=61), 14(=62), 15(=63)
          inner bot: idx 19(=67), 18(=66), 17(=65)
          corners:   idx 0(=48), 6(=54)
        """
        pts = pts_48_to_67
        if len(pts) < 20:
            return 0.0
        v1 = dist.euclidean(pts[13], pts[19])  # 61-67
        v2 = dist.euclidean(pts[14], pts[18])  # 62-66
        v3 = dist.euclidean(pts[15], pts[17])  # 63-65
        h  = dist.euclidean(pts[0], pts[6])    # 48-54
        return (v1 + v2 + v3) / (3.0 * max(h, 1e-6))

    # ── MAR (MediaPipe) ─────────────────────────────────────────────────
    @staticmethod
    def _mar_mp(landmarks, w, h):
        """
        MAR từ MediaPipe: inner lip 13(top), 14(bottom), 78(left), 308(right).
        Thêm cặp phụ 82-87, 312-317 cho chính xác.
        """
        def pt(i):
            lm = landmarks[i]
            return np.array([lm.x * w, lm.y * h])

        v1 = np.linalg.norm(pt(13)  - pt(14))    # center
        v2 = np.linalg.norm(pt(82)  - pt(87))    # left inner
        v3 = np.linalg.norm(pt(312) - pt(317))   # right inner
        h_dist = np.linalg.norm(pt(78) - pt(308))
        return (v1 + v2 + v3) / (3.0 * max(h_dist, 1e-6))

    # ── Head Pose ────────────────────────────────────────────────────────
    def _solve_head_pose(self, image_pts_6, frame_shape):
        """
        Tính pitch, yaw, roll từ 6 điểm ảnh.
        Thứ tự: nose, chin, L_eye, R_eye, L_mouth, R_mouth
        """
        h, w = frame_shape[:2]
        focal = w * FOCAL_LENGTH_RATIO  # cải thiện ước lượng focal length
        cam_matrix = np.array([
            [focal, 0,     w / 2],
            [0,     focal, h / 2],
            [0,     0,     1    ]
        ], dtype=np.float64)
        dist_coeffs = np.zeros((4, 1))

        success, rvec, tvec = cv2.solvePnP(
            self.MODEL_POINTS,
            np.array(image_pts_6, dtype=np.float64),
            cam_matrix, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE
        )
        if not success:
            return 0.0, 0.0, 0.0

        rmat, _ = cv2.Rodrigues(rvec)
        euler, _, _, _, _, _ = cv2.RQDecomp3x3(rmat)
        return float(euler[0]), float(euler[1]), float(euler[2])

    # ── Gaze ratio (MediaPipe iris) ─────────────────────────────────────
    def _gaze_ratio_mp(self, lms, w, h):
        """
        Tính gaze ratio từ iris landmarks.
        Trả về 0.0 (nhìn giữa) → 1.0 (nhìn cực lệch).
        Dùng vị trí iris center so với eye corners.
        """
        if len(lms) < 478:  # cần refine_landmarks
            return 0.0

        def _pt(i):
            return np.array([lms[i].x * w, lms[i].y * h])

        # Left eye
        l_iris  = _pt(self.MP_LEFT_IRIS_CENTER)
        l_inner = _pt(133)   # left eye inner corner
        l_outer = _pt(33)    # left eye outer corner
        l_eye_w = max(np.linalg.norm(l_outer - l_inner), 1e-6)
        l_center = (l_inner + l_outer) / 2.0
        l_offset = np.linalg.norm(l_iris - l_center) / l_eye_w

        # Right eye
        r_iris  = _pt(self.MP_RIGHT_IRIS_CENTER)
        r_inner = _pt(362)   # right eye inner corner
        r_outer = _pt(263)   # right eye outer corner
        r_eye_w = max(np.linalg.norm(r_outer - r_inner), 1e-6)
        r_center = (r_inner + r_outer) / 2.0
        r_offset = np.linalg.norm(r_iris - r_center) / r_eye_w

        return float((l_offset + r_offset) / 2.0)

    # ── MediaPipe analyze ────────────────────────────────────────────────
    def analyze_mp(self, frame: np.ndarray):
        if not self.use_mp or self.face_mesh is None:
            return []

        # Tiền xử lý CLAHE trước khi đưa vào MediaPipe
        enhanced = self.enhance_frame(frame)
        rgb    = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)
        faces  = []

        if not result.multi_face_landmarks:
            return faces

        h, w = frame.shape[:2]
        for face_lms in result.multi_face_landmarks:
            lms = face_lms.landmark

            def _pts(indices):
                return [(int(lms[i].x * w), int(lms[i].y * h)) for i in indices]

            left_pts  = _pts(MP_LEFT_EYE)
            right_pts = _pts(MP_RIGHT_EYE)
            ear = (self._ear(left_pts) + self._ear(right_pts)) / 2.0

            mar = self._mar_mp(lms, w, h)

            # Head pose
            hp_indices = [4, 152, 33, 263, 57, 287]
            image_pts = [(lms[i].x * w, lms[i].y * h) for i in hp_indices]
            pitch, yaw, roll = self._solve_head_pose(image_pts, frame.shape)

            # Gaze ratio từ iris
            gaze_ratio = self._gaze_ratio_mp(lms, w, h)

            # Bbox from landmarks
            xs = [int(lms[i].x * w) for i in range(min(468, len(lms)))]
            ys = [int(lms[i].y * h) for i in range(min(468, len(lms)))]
            margin = 15
            bbox = (max(0, min(xs) - margin), max(0, min(ys) - margin),
                    min(w, max(xs) + margin), min(h, max(ys) + margin))

            mouth_pts = _pts(MP_MOUTH_OUTER[:20]) if len(MP_MOUTH_OUTER) >= 20 else []

            faces.append({
                'ear': ear, 'mar': mar,
                'pitch': pitch, 'yaw': yaw, 'roll': roll,
                'gaze_ratio': gaze_ratio,
                'landmarks': face_lms,
                'left_eye': left_pts, 'right_eye': right_pts,
                'mouth': mouth_pts, 'bbox': bbox,
            })
        return faces

    # ── Dlib analyze ─────────────────────────────────────────────────────
    def analyze_dlib(self, frame: np.ndarray):
        if not self.use_dlib:
            return []
        h, w  = frame.shape[:2]
        gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Tiền xử lý CLAHE cho Dlib (cải thiện ánh sáng yếu)
        gray  = self.enhance_gray(gray)
        rects = self.dlib_detector(gray, 1)
        faces = []

        for rect in rects:
            shape = self.dlib_predictor(gray, rect)
            pts   = [(shape.part(i).x, shape.part(i).y) for i in range(68)]

            left_eye  = pts[36:42]
            right_eye = pts[42:48]
            ear = (self._ear(left_eye) + self._ear(right_eye)) / 2.0

            mouth_all = pts[48:68]
            mar = self._mar_dlib(mouth_all)

            # Head pose
            image_pts = [
                pts[30], pts[8], pts[36], pts[45], pts[48], pts[54]
            ]
            pitch, yaw, roll = self._solve_head_pose(image_pts, frame.shape)

            bbox = (rect.left(), rect.top(), rect.right(), rect.bottom())

            faces.append({
                'ear': ear, 'mar': mar,
                'pitch': pitch, 'yaw': yaw, 'roll': roll,
                'gaze_ratio': 0.0,  # Dlib không có iris → gaze = 0
                'landmarks': pts,
                'left_eye': left_eye, 'right_eye': right_eye,
                'mouth': mouth_all, 'bbox': bbox,
            })
        return faces

    def analyze(self, frame: np.ndarray):
        """Ưu tiên MediaPipe, fallback Dlib"""
        faces = []
        if self.use_mp and mp_face_mesh is not None:
            faces = self.analyze_mp(frame)
        if not faces:
            faces = self.analyze_dlib(frame)
        return faces


# =============================================================================
# FACE TRACKER  – matching faces giữa các frame bằng IOU
# =============================================================================
class FaceTracker:
    """Gán person_id ổn định qua IOU matching + face embedding re-ID.
    v5: dùng dlib face_recognition_resnet_model để tạo embedding 128D,
    nhận diện lại cùng 1 học sinh khi rời rồi quay lại khung hình."""

    _face_rec_model = None   # class-level: load 1 lần duy nhất
    _face_rec_loaded = False

    @classmethod
    def _load_face_rec(cls):
        """Load dlib face recognition model (1 lần)."""
        if cls._face_rec_loaded:
            return
        cls._face_rec_loaded = True
        if not DLIB_AVAILABLE:
            return
        rec_path = os.path.join(_BASE_DIR, "models", "dlib_face_recognition_resnet_model_v1.dat")
        if os.path.exists(rec_path):
            try:
                cls._face_rec_model = dlib.face_recognition_model_v1(rec_path)
                print("[FaceTracker] Face recognition model đã tải — hỗ trợ nhận diện lại học sinh")
            except Exception as e:
                print(f"[FaceTracker] Không tải được face recognition model: {e}")
        else:
            print(f"[FaceTracker] Không tìm thấy {rec_path}")
            print("              Chạy: python download_models.py để tải")
            print("              Hệ thống vẫn hoạt động nhưng dùng vị trí để re-ID (kém chính xác hơn)")

    def __init__(self):
        self._load_face_rec()
        self._next_id = 0
        # Active: đang thấy hoặc mới mất vài frame (IoU matching)
        self._prev_faces: dict[int, tuple] = {}    # pid -> bbox
        self._grace_counter: dict[int, int] = {}   # pid -> frames since last seen
        # Embeddings cho mỗi person (cập nhật liên tục bằng EMA)
        self._embeddings: dict[int, np.ndarray] = {}  # pid -> 128D embedding
        # Pool re-ID: đã mất lâu hơn grace frames, giữ lại để re-identify
        self._reid_pool: dict[int, dict] = {}      # pid -> {bbox, embedding, last_seen}

    def _compute_embedding(self, frame: np.ndarray, bbox: tuple) -> np.ndarray | None:
        """Tính 128D face embedding từ frame + bbox. Trả None nếu không có model."""
        if self._face_rec_model is None or not DLIB_AVAILABLE:
            return None
        try:
            x1, y1, x2, y2 = [int(v) for v in bbox]
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            rect = dlib.rectangle(x1, y1, x2, y2)

            # Cần shape predictor để align face
            sp_path = os.path.join(_BASE_DIR, "models", "shape_predictor_68_face_landmarks.dat")
            if not hasattr(self, '_sp') or self._sp is None:
                if os.path.exists(sp_path):
                    self._sp = dlib.shape_predictor(sp_path)
                else:
                    return None

            shape = self._sp(frame, rect)
            embedding = np.array(self._face_rec_model.compute_face_descriptor(frame, shape))
            return embedding
        except Exception:
            return None

    @staticmethod
    def _embedding_dist(e1: np.ndarray, e2: np.ndarray) -> float:
        """Khoảng cách Euclidean giữa 2 embedding."""
        return float(np.linalg.norm(e1 - e2))

    def _try_reid(self, bbox, frame_w, embedding=None) -> int | None:
        """Thử match face mới với pool re-ID bằng embedding (ưu tiên) hoặc vị trí."""
        now = time.time()

        # Dọn hết hạn
        expired = [p for p, v in self._reid_pool.items()
                   if now - v['last_seen'] > FACE_TRACK_REID_SECS]
        for pid in expired:
            self._reid_pool.pop(pid, None)

        if not self._reid_pool:
            return None

        # --- Ưu tiên 1: So sánh embedding (chính xác nhất) ---
        if embedding is not None:
            best_pid = None
            best_dist = float('inf')
            for pid, info in self._reid_pool.items():
                pool_emb = info.get('embedding')
                if pool_emb is not None:
                    d = self._embedding_dist(embedding, pool_emb)
                    if d < best_dist:
                        best_dist = d
                        best_pid = pid
            if best_pid is not None and best_dist < FACE_TRACK_EMBED_THRESH:
                self._reid_pool.pop(best_pid, None)
                return best_pid

        # --- Fallback: Nếu chỉ có 1 person trong pool → gán luôn ---
        remaining = {p: v for p, v in self._reid_pool.items()
                     if now - v['last_seen'] <= FACE_TRACK_REID_SECS}
        if len(remaining) == 1:
            pid = next(iter(remaining))
            self._reid_pool.pop(pid, None)
            return pid

        # --- Fallback: So sánh vị trí center ---
        best_pid = None
        best_dist = float('inf')
        cx1 = (bbox[0] + bbox[2]) / 2.0
        cy1 = (bbox[1] + bbox[3]) / 2.0
        for pid, info in self._reid_pool.items():
            bb2 = info['bbox']
            cx2 = (bb2[0] + bb2[2]) / 2.0
            cy2 = (bb2[1] + bb2[3]) / 2.0
            d = math.sqrt((cx1 - cx2)**2 + (cy1 - cy2)**2) / max(frame_w, 1)
            if d < best_dist:
                best_dist = d
                best_pid = pid
        if best_pid is not None and best_dist < FACE_TRACK_REID_DIST:
            self._reid_pool.pop(best_pid, None)
            return best_pid

        return None

    def update(self, face_bboxes: list[tuple], frame_w: int = 640,
               frame: np.ndarray = None) -> list[int]:
        """Nhận list bbox + frame, trả về list person_id tương ứng.
        frame dùng để tính embedding (optional, nếu có face_rec model)."""
        now = time.time()

        if not face_bboxes:
            # Tăng grace counter, chuyển vào reid_pool nếu hết grace
            expired = []
            for pid in list(self._grace_counter.keys()):
                self._grace_counter[pid] += 1
                if self._grace_counter[pid] > FACE_TRACK_GRACE_FRAMES:
                    expired.append(pid)
            for pid in expired:
                bbox = self._prev_faces.pop(pid, None)
                self._grace_counter.pop(pid, None)
                if bbox is not None:
                    self._reid_pool[pid] = {
                        'bbox': bbox,
                        'embedding': self._embeddings.get(pid),
                        'last_seen': now,
                    }
            # Dọn reid_pool hết hạn
            for pid in [p for p, v in self._reid_pool.items()
                        if now - v['last_seen'] > FACE_TRACK_REID_SECS]:
                self._reid_pool.pop(pid, None)
            return []

        n = len(face_bboxes)

        # Tính embedding cho mỗi face (nếu có model + frame)
        cur_embeddings = [None] * n
        if frame is not None and self._face_rec_model is not None:
            for i, bbox in enumerate(face_bboxes):
                cur_embeddings[i] = self._compute_embedding(frame, bbox)

        if not self._prev_faces:
            # Không có active → thử re-ID từ pool
            ids = []
            new_prev = {}
            for i, bbox in enumerate(face_bboxes):
                reid_pid = self._try_reid(bbox, frame_w, cur_embeddings[i])
                if reid_pid is not None:
                    pid = reid_pid
                else:
                    pid = self._next_id
                    self._next_id += 1
                ids.append(pid)
                new_prev[pid] = bbox
                # Lưu / cập nhật embedding
                if cur_embeddings[i] is not None:
                    self._embeddings[pid] = cur_embeddings[i]
            self._prev_faces = new_prev
            self._grace_counter = {pid: 0 for pid in new_prev}
            return ids

        prev_ids    = list(self._prev_faces.keys())
        prev_bboxes = list(self._prev_faces.values())
        m = len(prev_ids)

        iou_matrix = np.zeros((n, m))
        for i, cur_bb in enumerate(face_bboxes):
            for j, prev_bb in enumerate(prev_bboxes):
                iou_matrix[i, j] = _bbox_iou(cur_bb, prev_bb)

        used_cur  = set()
        used_prev = set()
        assigned  = [None] * n

        while True:
            if len(used_cur) == n or len(used_prev) == m:
                break
            best_val = -1
            best_i, best_j = -1, -1
            for i in range(n):
                if i in used_cur:
                    continue
                for j in range(m):
                    if j in used_prev:
                        continue
                    if iou_matrix[i, j] > best_val:
                        best_val = iou_matrix[i, j]
                        best_i, best_j = i, j
            if best_val < FACE_TRACK_IOU_MIN:
                break
            assigned[best_i] = prev_ids[best_j]
            used_cur.add(best_i)
            used_prev.add(best_j)

        # Chưa match → thử re-ID bằng embedding trước khi tạo ID mới
        for i in range(n):
            if assigned[i] is None:
                reid_pid = self._try_reid(face_bboxes[i], frame_w, cur_embeddings[i])
                if reid_pid is not None:
                    assigned[i] = reid_pid
                else:
                    assigned[i] = self._next_id
                    self._next_id += 1

        # Cập nhật embeddings (EMA cho smooth)
        for i, pid in enumerate(assigned):
            if cur_embeddings[i] is not None:
                old = self._embeddings.get(pid)
                if old is not None:
                    # EMA: 80% cũ + 20% mới → ổn định
                    self._embeddings[pid] = 0.8 * old + 0.2 * cur_embeddings[i]
                else:
                    self._embeddings[pid] = cur_embeddings[i]

        # Cập nhật prev_faces + grace period
        new_prev = {}
        new_grace = {}
        matched_pids = set(assigned)

        for i, pid in enumerate(assigned):
            new_prev[pid] = face_bboxes[i]
            new_grace[pid] = 0

        for pid in prev_ids:
            if pid not in matched_pids:
                grace = self._grace_counter.get(pid, 0) + 1
                if grace <= FACE_TRACK_GRACE_FRAMES:
                    new_prev[pid] = self._prev_faces[pid]
                    new_grace[pid] = grace
                else:
                    self._reid_pool[pid] = {
                        'bbox': self._prev_faces[pid],
                        'embedding': self._embeddings.get(pid),
                        'last_seen': now,
                    }

        self._prev_faces = new_prev
        self._grace_counter = new_grace

        return assigned


# =============================================================================
# BEHAVIOR ANALYZER – logic cảnh báo (dùng smooth values)
# =============================================================================
class BehaviorAnalyzer:
    """v3: time-based, adaptive EAR, gaze, blink rate, yawn/talk separation."""

    def __init__(self):
        self.states: dict[int, PersonState] = {}

    def get_state(self, person_id: int) -> PersonState:
        if person_id not in self.states:
            self.states[person_id] = PersonState(person_id)
        return self.states[person_id]

    def analyze(self, person_id: int, face_data: dict,
                phones_for_person: list) -> list[AlertEvent]:
        """Phân tích hành vi cho 1 học sinh (v3)."""
        alerts = []
        now = time.time()
        st = self.get_state(person_id)
        st.face_visible   = True
        st.last_face_seen = now

        ear   = face_data.get('ear', 0.3)
        mar   = face_data.get('mar', 0.0)
        pitch = face_data.get('pitch', 0.0)
        yaw   = face_data.get('yaw', 0.0)
        gaze  = face_data.get('gaze_ratio', 0.0)
        bbox  = face_data.get('bbox')

        # Adaptive calibration (4s đầu tiên): EAR + Head Pose
        st.calibrate_ear(ear)
        st.calibrate_pose(pitch, yaw)

        st.update_smooth(ear, mar, pitch, yaw)
        if bbox:
            st.bbox = bbox

        # Dùng detect EMA (phản hồi nhanh) cho logic cảnh báo
        d_ear   = st.detect_ear
        d_mar   = st.detect_mar
        # Head pose: trừ baseline để bù góc camera
        s_pitch = st.smooth_pitch - st.pitch_baseline
        s_yaw   = st.smooth_yaw  - st.yaw_baseline

        # ── 1. EAR: Blink / Drowsy / Microsleep (TIME-BASED) ────────────
        ear_thresh = st.ear_threshold  # adaptive per person

        if d_ear < ear_thresh:
            # Mắt đang nhắm
            if st.eyes_closed_since is None:
                st.eyes_closed_since = now
            closed_duration = now - st.eyes_closed_since

            if closed_duration >= MICROSLEEP_TIME_SECS:
                alerts.append(AlertEvent("MICROSLEEP", SEVERITY_CRITICAL,
                    f"Mắt nhắm {closed_duration:.1f}s (EAR={d_ear:.3f})",
                    person_id))
            elif closed_duration >= DROWSY_TIME_SECS:
                alerts.append(AlertEvent("DROWSY", SEVERITY_HIGH,
                    f"Mắt nhắm {closed_duration:.1f}s (EAR={d_ear:.3f})",
                    person_id))
        else:
            # Mắt mở — kiểm tra blink
            if st.eyes_closed_since is not None:
                closed_dur = now - st.eyes_closed_since
                # Blink hợp lệ: 0.05s - 0.4s
                if 0.05 <= closed_dur < DROWSY_TIME_SECS:
                    st.total_blinks += 1
                    st.blink_timestamps.append(now)
                st.eyes_closed_since = None

        # Cập nhật blink rate
        st.update_blink_rate()

        # ── 1b. Blink Rate Analysis ──────────────────────────────────────
        # Chỉ đánh giá khi window đã đủ 30s để tránh false positive đầu session
        blink_window_ready = (
            st._calibrated
            and len(st.blink_timestamps) >= 3
            and (time.time() - st.blink_timestamps[0]) >= 30.0
        )
        if blink_window_ready:
            if st.blink_rate < BLINK_RATE_LOW:
                alerts.append(AlertEvent("DROWSY", SEVERITY_MEDIUM,
                    f"Nhịp nháy mắt thấp ({st.blink_rate}/phút)",
                    person_id))
            elif st.blink_rate > BLINK_RATE_HIGH:
                alerts.append(AlertEvent("FATIGUE_HIGH", SEVERITY_MEDIUM,
                    f"Nhịp nháy mắt cao ({st.blink_rate}/phút)",
                    person_id))

        # ── 2. MAR: Yawning (time-based + variance filter) ──────────────
        if d_mar > MAR_THRESH:
            if st.mouth_open_since is None:
                st.mouth_open_since = now
            open_duration = now - st.mouth_open_since
            mar_var = st.mar_variance()

            # Phân biệt ngáp vs nói:
            # - Ngáp: MAR ổn định (variance thấp), kéo dài 1.5-6s
            # - Nói:  MAR dao động (variance cao)
            is_talking = mar_var > MAR_TALK_VARIANCE

            if (not is_talking
                    and MAR_YAWN_MIN_SECS <= open_duration <= MAR_YAWN_MAX_SECS):
                alerts.append(AlertEvent("YAWNING", SEVERITY_MEDIUM,
                    f"Ngáp {open_duration:.1f}s (MAR={d_mar:.3f})",
                    person_id))
        else:
            # Miệng đóng — ghi nhận ngáp nếu đủ dài
            if st.mouth_open_since is not None:
                dur = now - st.mouth_open_since
                mar_var = st.mar_variance()
                if (MAR_YAWN_MIN_SECS <= dur <= MAR_YAWN_MAX_SECS
                        and mar_var <= MAR_TALK_VARIANCE):
                    st.yawn_count += 1
                    st.yawn_timestamps.append(now)
                st.mouth_open_since = None

        # ── 3. Head down (+ combo với EAR) ───────────────────────────────
        if s_pitch > PITCH_DOWN_THRESH:
            st.head_down_cnt += 1
        else:
            st.head_down_cnt = max(0, st.head_down_cnt - 2)

        if st.head_down_cnt >= HEAD_POSE_FRAMES:
            # Combo: cúi đầu + mắt nhắm → ngủ gục (HIGH)
            if d_ear < ear_thresh:
                alerts.append(AlertEvent("HEAD_DOWN", SEVERITY_HIGH,
                    f"Ngủ gục: cúi đầu + mắt nhắm (pitch={s_pitch:.1f}°)",
                    person_id))
            else:
                # Cúi đầu alone → có thể viết bài (MEDIUM)
                alerts.append(AlertEvent("HEAD_DOWN", SEVERITY_MEDIUM,
                    f"Cúi đầu (pitch={s_pitch:.1f}°)", person_id))

        # ── 4. Head turn ─────────────────────────────────────────────────
        if abs(s_yaw) > YAW_THRESH:
            st.head_turn_cnt += 1
        else:
            st.head_turn_cnt = max(0, st.head_turn_cnt - 2)

        if st.head_turn_cnt >= HEAD_POSE_FRAMES:
            direction = "trái" if s_yaw < 0 else "phải"
            alerts.append(AlertEvent("HEAD_TURN", SEVERITY_MEDIUM,
                f"Quay đầu {direction} (yaw={s_yaw:.1f}°)", person_id))

        # ── 5. Gaze detection (iris-based) ───────────────────────────────
        st.gaze_ratio = gaze
        if gaze > GAZE_OFF_CENTER_THRESH:
            if st.gaze_off_since is None:
                st.gaze_off_since = now
            gaze_duration = now - st.gaze_off_since
            if gaze_duration >= GAZE_CONSEC_SECS:
                alerts.append(AlertEvent("DISTRACTED", SEVERITY_MEDIUM,
                    f"Mất tập trung {gaze_duration:.1f}s (gaze={gaze:.2f})",
                    person_id))
        else:
            st.gaze_off_since = None

        # ── 6. Phone usage ───────────────────────────────────────────────
        if len(phones_for_person) > 0:
            alerts.append(AlertEvent("PHONE_USE", SEVERITY_HIGH,
                f"Dùng điện thoại ({len(phones_for_person)} phone gần)",
                person_id))

        # ── 7. Combo: Yawn (sliding window) + drowsy = CRITICAL ─────────
        recent_yawns = st.recent_yawn_count()
        eyes_closing = st.eyes_closed_since is not None
        if recent_yawns >= COMBO_YAWN_THRESHOLD and eyes_closing:
            alerts.append(AlertEvent("FATIGUE_HIGH", SEVERITY_CRITICAL,
                f"Mệt mỏi: ngáp {recent_yawns} lần (5 phút) + mắt nhắm",
                person_id))

        return alerts

    def check_no_face(self, visible_ids: set) -> list[AlertEvent]:
        alerts = []
        for pid, st in self.states.items():
            if pid not in visible_ids:
                st.face_visible = False
                elapsed = time.time() - st.last_face_seen
                if elapsed > NO_FACE_SECONDS:
                    alerts.append(AlertEvent("NO_FACE", SEVERITY_MEDIUM,
                        f"Không thấy mặt {elapsed:.1f}s", pid))
        return alerts


# =============================================================================
# OVERLAY RENDERER
# =============================================================================
class OverlayRenderer:
    SEVERITY_COLORS = {
        SEVERITY_LOW:      COLOR_GREEN,
        SEVERITY_MEDIUM:   COLOR_YELLOW,
        SEVERITY_HIGH:     COLOR_ORANGE,
        SEVERITY_CRITICAL: COLOR_RED,
    }

    VN_LABELS = {
        "DROWSY":       "Ngủ gật",
        "MICROSLEEP":   "Ngủ sâu",
        "YAWNING":      "Ngáp",
        "HEAD_DOWN":    "Cúi đầu",
        "HEAD_TURN":    "Quay đầu",
        "PHONE_USE":    "Điện thoại",
        "DISTRACTED":   "Mất tập trung",
        "NO_FACE":      "Không thấy mặt",
        "MULTI_PERSON": "Nhiều người",
        "FATIGUE_HIGH": "Mệt mỏi nghiêm trọng",
        "NORMAL":       "Bình thường",
    }

    def draw_metrics_panel(self, frame, states, alerts):
        h, w = frame.shape[:2]
        panel_w = 320
        panel_h = min(40 + len(states) * 80 + 30, h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (panel_w, panel_h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        y = 20
        put_vn_text(frame, "GIÁM SÁT HỌC SINH",
                    (10, y - 14), font_size=16, color=(255, 200, 0))
        y += 8
        cv2.line(frame, (10, y), (panel_w - 10, y), COLOR_CYAN, 1)
        y += 18

        for pid, st in states.items():
            if not st.face_visible and (time.time() - st.last_face_seen > 10):
                continue

            face_color = COLOR_GREEN if st.face_visible else COLOR_RED
            pil_clr = (face_color[2], face_color[1], face_color[0])
            put_vn_text(frame, f"Học sinh #{pid + 1}",
                        (10, y - 12), font_size=14, color=pil_clr)
            y += 16

            ear_val = max(0.0, min(1.0, st.smooth_ear / 0.4))
            bar_w   = int(ear_val * 120)
            ear_clr = COLOR_GREEN if st.smooth_ear > st.ear_threshold else COLOR_RED
            cv2.rectangle(frame, (10, y), (130, y + 8), (60, 60, 60), -1)
            cv2.rectangle(frame, (10, y), (10 + bar_w, y + 8), ear_clr, -1)
            # Hiển thị cả threshold adaptive
            thresh_txt = f"EAR:{st.smooth_ear:.2f}/{st.ear_threshold:.2f}"
            if not st._calibrated:
                thresh_txt += " [cal]"
            cv2.putText(frame, thresh_txt,
                        (135, y + 8), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        COLOR_WHITE, 1, cv2.LINE_AA)
            y += 14

            cv2.putText(frame,
                        f"MAR:{st.smooth_mar:.2f}  P:{st.smooth_pitch:.0f} Y:{st.smooth_yaw:.0f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        COLOR_WHITE, 1, cv2.LINE_AA)
            y += 14

            cv2.putText(frame,
                        f"Blinks:{st.total_blinks}({st.blink_rate:.0f}/m)"
                        f"  Yawns:{st.yawn_count}  Gz:{st.gaze_ratio:.2f}",
                        (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35,
                        COLOR_WHITE, 1, cv2.LINE_AA)
            y += 18

        return frame

    def draw_alert_banner(self, frame, alerts):
        if not alerts:
            return frame
        h, w = frame.shape[:2]

        severity_order = {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 1,
                          SEVERITY_HIGH: 2, SEVERITY_CRITICAL: 3}
        top_alert = max(alerts, key=lambda a: severity_order.get(a.severity, 0))
        color     = self.SEVERITY_COLORS.get(top_alert.severity, COLOR_RED)

        if top_alert.severity == SEVERITY_CRITICAL:
            if int(time.time() * 4) % 2 == 0:
                color = COLOR_RED
            else:
                color = COLOR_ORANGE

        banner_h = 60
        overlay  = frame.copy()
        cv2.rectangle(overlay, (0, h - banner_h), (w, h), (10, 10, 10), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        vn_label   = self.VN_LABELS.get(top_alert.behavior_type, top_alert.behavior_type)
        person_txt = f"HS #{top_alert.person_id + 1}" if top_alert.person_id >= 0 else ""
        banner_text = f"[!] {person_txt} {vn_label}: {top_alert.message}"
        pil_color = (color[2], color[1], color[0])
        put_vn_text(frame, banner_text, (10, h - banner_h + 8),
                    font_size=20, color=pil_color)

        badge_txt = top_alert.severity
        (tw, _), _ = cv2.getTextSize(badge_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (w - tw - 20, h - banner_h + 5),
                      (w - 5, h - banner_h + 28), color, -1)
        cv2.putText(frame, badge_txt,
                    (w - tw - 12, h - banner_h + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 1, cv2.LINE_AA)

        if len(alerts) > 1:
            others = []
            for a in alerts:
                if a is not top_alert:
                    pid_s = f"HS#{a.person_id + 1}" if a.person_id >= 0 else ""
                    vn = self.VN_LABELS.get(a.behavior_type, a.behavior_type)
                    others.append(f"{pid_s} {vn}")
            if others:
                line2 = "  |  ".join(others[:3])
                put_vn_text(frame, line2, (10, h - banner_h + 34),
                            font_size=15, color=(200, 200, 200))

        cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)
        return frame

    def draw_face_landmarks(self, frame, face_data, person_id=-1, alerts=None):
        for pt in face_data.get('left_eye', []):
            cv2.circle(frame, pt, 2, COLOR_GREEN, -1, cv2.LINE_AA)
        for pt in face_data.get('right_eye', []):
            cv2.circle(frame, pt, 2, COLOR_GREEN, -1, cv2.LINE_AA)
        for pt in face_data.get('mouth', []):
            cv2.circle(frame, pt, 1, COLOR_YELLOW, -1, cv2.LINE_AA)

        bbox = face_data.get('bbox')
        label_x, label_y = 10, 30
        if bbox:
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                          COLOR_CYAN, 1)
            label_x = bbox[0]
            label_y = bbox[1] - 8
        else:
            eyes = face_data.get('left_eye', []) + face_data.get('right_eye', [])
            if eyes:
                min_y = min(p[1] for p in eyes)
                avg_x = sum(p[0] for p in eyes) // max(len(eyes), 1)
                label_x = avg_x - 30
                label_y = min_y - 15

        if person_id >= 0:
            label = f"HS #{person_id + 1}"
            color = COLOR_GREEN
            if alerts:
                sev_ord = {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 1,
                           SEVERITY_HIGH: 2, SEVERITY_CRITICAL: 3}
                worst = max(alerts, key=lambda a: sev_ord.get(a.severity, 0))
                color = self.SEVERITY_COLORS.get(worst.severity, COLOR_RED)
                vn = self.VN_LABELS.get(worst.behavior_type, worst.behavior_type)
                label = f"HS #{person_id + 1} - {vn}"

            pil_color = (color[2], color[1], color[0])
            put_vn_text(frame, label, (max(0, label_x), max(0, label_y)),
                        font_size=16, color=pil_color, bg_color=(10, 10, 10))

        return frame

    def draw_multi_person_warning(self, frame, n_persons):
        if n_persons > 1:
            h, w = frame.shape[:2]
            put_vn_text(frame, f"Phát hiện: {n_persons} học sinh",
                        (w // 2 - 120, 10), font_size=20, color=(255, 200, 0))
        return frame

    def draw_status_icon(self, frame, alerts):
        h, w = frame.shape[:2]
        cx, cy, r = w - 40, 40, 25
        if not alerts:
            cv2.circle(frame, (cx, cy), r, COLOR_GREEN, -1)
            cv2.putText(frame, "OK", (cx - 12, cy + 6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BLACK, 2)
        else:
            sev_ord = {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 1,
                       SEVERITY_HIGH: 2, SEVERITY_CRITICAL: 3}
            top = max(alerts, key=lambda a: sev_ord.get(a.severity, 0))
            color = self.SEVERITY_COLORS.get(top.severity, COLOR_RED)
            cv2.circle(frame, (cx, cy), r, color, -1)
            cv2.putText(frame, "!", (cx - 4, cy + 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, COLOR_BLACK, 3)
        return frame


# =============================================================================
# MAIN DETECTOR  (v2)
# =============================================================================
class BehaviorDetector:
    """
    v2 changes:
      - Analyze face on CLEAN frame (before YOLO drawing)
      - Face tracking via FaceTracker (IOU)
      - Phone assigned to nearest person only
      - EMA smoothing on EAR/MAR/Pose
      - Separate blink counting from drowsy detection
      - display_alerts (realtime) vs log_alerts (cooldown)
    """

    def __init__(self,
                 camera_id: int = 0,
                 sound_path: str = None,
                 use_yolo: bool = True,
                 yolo_model: str = None,
                 show_landmarks: bool = True,
                 show_metrics: bool = True,
                 record_output: bool = False,
                 output_path: str = "output_behavior.avi"):
        if sound_path is None:
            sound_path = os.path.join(_BASE_DIR, "alarm.wav")
        if yolo_model is None:
            yolo_model = os.path.join(_BASE_DIR, "yolov8n.pt")

        print("=" * 65)
        print("  BEHAVIOR DETECTOR v2 – Hệ thống giám sát tập trung học sinh")
        print("=" * 65)

        self.camera_id      = camera_id
        self.show_landmarks = show_landmarks
        self.show_metrics   = show_metrics
        self.record_output  = record_output
        self.output_path    = output_path

        self.yolo     = YOLODetector(yolo_model) if use_yolo else None
        self.analyzer = FacialAnalyzer()
        self.behavior = BehaviorAnalyzer()
        self.tracker  = FaceTracker()
        self.renderer = OverlayRenderer()
        self.audio    = AudioManager(sound_path)

        self.session_start  = datetime.now()
        self.frame_count    = 0
        self.all_alerts     = []

        self._last_alert_time: dict[str, float] = {}
        self._alert_cooldown = 8.0
        self._alarm_clear_counter = 0          # hysteresis: frames liên tiếp không có high alert
        self._ALARM_CLEAR_FRAMES  = 10         # cần 10 frame sạch liên tiếp mới dừng alarm

        print(f"  YOLO      : {'OK YOLOv8' if (self.yolo and self.yolo.available) else 'OFF'}")
        print(f"  MediaPipe : {'OK FaceMesh' if MP_AVAILABLE else 'OFF'}")
        print(f"  Dlib      : {'OK Active' if DLIB_AVAILABLE else 'OFF'}")
        print(f"  Audio     : {'OK ' + AUDIO_LIB if AUDIO_LIB else 'OFF'}")
        print("=" * 65)

    def _should_log_alert(self, alert_type: str) -> bool:
        now = time.time()
        last = self._last_alert_time.get(alert_type, 0)
        if now - last > self._alert_cooldown:
            self._last_alert_time[alert_type] = now
            return True
        return False

    @staticmethod
    def _assign_phones_to_persons(persons, phones):
        """Gán phone cho person gần nhất / overlap lớn nhất."""
        result = {}
        if not persons or not phones:
            return result
        for phone in phones:
            best_idx = -1
            best_overlap = 0.0
            for i, person in enumerate(persons):
                overlap = _bbox_overlap_ratio(phone, person)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = i
            if best_idx >= 0 and best_overlap > 0.15:
                result.setdefault(best_idx, []).append(phone)
            else:
                # Fallback: overlap thấp hoặc = 0 → dùng khoảng cách tâm
                dists = [_center_distance(phone, p) for p in persons]
                nearest = int(np.argmin(dists))
                person_h = persons[nearest][3] - persons[nearest][1]
                if dists[nearest] < person_h * 1.5:
                    result.setdefault(nearest, []).append(phone)
        return result

    def process_frame(self, frame: np.ndarray) -> dict:
        self.frame_count += 1

        # 1. YOLO on CLEAN frame
        persons, phones = [], []
        if self.yolo and self.yolo.available:
            persons, phones, _ = self.yolo.detect(frame)

        # 2. Face analysis on CLEAN frame
        faces = self.analyzer.analyze(frame)

        # 3. Face tracking
        face_bboxes = [fd.get('bbox', (0, 0, 1, 1)) for fd in faces]
        person_ids = self.tracker.update(face_bboxes, frame_w=frame.shape[1], frame=frame)

        # 4. Phone → person assignment
        # Dùng overlap_ratio (face nằm trong person) thay vì IoU vì face << person
        face_to_yolo = {}
        if persons:
            for fi, fb in enumerate(face_bboxes):
                best_score, best_pi = 0.0, -1
                for pi, pb in enumerate(persons):
                    score = _bbox_overlap_ratio(fb, pb)
                    if score > best_score:
                        best_score = score
                        best_pi = pi
                if best_pi >= 0 and best_score > 0.2:
                    face_to_yolo[fi] = best_pi

        phone_map = self._assign_phones_to_persons(persons, phones)

        # 5. NOW draw YOLO (after analysis)
        if self.yolo and self.yolo.available:
            frame = self.yolo.draw_detections(frame, persons, phones)

        # Dùng faces (facial detections) để nhận biết nhiều người — hoạt động cả khi tắt YOLO
        n_detected = max(len(persons), len(faces))
        if n_detected > 1:
            frame = self.renderer.draw_multi_person_warning(frame, n_detected)

        # 6. Behavior + overlay per face
        display_alerts = []
        log_alerts     = []
        raw_high       = []
        visible_pids   = set()

        for fi, (face_data, pid) in enumerate(zip(faces, person_ids)):
            visible_pids.add(pid)
            yolo_pi = face_to_yolo.get(fi, -1)
            phones_this = phone_map.get(yolo_pi, [])

            beh_alerts = self.behavior.analyze(pid, face_data, phones_this)

            if self.show_landmarks:
                frame = self.renderer.draw_face_landmarks(
                    frame, face_data, person_id=pid, alerts=beh_alerts)

            for alert in beh_alerts:
                display_alerts.append(alert)
                if alert.severity in (SEVERITY_HIGH, SEVERITY_CRITICAL):
                    raw_high.append(alert)
                if self._should_log_alert(f"{pid}_{alert.behavior_type}"):
                    log_alerts.append(alert)
                    self.all_alerts.append(alert)

        # Multi-person — dùng n_detected để hoạt động cả khi tắt YOLO
        if n_detected > 1:
            mp_alert = AlertEvent("MULTI_PERSON", SEVERITY_MEDIUM,
                                  f"{n_detected} học sinh trong khung hình", -1)
            display_alerts.append(mp_alert)
            if self._should_log_alert("MULTI_PERSON"):
                log_alerts.append(mp_alert)

        # No-face
        nf_alerts = self.behavior.check_no_face(visible_pids)
        for alert in nf_alerts:
            display_alerts.append(alert)
            if self._should_log_alert(f"{alert.person_id}_NO_FACE"):
                log_alerts.append(alert)

        # 7. Audio (hysteresis: tránh flutter khi alert không ổn định)
        if raw_high:
            self._alarm_clear_counter = 0
            self.audio.play()
        else:
            self._alarm_clear_counter += 1
            if self._alarm_clear_counter >= self._ALARM_CLEAR_FRAMES:
                self.audio.stop()

        # 8. Overlay
        if self.show_metrics:
            frame = self.renderer.draw_metrics_panel(
                frame, self.behavior.states, display_alerts)

        frame = self.renderer.draw_alert_banner(frame, display_alerts)
        frame = self.renderer.draw_status_icon(frame, display_alerts)

        h, w = frame.shape[:2]
        cv2.putText(frame, f"Faces: {len(faces)}  |  F:{self.frame_count}",
                    (10, h - 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

        return {
            'frame':          frame,
            'alerts':         log_alerts,
            'display_alerts': display_alerts,
            'persons':        len(persons),
            'phones':         len(phones),
            'faces':          len(faces),
        }

    def run(self):
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened():
            print(f"[ERROR] Không mở được camera {self.camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)

        vid_writer = None
        if self.record_output:
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            vid_writer = cv2.VideoWriter(
                self.output_path, fourcc, 15, (actual_w, actual_h))

        print("\n  Đang chạy — 'q' thoát, 'r' reset, 's' screenshot\n")

        fps_t    = time.time()
        fps_hist = deque(maxlen=30)

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                continue

            result = self.process_frame(frame)
            out    = result['frame']

            now_t = time.time()
            fps_hist.append(now_t - fps_t)
            fps_t = now_t
            if fps_hist:
                avg_dt = sum(fps_hist) / len(fps_hist)
                fps = 1.0 / max(avg_dt, 1e-6)
            else:
                fps = 0
            cv2.putText(out, f"FPS: {fps:.1f}",
                        (out.shape[1] - 100, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

            cv2.imshow("Classroom Attention Monitor", out)

            if vid_writer:
                vid_writer.write(out)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('r'):
                self.behavior.states.clear()
                self.tracker = FaceTracker()
                self._last_alert_time.clear()
                print("[RESET]")
            elif key == ord('s'):
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                fname = f"screenshot_{ts}.jpg"
                cv2.imwrite(fname, out)
                print(f"[SCREENSHOT] {fname}")

        cap.release()
        if vid_writer:
            vid_writer.release()
        cv2.destroyAllWindows()
        self.audio.stop()
        self._print_session_summary()

    def _print_session_summary(self):
        duration = (datetime.now() - self.session_start).total_seconds() / 60
        print("\n" + "=" * 65)
        print("  SESSION SUMMARY")
        print("=" * 65)
        print(f"  Thời gian    : {duration:.2f} phút")
        print(f"  Tổng frames  : {self.frame_count}")
        from collections import Counter
        type_counts = Counter(a.behavior_type for a in self.all_alerts)
        if type_counts:
            print("  Cảnh báo:")
            for btype, count in type_counts.most_common():
                print(f"    {btype:20s}: {count}")
        else:
            print("  Không có cảnh báo")
        print("=" * 65)


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Behavior Detector v2 – Giám sát tập trung học sinh")
    parser.add_argument("--camera",       type=int,  default=0)
    parser.add_argument("--no-yolo",      action="store_true")
    parser.add_argument("--yolo-model",   type=str,  default=os.path.join(_BASE_DIR, "yolov8n.pt"))
    parser.add_argument("--record",       action="store_true")
    parser.add_argument("--no-landmarks", action="store_true")
    parser.add_argument("--sound",        type=str,  default=os.path.join(_BASE_DIR, "alarm.wav"))
    args = parser.parse_args()

    detector = BehaviorDetector(
        camera_id      = args.camera,
        sound_path     = args.sound,
        use_yolo       = not args.no_yolo,
        yolo_model     = args.yolo_model,
        show_landmarks = not args.no_landmarks,
        record_output  = args.record,
    )
    detector.run()
