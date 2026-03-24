# Hệ Thống Giám Sát Tập Trung Học Sinh

Phát hiện hành vi học sinh theo thời gian thực qua camera: ngủ gật, microsleep, ngáp, cúi đầu, quay đầu, sử dụng điện thoại, nhiều người trong khung hình.

---

## Công Nghệ Sử Dụng

### Computer Vision & AI

| Thư viện / Mô hình | Phiên bản | Chức năng |
|--------------------|-----------|-----------|
| OpenCV | >= 4.8 | Đọc camera, xử lý khung hình, vẽ overlay |
| MediaPipe FaceMesh | >= 0.10.14 | EAR, MAR, Head Pose, Gaze/Iris — **dùng chính** |
| Dlib shape_predictor_68 | 19.24+ | Fallback khi MediaPipe không cài được |
| YOLOv8 Nano | >= 8.0 | Phát hiện điện thoại, nhiều người |
| ResNet50 (Transfer Learning) | PyTorch | Phân loại mắt mở/nhắm — 95.43% accuracy |
| Haar Cascade | OpenCV | Phát hiện khuôn mặt cơ bản |

### Giao Diện & Âm Thanh

| Thư viện | Chức năng |
|----------|-----------|
| Tkinter | GUI desktop launcher |
| PIL / Pillow | Render chữ tiếng Việt lên frame |
| pygame | Âm thanh cảnh báo (macOS / Linux) |
| playsound | Âm thanh cảnh báo (Windows) |

---

## Yêu Cầu Hệ Thống

- Python **3.9 – 3.12** (khuyên dùng 3.12 — PyTorch chưa hỗ trợ 3.14)
- RAM tối thiểu 4GB (khuyên 8GB khi chạy YOLO)
- Webcam hoặc camera ngoài

---

## Cài Đặt

### macOS

```bash
# Bước 1 — Kiểm tra Python 3.12
python3.12 --version
# Nếu chưa có:
brew install python@3.12

# Bước 2 — Tạo môi trường ảo
cd /đường-dẫn-đến-dự-án
python3.12 -m venv venv
source venv/bin/activate

# Bước 3 — Cài thư viện
pip install -r requirements.txt

# Bước 4 — Cài PyTorch (cho ResNet AI)
pip install torch torchvision

# Bước 5 — Tải model Dlib (fallback, không bắt buộc)
python3.12 download_models.py

# Bước 6 — Train ResNet model (chỉ làm 1 lần, ~15–25 phút)
python3.12 train_resnet.py
```

> **Lưu ý macOS:** Nếu gặp lỗi `externally-managed-environment` khi cài ngoài venv:
> ```bash
> pip install -r requirements.txt --break-system-packages
> ```

---

### Windows

```bat
:: Bước 1 — Kiểm tra Python 3.12
python --version
:: Nếu chưa có: tải tại python.org, chọn Python 3.12

:: Bước 2 — Tạo môi trường ảo
cd đường-dẫn-đến-dự-án
python -m venv venv
venv\Scripts\activate

:: Bước 3 — Cài thư viện
pip install -r requirements.txt

:: Bước 4 — Cài PyTorch (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

:: Bước 5 — Tải model Dlib (fallback, không bắt buộc)
python download_models.py

:: Bước 6 — Train ResNet model (chỉ làm 1 lần)
python train_resnet.py
```

> **Lưu ý Windows — Cài Dlib (nếu cần fallback):**
>
> Cách 1 — Dùng file wheel prebuilt (dễ nhất):
> ```bat
> pip install dlib --find-links https://github.com/z-mahmud22/Dlib_Windows_Python3.x/releases
> ```
>
> Cách 2 — Cài Visual Studio Build Tools rồi compile:
> ```bat
> pip install cmake
> pip install dlib
> ```

---

## Chạy Ứng Dụng

### macOS

```bash
source venv/bin/activate
python3.12 main.py
```

### Windows

```bat
venv\Scripts\activate
python main.py
```

---

## Các Chức Năng

### 1. Phát hiện khuôn mặt — `face-try.py`

**Thuật toán:** Viola-Jones (Haar Cascade)
- Bộ lọc Haar + AdaBoost + Integral Image → phân loại nhanh O(1)
- Sliding window quét nhiều tỉ lệ (scale pyramid)

**Đầu ra:** Khung xanh quanh mặt, `Phat hien N khuon mat` (xanh) / `Khong phat hien` (đỏ)

---

### 2. Phát hiện ngủ gật — `blinkDetect.py`

**Thuật toán:** EAR (Eye Aspect Ratio) qua MediaPipe FaceMesh
```
EAR = (||p1-p5|| + ||p2-p4||) / (2 × ||p0-p3||)
```
- EAR < 0.22 → mắt nhắm
- Nhắm mắt > 1.5 giây → cảnh báo + âm thanh

**Đầu ra:**
- Trạng thái mắt: `MAT: MO` (xanh) / `MAT: NHAM` (đỏ)
- `EAR: 0.xx` | `Blinks: N` | `Mo: XX.X%  Nham: XX.X%`

---

### 3. Phát hiện hành vi (YOLO + AI) — `behavior_detector.py`

**Thuật toán tổng hợp:**

| Hành vi | Phương pháp | Ngưỡng |
|---------|-------------|--------|
| Ngủ gật (DROWSY) | EAR < 0.22 kéo dài | > 0.8 giây |
| Microsleep | EAR liên tục rất thấp | > 1.8 giây |
| Ngáp (YAWNING) | MAR – Mouth Aspect Ratio | MAR > 0.45 |
| Cúi đầu (HEAD_DOWN) | Góc Pitch — solvePnP | > 18° |
| Quay đầu (HEAD_TURN) | Góc Yaw — solvePnP | > 25° |
| Dùng điện thoại (PHONE_USE) | YOLO + IoU overlap | Phone gần người |
| Mất tập trung (DISTRACTED) | Gaze/Iris MediaPipe | Lệch > 28% |
| Nhiều người (MULTI_PERSON) | YOLO đếm người | > 1 người |
| Không có mặt (NO_FACE) | Mất tracking | > 3 giây |
| Mệt mỏi (FATIGUE_HIGH) | Combo ngáp + nhắm mắt | Cả 2 cùng lúc |

**Tùy chọn dòng lệnh:**
```bash
# macOS
python3.12 behavior_detector.py              # đầy đủ
python3.12 behavior_detector.py --no-yolo    # nhẹ hơn, không YOLO
python3.12 behavior_detector.py --camera 1   # dùng camera số 1
python3.12 behavior_detector.py --record     # ghi video

# Windows
python behavior_detector.py --no-yolo
```

---

### 4. UI Dashboard — `app_gui.py`

Giao diện Tkinter tích hợp camera + bảng thống kê:
- Video preview realtime
- FPS, số học sinh, số điện thoại, thời gian phiên
- Chi tiết từng học sinh: EAR, MAR, Pitch/Yaw, Blinks, Yawns
- Log cảnh báo theo thời gian thực
- Chụp màn hình, ghi video, Light/Dark mode

---

### 5. Phát hiện ngủ gật ResNet50 AI — `resnet_detector.py`

**Thuật toán:**
1. MediaPipe xác định vùng mắt → cắt ROI
2. ResNet50 phân loại `awake` / `sleepy` (Resize 224×224, normalize ImageNet)
3. EMA smoothing (α=0.4) làm mượt xác suất
4. Kết hợp EAR: nếu EAR < 0.22 → boost sleepy score ≥ 0.7

**Độ chính xác:** 95.43% (MRL Eye Dataset)

> **Lưu ý:** Phải chạy bằng `python3.12` (macOS) hoặc `python` 3.12 (Windows)

---

## So Sánh Hiệu Năng

| Chức năng | RAM | FPS | Ghi chú |
|-----------|-----|-----|---------|
| Phát hiện khuôn mặt | ~50 MB | 30 | Nhẹ nhất |
| Phát hiện ngủ gật | ~150 MB | 30 | MediaPipe + EAR |
| Hành vi (YOLO + AI) | ~800 MB | 15–25 | Đầy đủ nhất |
| Hành vi (Không YOLO) | ~200 MB | 25–30 | Nhẹ hơn |
| UI Dashboard | ~800 MB | 15–25 | Có giao diện |
| ResNet50 AI | ~300 MB | 20–25 | Chính xác nhất |

---

## Phím Tắt

| Phím | Tác dụng |
|------|----------|
| `q` hoặc `ESC` | Thoát |
| `r` | Reset cảnh báo |

---

## Xử Lý Lỗi Thường Gặp

**`No module named 'cv2'` hoặc thư viện khác**
```bash
pip install -r requirements.txt   # trong venv đã activate
```

**`No module named 'torch'`**
```bash
# macOS
pip install torch torchvision
# Windows (CPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**`mediapipe has no attribute 'solutions'`** (mediapipe quá mới)
```bash
pip install mediapipe==0.10.14
```

**`shape_predictor_68_face_landmarks.dat not found`**
```bash
python3.12 download_models.py   # macOS
python download_models.py       # Windows
```

**`resnet_drowsiness.pth not found`**
```bash
python3.12 train_resnet.py   # macOS
python train_resnet.py       # Windows
```

**`Could not open webcam`**
- macOS: System Settings > Privacy & Security > Camera → cấp quyền
- Windows: Settings > Privacy > Camera → bật quyền truy cập
- Thử đổi camera: `--camera 1`

---

## Cấu Trúc Thư Mục

```
CAM/
├── main.py                  GUI launcher chính
├── app_gui.py               Dashboard tích hợp camera
├── behavior_detector.py     Engine phát hiện hành vi
├── blinkDetect.py           Phát hiện ngủ gật (EAR)
├── resnet_detector.py       Phát hiện ngủ gật (ResNet AI)
├── face-try.py              Phát hiện khuôn mặt cơ bản
├── train_resnet.py          Train ResNet model (chạy 1 lần)
├── download_models.py       Tải model Dlib
├── requirements.txt         Danh sách thư viện
├── alarm.wav                Âm thanh cảnh báo
├── start.sh                 Script khởi động (macOS/Linux)
├── yolov8n.pt               YOLOv8 model
└── models/
    ├── shape_predictor_68_face_landmarks.dat   Dlib model (95MB)
    ├── haarcascade_frontalface_default.xml     Haar Cascade
    └── resnet_drowsiness.pth                   ResNet AI model (91MB)
```
