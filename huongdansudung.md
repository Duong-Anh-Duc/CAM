# Hướng Dẫn Sử Dụng - Hệ Thống Giám Sát Tập Trung Học Sinh

---

## Tổng Quan Dự Án

Hệ thống phát hiện hành vi học sinh theo thời gian thực thông qua camera. Sử dụng trí tuệ nhân tạo để phát hiện các trạng thái: ngủ gật, microsleep, ngáp, cúi đầu, quay đầu, sử dụng điện thoại, nhiều người trong khung hình.

---

## Công Nghệ Sử Dụng

### Computer Vision

| Thư viện | Phiên bản | Chức năng |
|----------|-----------|-----------|
| OpenCV | >= 4.8 | Đọc camera, xử lý khung hình, vẽ overlay |
| Dlib | 19.24 | Phát hiện khuôn mặt, 68 điểm landmark |
| MediaPipe | >= 0.10 | Phân tích khuôn mặt (fallback khi thiếu Dlib) |
| Ultralytics YOLOv8 | >= 8.0 | Phát hiện điện thoại, nhiều người |

### Machine Learning / Deep Learning

| Mô hình | Chức năng |
|---------|-----------|
| ResNet50 (Transfer Learning) | Phân loại mắt mở / mắt nhắm (95.43% accuracy) |
| Dlib shape_predictor_68 | Dự đoán 68 điểm landmark trên khuôn mặt |
| YOLOv8 Nano | Phát hiện vật thể (điện thoại, người) |
| Haar Cascade | Phát hiện khuôn mặt cơ bản |

### Giao Diện

| Thư viện | Chức năng |
|----------|-----------|
| Tkinter | GUI desktop launcher |
| Streamlit | Web app chạy trên trình duyệt |
| PIL / Pillow | Render chữ tiếng Việt trên khung hình |

### Ngôn Ngữ và Môi Trường

- Python 3.9 - 3.12 (khuyên dùng 3.12)
- PyTorch >= 2.0 với Apple Silicon MPS
- macOS / Windows / Linux

### Âm Thanh

- pygame (macOS / Linux)
- playsound (Windows)

---

## Cài Đặt Môi Trường

### Yêu Cầu Hệ Thống

- Python 3.9 trở lên (khuyên dùng 3.12 vì PyTorch chưa hỗ trợ 3.14)
- RAM tối thiểu 4GB (khuyên dùng 8GB khi chạy YOLO)
- Camera (webcam hoặc camera ngoài)
- macOS Apple Silicon: hỗ trợ GPU qua MPS

### Bước 1 - Kiểm Tra Python

```bash
python3 --version
python3.12 --version
```

Nếu chưa có Python 3.12 trên macOS:

```bash
brew install python@3.12
```

### Bước 2 - Cài Thư Viện Chính

```bash
cd /đường-dẫn-đến-dự-án
pip install -r requirements.txt
```

Nếu dlib bị lỗi:

```bash
pip install cmake
pip install dlib
```

### Bước 3 - Cài PyTorch (cho ResNet AI)

PyTorch chưa hỗ trợ Python 3.14, cần dùng Python 3.12:

```bash
python3.12 -m pip install torch torchvision --break-system-packages
```

Trên Windows hoặc Linux:

```bash
python3.12 -m pip install torch torchvision
```

### Bước 4 - Tải Model Dlib

```bash
python3 download_models.py
```

File được tải về: `models/shape_predictor_68_face_landmarks.dat` (~95MB)

Nếu script bị lỗi, tải thủ công:

```bash
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 \
  "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bunzip2 models/shape_predictor_68_face_landmarks.dat.bz2
```

### Bước 5 - Train ResNet Model (chỉ làm 1 lần)

Chỉ cần chạy lần đầu tiên. Dataset MRL Eye sẽ tự động dùng cache nếu đã có:

```bash
python3.12 train_resnet.py
```

Thời gian train: khoảng 15-25 phút trên Apple Silicon MPS.
Kết quả lưu tại: `models/resnet_drowsiness.pth`

---

## Cách Chạy Ứng Dụng

### Cách 1 - GUI Launcher (khuyên dùng)

Mở cửa sổ chọn chế độ, click nút để bật / tắt từng tính năng:

```bash
python3 main.py
```

Các nút trong GUI:

| Nút | Chức năng |
|-----|-----------|
| Phát hiện khuôn mặt | Nhận diện khuôn mặt cơ bản (Haar Cascade) |
| Phát hiện ngủ gật | Đo EAR bằng Dlib, cảnh báo khi nhắm mắt lâu |
| Phát hiện hành vi (YOLO + Dlib) | Phát hiện đầy đủ các hành vi, có YOLO |
| Phát hiện hành vi (Chỉ Dlib) | Như trên nhưng không dùng YOLO, nhẹ hơn |
| Mở giao diện giám sát (UI Dashboard) | Camera + bảng thống kê tích hợp |
| Phát hiện Ngủ Gật (ResNet50 AI) | Sử dụng mô hình AI đã train (95.43%) |

### Cách 2 - Chạy Từng Script

**Phát hiện hành vi đầy đủ (YOLO + Dlib + MediaPipe):**

```bash
python3 behavior_detector.py
```

**Phát hiện hành vi không dùng YOLO (nhẹ hơn, ít RAM hơn):**

```bash
python3 behavior_detector.py --no-yolo
```

**Dùng camera số 1 (mặc định là 0):**

```bash
python3 behavior_detector.py --camera 1
```

**Ghi lại video:**

```bash
python3 behavior_detector.py --record
```

**Kết hợp nhiều tùy chọn:**

```bash
python3 behavior_detector.py --no-yolo --camera 1 --record
```

**Phát hiện ngủ gật đơn giản (EAR + Dlib):**

```bash
python3 blinkDetect.py
```

**Test nhận diện khuôn mặt:**

```bash
python3 face-try.py
```

**Phát hiện ngủ gật bằng ResNet50 AI:**

```bash
python3.12 resnet_detector.py
```

**Mở giao diện dashboard tích hợp:**

```bash
python3 app_gui.py
```

**Web app trên trình duyệt:**

```bash
streamlit run streamlit_app/streamlit_app.py
```

Truy cập tại: `http://localhost:8501`

---

## Phím Tắt Khi Đang Chạy Camera

| Phím | Tác dụng |
|------|----------|
| `q` hoặc `ESC` | Thoát chương trình |
| `r` | Reset cảnh báo, xóa trạng thái ngủ gật |

---

## Các Hành Vi Được Phát Hiện

| Hành vi | Phương pháp | Ngưỡng |
|---------|-------------|--------|
| Ngủ gật (DROWSY) | Eye Aspect Ratio (EAR) | EAR < 0.22 |
| Microsleep | EAR liên tục | > 1.8 giây nhắm mắt |
| Ngáp | Mouth Aspect Ratio (MAR) | MAR > ngưỡng |
| Cúi đầu | Ước lượng góc đầu (Pitch) | Pitch âm |
| Quay đầu | Ước lượng góc đầu (Yaw) | Yaw lớn |
| Dùng điện thoại | YOLO phát hiện vật thể | Phone + IoU overlap |
| Nhiều người | Đếm số khuôn mặt | > 1 khuôn mặt |
| Không có mặt | Mất theo dõi khuôn mặt | > N giây |
| Mất tập trung | Gaze lệch khỏi trung tâm | Độ lệch lớn |

---

## Cấu Trúc Thư Mục

```
student-drowsiness-detection-system/
├── main.py                  GUI launcher chính
├── app_gui.py               Dashboard tích hợp camera
├── behavior_detector.py     Engine phát hiện hành vi chính
├── blinkDetect.py           Phát hiện ngủ gật đơn giản
├── resnet_detector.py       Phát hiện ngủ gật bằng ResNet AI
├── train_resnet.py          Script train ResNet model
├── face-try.py              Test nhận diện khuôn mặt
├── download_models.py       Tải model Dlib
├── requirements.txt         Danh sách thư viện
├── alarm.wav                Âm thanh cảnh báo
├── models/
│   ├── shape_predictor_68_face_landmarks.dat   Dlib model (95MB)
│   ├── haarcascade_frontalface_default.xml     OpenCV face detector
│   ├── resnet_drowsiness.pth                   ResNet AI model (91MB)
│   └── ML_Models/
│       └── ResNET+CNN (Tranfer Learning)/
│           └── Drowsiness.ipynb                Notebook train trên Colab
└── streamlit_app/
    ├── streamlit_app.py     Web interface
    └── streamlit_app_pwa.py PWA version
```

---

## Xử Lý Lỗi Thường Gặp

**Lỗi: No module named 'dlib'**
```bash
pip install cmake
pip install dlib
```

**Lỗi: No module named 'torch'**
```bash
python3.12 -m pip install torch torchvision --break-system-packages
```

**Lỗi: shape_predictor_68_face_landmarks.dat not found**
```bash
python3 download_models.py
```

**Lỗi: resnet_drowsiness.pth not found**
```bash
python3.12 train_resnet.py
```

**Lỗi: Could not open webcam**

Kiểm tra camera có được cấp quyền trong System Preferences > Privacy > Camera.
Thử đổi số camera: `--camera 1`

**Lỗi: Error deserializing object (dlib)**

File .dat bị hỏng, tải lại:
```bash
curl -L -o models/shape_predictor_68_face_landmarks.dat.bz2 \
  "https://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
bunzip2 models/shape_predictor_68_face_landmarks.dat.bz2
```

---

## Hiệu Năng

| Chế độ | RAM | FPS ước tính |
|--------|-----|--------------|
| behavior_detector (YOLO + Dlib) | ~800MB | 15-25 fps |
| behavior_detector (chỉ Dlib) | ~200MB | 25-30 fps |
| resnet_detector | ~300MB | 20-25 fps |
| blinkDetect | ~150MB | 30 fps |

---

## Lưu Ý

- Script `resnet_detector.py` và `train_resnet.py` phải chạy bằng `python3.12`, không phải `python3`, vì PyTorch chưa hỗ trợ Python 3.14.
- YOLOv8 sẽ tự động tải file `yolov8n.pt` (~6MB) lần đầu chạy nếu chưa có.
- Video ghi được lưu thành `recording_YYYYMMDD_HHMMSS.avi` tại thư mục gốc.
- Ảnh chụp màn hình lưu thành `screenshot_YYYYMMDD_HHMMSS.jpg` tại thư mục gốc.
