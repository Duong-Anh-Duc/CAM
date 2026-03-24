# Tài Liệu Chức Năng – Hệ Thống Giám Sát Tập Trung Học Sinh

---

## 1. Phát hiện khuôn mặt

**Script:** `face-try.py`

**Công nghệ sử dụng:**
- OpenCV – Haar Cascade Classifier (`haarcascade_frontalface_default.xml`)

**Thuật toán:**
- **Viola-Jones (Haar Cascade):** Dùng bộ lọc Haar (đặc trưng hình chữ nhật) kết hợp với AdaBoost để phân loại nhanh. Kỹ thuật Integral Image giúp tính toán đặc trưng trong O(1).
- Sliding window quét toàn bộ khung hình ở nhiều tỉ lệ (scale pyramid).
- `scaleFactor=1.1`, `minNeighbors=4` — điều chỉnh độ nhạy và tránh false positive.

**Đầu ra:**
- Vẽ khung xanh quanh khuôn mặt phát hiện được.
- Hiển thị số khuôn mặt + label `#1, #2...` trên mỗi khung.
- Text trạng thái: `Phat hien N khuon mat` (xanh) hoặc `Khong phat hien khuon mat` (đỏ).

---

## 2. Phát hiện ngủ gật

**Script:** `blinkDetect.py`

**Công nghệ sử dụng:**
- MediaPipe FaceMesh (468 landmarks)
- SciPy – tính khoảng cách Euclidean

**Thuật toán:**
- **EAR – Eye Aspect Ratio:** Đo tỉ lệ chiều cao / chiều rộng mắt từ 6 điểm landmark.
  ```
  EAR = (||p1-p5|| + ||p2-p4||) / (2 × ||p0-p3||)
  ```
  - EAR < 0.22 → mắt nhắm
  - EAR ≥ 0.22 → mắt mở
- Đếm blink: phát hiện transition từ nhắm → mở.
- Nếu nhắm mắt liên tục > 1.5 giây → cảnh báo ngủ gật + phát âm thanh.

**Đầu ra:**
- Trạng thái mắt: `MAT: MO` (xanh) / `MAT: NHAM` (đỏ)
- `EAR: 0.xx` – chỉ số độ mở mắt
- `Blinks: N` – số lần nháy mắt
- `Mo: XX.X%  Nham: XX.X%` – % thời gian mắt mở/nhắm trong phiên
- Cảnh báo chữ đỏ + âm thanh khi ngủ gật

---

## 3. Phát hiện hành vi (YOLO + AI)

**Script:** `behavior_detector.py`

**Công nghệ sử dụng:**
- YOLOv8 Nano (`yolov8n.pt`) – phát hiện người và điện thoại
- MediaPipe FaceMesh (468 landmarks) – phân tích khuôn mặt (**dùng chính**)
- Dlib 68-landmark – fallback khi MediaPipe không cài được
- OpenCV – xử lý ảnh, vẽ overlay

**Các hành vi phát hiện & thuật toán:**

| Hành vi | Phương pháp | Ngưỡng |
|---------|-------------|--------|
| **Ngủ gật (DROWSY)** | EAR < 0.22 kéo dài | > 1.5 giây |
| **Microsleep** | EAR liên tục rất thấp | > 1.8 giây |
| **Ngáp (YAWNING)** | MAR – Mouth Aspect Ratio (khoảng cách dọc / ngang miệng) | MAR vượt ngưỡng |
| **Cúi đầu (HEAD_DOWN)** | Ước lượng góc Pitch từ 68 landmarks (solvePnP) | Pitch âm lớn |
| **Quay đầu (HEAD_TURN)** | Ước lượng góc Yaw từ 68 landmarks (solvePnP) | Yaw lớn |
| **Dùng điện thoại (PHONE_USE)** | YOLO phát hiện phone + IoU overlap với vùng người | Phone xuất hiện gần người |
| **Mất tập trung (DISTRACTED)** | Gaze estimation – độ lệch tâm nhìn khỏi camera | Độ lệch lớn |
| **Nhiều người (MULTI_PERSON)** | YOLO đếm số người trong khung | > 1 người |
| **Không có mặt (NO_FACE)** | Mất tracking khuôn mặt | > N giây |
| **Mệt mỏi nghiêm trọng (FATIGUE_HIGH)** | Combo: ngáp + mắt nhắm đồng thời | Cả hai cùng kích hoạt |

**Head Pose Estimation:**
- Dùng `cv2.solvePnP` với 6 điểm 3D chuẩn (mũi, cằm, mắt, miệng) để tính ma trận quay.
- Chuyển về góc Euler (Pitch / Yaw / Roll).

**Đầu ra:**
- Overlay realtime trên camera với nhãn cảnh báo từng học sinh.
- Âm thanh cảnh báo khi phát hiện hành vi bất thường.

---

## 4. Phát hiện hành vi (Không YOLO, nhẹ hơn)

**Script:** `behavior_detector.py --no-yolo`

Giống chức năng 3 nhưng **tắt YOLOv8**, chỉ dùng Dlib + MediaPipe.

| So sánh | Có YOLO | Không YOLO |
|---------|---------|------------|
| RAM | ~800 MB | ~200 MB |
| FPS | 15–25 | 25–30 |
| Phát hiện điện thoại | Có | Không |
| Phát hiện nhiều người | Có (YOLO) | Giới hạn (đếm mặt) |

---

## 5. Mở giao diện giám sát (UI Dashboard)

**Script:** `app_gui.py`

**Công nghệ sử dụng:**
- Tkinter – giao diện desktop
- BehaviorDetector (toàn bộ engine từ chức năng 3)
- PIL/Pillow – render frame lên Tkinter Canvas

**Chức năng dashboard:**
- Video preview realtime bên trái.
- Bảng thông số bên phải: FPS, số học sinh, số điện thoại, thời gian phiên.
- Chi tiết từng học sinh: EAR, MAR, góc Pitch/Yaw, số blink, số ngáp.
- Log cảnh báo theo thời gian thực (có timestamp, mức độ nghiêm trọng).
- Chụp màn hình, ghi video, reset trạng thái.
- Hỗ trợ Light/Dark mode.

---

## 6. Phát hiện Ngủ Gật (ResNet50 AI)

**Script:** `resnet_detector.py`
**Train model:** `train_resnet.py` (chạy 1 lần, ~15–25 phút)

**Công nghệ sử dụng:**
- ResNet50 (Transfer Learning, PyTorch) – model AI đã train sẵn
- MediaPipe FaceMesh – xác định vùng mắt
- Dataset: MRL Eye Dataset

**Thuật toán:**
1. **MediaPipe** xác định 6 landmarks mắt trái + phải → cắt ROI (vùng mắt).
2. **Preprocessing:** Resize 224×224, normalize theo ImageNet mean/std.
3. **ResNet50** phân loại ROI thành `awake` / `sleepy`.
   - Backbone ResNet50 (pretrained ImageNet) + custom head: `Linear(2048→128) → ReLU → Dropout(0.3) → Linear(128→2)`.
4. **EMA smoothing** (α=0.4) làm mượt xác suất qua các frame.
5. **Kết hợp EAR + ResNet:** Nếu EAR < 0.22 thì boost xác suất sleepy lên ≥ 0.7.
6. Ngủ gật nếu cả 2 mắt closed > 0.8s; Microsleep nếu > 1.8s.

**Độ chính xác:** 95.43% trên tập validation (MRL Eye Dataset).

**Đầu ra:**
- Thanh xác suất `Mắt trái` / `Mắt phải` ở cuối màn hình.
- Trạng thái: `Tỉnh táo`, `Mắt đang nhép`, `CẢNH BÁO NGỦ GẬT`, `MICROSLEEP`.
- Âm thanh cảnh báo.

---

## So Sánh Tổng Quan

| Chức năng | Thư viện chính | Phát hiện | RAM | FPS |
|-----------|---------------|-----------|-----|-----|
| Phát hiện khuôn mặt | OpenCV Haar Cascade | Khuôn mặt | ~50 MB | 30 |
| Phát hiện ngủ gật | MediaPipe + EAR | Ngủ gật, blink | ~150 MB | 30 |
| Hành vi (YOLO + AI) | YOLOv8 + Dlib + MediaPipe | 10 hành vi | ~800 MB | 15–25 |
| Hành vi (Không YOLO) | Dlib + MediaPipe | 8 hành vi | ~200 MB | 25–30 |
| UI Dashboard | BehaviorDetector + Tkinter | 10 hành vi | ~800 MB | 15–25 |
| ResNet50 AI | ResNet50 + MediaPipe | Ngủ gật (AI) | ~300 MB | 20–25 |
