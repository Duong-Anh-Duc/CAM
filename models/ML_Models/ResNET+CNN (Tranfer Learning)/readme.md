# Phát Hiện Ngủ Gật Bằng ResNet50

Một dự án học sâu để phát hiện ngủ gật bằng cách phân loại trạng thái mắt là "tỉnh táo" hoặc "ngủ gật" sử dụng transfer learning với ResNet50.

## Tổng Quan Dự Án

Dự án này triển khai một mô hình phân loại nhị phân để phát hiện ngủ gật bằng cách phân tích ảnh mắt. Mô hình đạt được độ chính xác xác thực là 98.5% sử dụng ResNet50 pre-trained có kết hợp các lớp phân loại tùy chỉnh.

### Tính Năng Chính
- Transfer Learning: Sử dụng ResNet50 pre-trained từ ImageNet
- Phân loại nhị phân: Phân loại mắt là "tỉnh táo" hoặc "ngủ gật"
- Độ chính xác cao: Đạt 98.5% độ chính xác xác thực
- Tối ưu GPU: Cấu hình cho Google Colab với gia tốc GPU
- Sẵn sàng sản xuất: Mô hình đã lưu sẵn để triển khai

## Hiệu Suất Mô Hình

| Chỉ Số | Huấn Luyện | Xác Thực |
|--------|----------|------------|
| Độ Chính Xác | 97.76% | 98.50% |
| Loss | 0.0600 | 0.0415 |

## Dataset

Dự án sử dụng MRL Eye Dataset từ Kaggle chứa:
- Tập Huấn Luyện: 50,937 ảnh (chia 80/20 cho huấn luyện/xác thực)
- Tập Xác Thực: 16,980 ảnh  
- Lớp: 2 (tỉnh táo, ngủ gật)
- Kích Thước Ảnh: 224x224 pixels

Nguồn Dataset: [MRL Eye Dataset trên Kaggle](https://www.kaggle.com/datasets/akashshingha850/mrl-eye-dataset)

## Kiến Trúc Mô Hình

```
ResNet50 (Pre-trained, Frozen)
    |
    v
GlobalAveragePooling2D
    |
    v
Dense(128, activation='relu')
    |
    v
Dropout(0.3)
    |
    v
Dense(2, activation='softmax')
```

### Thông Số Mô Hình
- Base Model: ResNet50 (ImageNet pre-trained, đóng băng)
- Hình dạng Đầu Vào: (224, 224, 3)
- Đầu Ra: 2 lớp (tỉnh táo/ngủ gật)
- Tổng Số Tham Số: ~25M (chỉ ~260K có thể huấn luyện)
- Bộ Tối Ưu Hóa: Adam
- Hàm Mất Mát: Binary Cross-Entropy

## Bắt Đầu

### Yêu Cầu Trước Tiên
- Python 3.7+
- TensorFlow 2.x
- Google Colab (khuyến nghị) hoặc môi trường GPU cục bộ
- Thông tin đăng nhập Kaggle API

### Cài Đặt & Thiết Lập

1. Clone repository:
```bash
git clone https://github.com/yourusername/eye-drowsiness-detection.git
cd eye-drowsiness-detection
```

2. Cài đặt phụ thuộc:
```bash
pip install tensorflow matplotlib kagglehub
```

3. Chạy trong Google Colab:
   - Upload Drowsiness.ipynb vào Google Colab
   - Bật GPU: Runtime → Change runtime type → Hardware accelerator → GPU
   - Chạy tất cả cell

### Cách Sử Dụng

1. Download Dataset:
```python
import kagglehub
path = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset")
```

2. Huấn Luyện Mô Hình:
```python
# Cấu hình đường dẫn
train_folder_path = '/path/to/train'
val_folder_path = '/path/to/val'

# Huấn luyện mô hình (5 epochs)
history = model.fit(train_ds, validation_data=val_ds, epochs=5)
```

3. Lưu Mô Hình:
```python
model.save('/content/saved_model/resnet.keras')
```

4. Tải và Sử Dụng Để Dự Đoán:
```python
import tensorflow as tf
loaded_model = tf.keras.models.load_model('/path/to/resnet.keras')
prediction = loaded_model.predict(new_image)
```
## Chi Tiết Huấn Luyện

### Siêu Tham Số
- Kích Thước Ảnh: 224x224 pixels
- Kích Thước Batch: 16
- Epochs: 5
- Tỷ Lệ Học: Adam mặc định (0.001)
- Chia Xác Thực: 20%
- Tăng Cường Dữ Liệu: Không (có thể thêm để cải thiện)

### Kết Quả Huấn Luyện
```
Epoch 1/5: val_accuracy: 0.9753
Epoch 2/5: val_accuracy: 0.9756  
Epoch 3/5: val_accuracy: 0.9761
Epoch 4/5: val_accuracy: 0.9806
Epoch 5/5: val_accuracy: 0.9850
```
## Tùy Chỉnh & Cải Tiến

### Những Cải Tiến Tiềm Năng
- Tăng Cường Dữ Liệu: Thêm xoay vòng, điều chỉnh độ sáng, độ tương phản
- Fine-tuning: Mở khóa các lớp trên cùng của ResNet50 để độ chính xác tốt hơn
- Xử Lý Thời Gian Thực: Tối ưu hóa để nhập camera/webcam
- Đa lớp: Mở rộng để phát hiện các mức độ ngủ gật khác nhau
- Phương Pháp Kết Hợp: Kết hợp các mô hình để hiệu suất tốt hơn

### Ví Dụ Fine-tuning
```python
# Mở khóa các lớp trên cùng để fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-10]:
    layer.trainable = False

# Sử dụng tỷ lệ học thấp hơn
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='categorical_crossentropy',
              metrics=['accuracy'])
```

## Phân Tích Hiệu Suất

Mô hình cho thấy hiệu suất xuất sắc với:
- Không Overfitting: Độ chính xác xác thực liên tục cao hơn huấn luyện
- Hội Tụ Nhanh: Đạt độ chính xác cao trong 5 epochs
- Huấn Luyện Ổn Định: Cải tiến nhất quán qua các epochs

## Đóng Góp

Đóng góp được hoan ngênh! Vui lòng:
- Báo cáo lỗi hoặc vấn đề
- Đề xuất các tính năng hoặc cải tiến mới
- Gửi pull request
- Chia sẻ kết quả và sửa đổi của bạn

## Lời Cảm Ơn

- Dataset: MRL Eye Dataset bởi akashshingha850 trên Kaggle
- Base Model: ResNet50 từ TensorFlow/Keras
- Platform: Google Colab vì cung cấp tài nguyên GPU miễn phí
