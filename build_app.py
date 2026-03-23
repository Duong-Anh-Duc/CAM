#!/usr/bin/env python3
"""
Build script — đóng gói ứng dụng thành file thực thi (.exe / .app)

Cách dùng:
    python build_app.py

Yêu cầu:
    pip install pyinstaller

Kết quả:
    dist/GiamSatHocSinh/       ← thư mục chứa app đã build
    dist/GiamSatHocSinh.exe    ← (Windows) hoặc .app (macOS)
"""

import os
import sys
import platform
import subprocess

APP_NAME = "GiamSatHocSinh"
MAIN_SCRIPT = "main.py"
ICON = None  # Đặt path icon nếu có, ví dụ: "icon.ico" (Windows) hoặc "icon.icns" (macOS)

# Các file data cần đóng gói kèm
DATA_FILES = [
    ("alarm.wav", "."),
    ("yolov8n.pt", "."),
    ("models/shape_predictor_68_face_landmarks.dat", "models"),
    ("models/haarcascade_frontalface_default.xml", "models"),
    ("models/dlib_face_recognition_resnet_model_v1.dat", "models"),
]

# Thêm resnet model nếu có
if os.path.exists("models/resnet_drowsiness.pth"):
    DATA_FILES.append(("models/resnet_drowsiness.pth", "models"))

# Các script con cần đóng gói kèm (vì main.py gọi subprocess)
EXTRA_SCRIPTS = [
    "behavior_detector.py",
    "blinkDetect.py",
    "face-try.py",
    "resnet_detector.py",
    "app_gui.py",
    "train_resnet.py",
    "download_models.py",
    "license_manager.py",
]

# Các hidden imports mà PyInstaller có thể bỏ sót
HIDDEN_IMPORTS = [
    "cv2",
    "dlib",
    "numpy",
    "scipy",
    "scipy.spatial",
    "PIL",
    "PIL.Image",
    "PIL.ImageDraw",
    "PIL.ImageFont",
    "pygame",
    "mediapipe",
    "ultralytics",
    "torch",
    "torchvision",
    "tkinter",
]


def build():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(base_dir)

    # Kiểm tra file chính
    if not os.path.exists(MAIN_SCRIPT):
        print(f"[LỖI] Không tìm thấy {MAIN_SCRIPT}")
        sys.exit(1)

    # Output ra Desktop
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    dist_path = os.path.join(desktop, "GiamSatHocSinh_Build")
    work_path = os.path.join(base_dir, "build")

    # Build lệnh PyInstaller
    cmd = [
        sys.executable, "-m", "PyInstaller",
        "--name", APP_NAME,
        "--noconfirm",        # ghi đè không hỏi
        "--windowed",         # không hiện console (GUI app)
        "--distpath", dist_path,
        "--workpath", work_path,
    ]

    # Thêm data files
    sep = ";" if platform.system() == "Windows" else ":"
    for src, dst in DATA_FILES:
        if os.path.exists(src):
            cmd.extend(["--add-data", f"{src}{sep}{dst}"])
        else:
            print(f"[CẢNH BÁO] Bỏ qua file không tồn tại: {src}")

    # Thêm extra scripts
    for script in EXTRA_SCRIPTS:
        if os.path.exists(script):
            cmd.extend(["--add-data", f"{script}{sep}."])

    # Hidden imports
    for imp in HIDDEN_IMPORTS:
        cmd.extend(["--hidden-import", imp])

    # Icon
    if ICON and os.path.exists(ICON):
        cmd.extend(["--icon", ICON])

    # Main script
    cmd.append(MAIN_SCRIPT)

    print("=" * 60)
    print(f"  Đang build {APP_NAME}...")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    print("=" * 60)
    print(f"\nLệnh: {' '.join(cmd)}\n")

    result = subprocess.run(cmd)

    if result.returncode == 0:
        print("\n" + "=" * 60)
        print(f"  BUILD THÀNH CÔNG!")
        print(f"  App tại: {dist_path}/")
        if platform.system() == "Darwin":
            print(f"  Chạy: open \"{dist_path}/{APP_NAME}.app\"")
        elif platform.system() == "Windows":
            print(f"  Chạy: \"{dist_path}\\{APP_NAME}\\{APP_NAME}.exe\"")
        print("=" * 60)
    else:
        print(f"\n[LỖI] Build thất bại với code {result.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    build()
