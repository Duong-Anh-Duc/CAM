#!/usr/bin/env python3
"""Test tương thích Windows — chạy bởi GitHub Actions."""
import sys
import os

print("=" * 50)
print(f"  Python: {sys.version}")
print(f"  Platform: {sys.platform}")
print("=" * 50)

errors = []

# Test imports
libs = [
    ("cv2", "OpenCV"),
    ("dlib", "Dlib"),
    ("numpy", "NumPy"),
    ("scipy", "SciPy"),
    ("PIL", "Pillow"),
    ("torch", "PyTorch"),
    ("ultralytics", "Ultralytics"),
    ("tkinter", "Tkinter"),
    ("pygame", "Pygame"),
    ("mediapipe", "MediaPipe"),
]

for mod, name in libs:
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "OK")
        print(f"  [OK] {name}: {ver}")
    except ImportError as e:
        print(f"  [FAIL] {name}: {e}")
        errors.append(name)

# Test project modules
print()
for mod in ["behavior_detector", "license_manager"]:
    try:
        __import__(mod)
        print(f"  [OK] {mod}")
    except Exception as e:
        print(f"  [FAIL] {mod}: {e}")
        errors.append(mod)

# Test models
print()
models = [
    "models/shape_predictor_68_face_landmarks.dat",
    "models/dlib_face_recognition_resnet_model_v1.dat",
    "models/haarcascade_frontalface_default.xml",
    "yolov8n.pt",
    "alarm.wav",
]
for m in models:
    if os.path.exists(m):
        size = os.path.getsize(m) / 1024 / 1024
        print(f"  [OK] {m} ({size:.1f}MB)")
    else:
        print(f"  [MISS] {m}")

# Test ResNet model
print()
try:
    import torch
    if os.path.exists("models/resnet_drowsiness.pth"):
        ckpt = torch.load("models/resnet_drowsiness.pth", map_location="cpu", weights_only=False)
        acc = ckpt.get("val_acc", 0)
        print(f"  [OK] ResNet model (val_acc={acc:.2f}%)")
    else:
        print("  [SKIP] ResNet model not found (need train)")
except Exception as e:
    print(f"  [FAIL] ResNet model: {e}")

# Test font
print()
try:
    from behavior_detector import _find_font
    font = _find_font(18)
    print(f"  [OK] Font: {font}")
except Exception as e:
    print(f"  [FAIL] Font: {e}")
    errors.append("Font")

# Result
print()
print("=" * 50)
if errors:
    print(f"  FAIL: {', '.join(errors)}")
    sys.exit(1)
else:
    print("  WINDOWS COMPATIBLE - TAT CA OK")
print("=" * 50)
