#!/usr/bin/env python3
"""
train_resnet.py
==============
Train ResNet50 phát hiện mắt mở/nhắm (ngủ gật) trên MRL Eye Dataset.
Dataset: kagglehub - akashshingha850/mrl-eye-dataset (awake / sleepy)
Output:  models/resnet_drowsiness.pth

Yêu cầu:
    pip install torch torchvision kagglehub

Chạy:
    python train_resnet.py
"""

import os
import sys
import time
from pathlib import Path

# ── Kiểm tra thư viện ─────────────────────────────────────────────
def _check_deps():
    missing = []
    for lib in ("torch", "torchvision", "kagglehub"):
        try:
            __import__(lib)
        except ImportError:
            missing.append(lib)
    if missing:
        print(f"[LỖI] Thiếu thư viện: {', '.join(missing)}")
        print(f"      Cài đặt: pip install {' '.join(missing)}")
        sys.exit(1)

_check_deps()

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
import kagglehub

# ── Cấu hình ──────────────────────────────────────────────────────
_DIR            = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(_DIR, "models", "resnet_drowsiness.pth")
IMG_SIZE        = 224
BATCH_SIZE      = 32
EPOCHS          = 5
LR              = 1e-3


# ── Tải dataset ───────────────────────────────────────────────────
def get_dataset_root() -> str:
    """Trả về đường dẫn thư mục chứa train/val (ưu tiên local cache)."""
    # 1. Kiểm tra kagglehub cache mặc định (cross-platform)
    default_cache = Path(os.path.expanduser("~")) / ".cache" / "kagglehub" / "datasets" / "akashshingha850" / "mrl-eye-dataset"
    if default_cache.exists():
        for version_dir in sorted(default_cache.iterdir(), reverse=True):
            candidate = version_dir / "data"
            if (candidate / "train").is_dir():
                print(f"[1/4] Dataset co san tai: {candidate}")
                return str(candidate)

    # 2. Tải từ Kaggle nếu chưa có
    print("[1/4] Dang tai MRL Eye Dataset tu Kaggle...")
    import kagglehub
    path = kagglehub.dataset_download("akashshingha850/mrl-eye-dataset")
    print(f"      Cache path: {path}")

    base = Path(path)
    for candidate in [base / "data", base]:
        if (candidate / "train").is_dir():
            return str(candidate)
    for p in base.rglob("train"):
        if p.is_dir():
            return str(p.parent)
    raise FileNotFoundError(
        f"Khong tim thay thu muc 'train' trong dataset tai: {path}"
    )


# ── Build model ───────────────────────────────────────────────────
def build_model(num_classes: int = 2) -> nn.Module:
    """ResNet50 pretrained ImageNet, chỉ train phần head."""
    model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
    # Đóng băng toàn bộ base
    for param in model.parameters():
        param.requires_grad = False
    # Thay fc head (giữ nguyên kiến trúc như notebook gốc)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 128),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(128, num_classes),
    )
    return model


# ── Training loop ─────────────────────────────────────────────────
def train():
    # Device: MPS (Apple Silicon) → CUDA → CPU
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"[2/4] Device: {device}")

    data_root = get_dataset_root()
    print(f"      Data root: {data_root}")

    # Transforms - ảnh MRL là grayscale 82x82, cần chuyển sang RGB
    transform_train = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.2, contrast=0.2),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    transform_val = T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((IMG_SIZE, IMG_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    train_ds = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
    val_ds   = torchvision.datasets.ImageFolder(val_dir,   transform=transform_val)

    print(f"[3/4] Dataset:")
    print(f"      Classes : {train_ds.classes}  (index: {train_ds.class_to_idx})")
    print(f"      Train   : {len(train_ds)} ảnh")
    print(f"      Val     : {len(val_ds)} ảnh")

    num_workers = 0 if str(device) == "mps" else 4  # MPS không hỗ trợ multiprocessing tốt
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=num_workers, pin_memory=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=num_workers, pin_memory=False)

    model     = build_model(num_classes=len(train_ds.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)

    print(f"[4/4] Bắt đầu training ({EPOCHS} epochs)...\n")
    best_val_acc = 0.0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────────────
        model.train()
        t0 = time.time()
        running_loss, correct, total = 0.0, 0, 0

        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch {epoch}/{EPOCHS} | Batch {batch_idx+1}/{len(train_loader)} "
                      f"| Loss: {running_loss/(batch_idx+1):.4f} "
                      f"| Acc: {100.*correct/total:.2f}%")

        train_acc = 100. * correct / total

        # ── Validation ─────────────────────────────────────────────
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                preds = outputs.argmax(dim=1)
                val_correct += preds.eq(labels).sum().item()
                val_total   += labels.size(0)

        val_acc = 100. * val_correct / val_total
        elapsed = time.time() - t0
        print(f"\nEpoch {epoch}/{EPOCHS} — Train: {train_acc:.2f}%  Val: {val_acc:.2f}%  "
              f"({elapsed:.0f}s)\n")

        # ── Lưu model tốt nhất ─────────────────────────────────────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_to_idx":     train_ds.class_to_idx,
                "idx_to_class":     {v: k for k, v in train_ds.class_to_idx.items()},
                "val_acc":          val_acc,
                "epoch":            epoch,
                "img_size":         IMG_SIZE,
            }, MODEL_SAVE_PATH)
            print(f"  ✓ Model tốt nhất lưu tại: {MODEL_SAVE_PATH}  (Val: {val_acc:.2f}%)\n")

        scheduler.step()

    print(f"=== Hoàn thành! Val accuracy tốt nhất: {best_val_acc:.2f}% ===")
    print(f"    Model: {MODEL_SAVE_PATH}")


if __name__ == "__main__":
    train()
