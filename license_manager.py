#!/usr/bin/env python3
"""
Hệ thống quản lý license key cho ứng dụng Giám Sát Học Sinh.

Dùng cho:
  - generate_key.py : tạo key (chỉ bạn giữ)
  - main.py         : kiểm tra key khi khởi động app
"""

import hashlib
import hmac
import base64
import time
import os
import json
from datetime import datetime, timedelta

# ── Secret key — CHỈ BẠN BIẾT, không gửi cho khách ──────────────
# Đổi chuỗi này thành bất kỳ chuỗi bí mật nào bạn muốn
_SECRET = b"CAM_GIAM_SAT_HOC_SINH_2024_SECRET_KEY_XYZ"

# ── Đường dẫn file license ───────────────────────────────────────
_LICENSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".license")


def generate_key(days: int = 3, customer_name: str = "") -> str:
    """Tạo license key có thời hạn.

    Args:
        days: số ngày hiệu lực (mặc định 3)
        customer_name: tên khách hàng (optional, ghi vào key)

    Returns:
        License key dạng string
    """
    expire_ts = int(time.time()) + days * 86400
    payload = f"{expire_ts}|{customer_name}|CAM"
    signature = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
    raw = f"{payload}|{signature}"
    key = base64.urlsafe_b64encode(raw.encode()).decode()
    return key


def verify_key(key: str) -> dict:
    """Kiểm tra license key.

    Returns:
        dict với:
          valid: bool
          message: str
          days_left: int (nếu valid)
          customer: str (nếu valid)
    """
    try:
        raw = base64.urlsafe_b64decode(key.encode()).decode()
        parts = raw.split("|")
        if len(parts) != 4:
            return {"valid": False, "message": "Key không hợp lệ"}

        expire_ts_str, customer, tag, signature = parts
        if tag != "CAM":
            return {"valid": False, "message": "Key không hợp lệ"}

        # Verify signature
        payload = f"{expire_ts_str}|{customer}|{tag}"
        expected_sig = hmac.new(_SECRET, payload.encode(), hashlib.sha256).hexdigest()[:16]
        if not hmac.compare_digest(signature, expected_sig):
            return {"valid": False, "message": "Key đã bị chỉnh sửa"}

        # Check expiry
        expire_ts = int(expire_ts_str)
        now = int(time.time())
        if now > expire_ts:
            expire_date = datetime.fromtimestamp(expire_ts).strftime("%d/%m/%Y %H:%M")
            return {
                "valid": False,
                "message": f"Key đã hết hạn từ {expire_date}",
                "expired": True,
            }

        days_left = (expire_ts - now) / 86400
        expire_date = datetime.fromtimestamp(expire_ts).strftime("%d/%m/%Y %H:%M")
        return {
            "valid": True,
            "message": f"Còn {days_left:.1f} ngày (hết hạn: {expire_date})",
            "days_left": days_left,
            "customer": customer,
            "expire_date": expire_date,
        }

    except Exception:
        return {"valid": False, "message": "Key không hợp lệ"}


def save_license(key: str):
    """Lưu key vào file .license"""
    with open(_LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump({"key": key.strip(), "activated": datetime.now().isoformat()}, f)


def load_license() -> str | None:
    """Đọc key từ file .license. Trả None nếu chưa có."""
    if not os.path.exists(_LICENSE_FILE):
        return None
    try:
        with open(_LICENSE_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data.get("key")
    except Exception:
        return None


def delete_license():
    """Xóa file license."""
    if os.path.exists(_LICENSE_FILE):
        os.remove(_LICENSE_FILE)


def check_or_prompt() -> bool:
    """Kiểm tra license, hiện dialog nhập key nếu chưa có / hết hạn.
    Trả True nếu hợp lệ, False nếu không.
    Dùng Tkinter dialog."""
    import tkinter as tk
    from tkinter import messagebox

    key = load_license()

    # Nếu đã có key → kiểm tra
    if key:
        result = verify_key(key)
        if result["valid"]:
            return True
        # Key hết hạn hoặc sai → xóa, yêu cầu nhập lại
        delete_license()

    # Hiện dialog nhập key
    activated = [False]

    def on_activate():
        entered = entry.get().strip()
        if not entered:
            messagebox.showwarning("Cảnh báo", "Vui lòng nhập license key!")
            return
        result = verify_key(entered)
        if result["valid"]:
            save_license(entered)
            customer = result.get("customer", "")
            name_str = f" ({customer})" if customer else ""
            messagebox.showinfo(
                "Kích hoạt thành công",
                f"Chào mừng{name_str}!\n{result['message']}"
            )
            activated[0] = True
            root.destroy()
        else:
            messagebox.showerror("Lỗi", f"Key không hợp lệ:\n{result['message']}")

    def on_quit():
        root.destroy()

    root = tk.Tk()
    root.title("Kích hoạt bản quyền - Giám Sát Học Sinh")
    root.geometry("520x280")
    root.resizable(False, False)
    root.configure(bg="#f0f0f0")

    # Center on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - 260
    y = (root.winfo_screenheight() // 2) - 140
    root.geometry(f"+{x}+{y}")

    frame = tk.Frame(root, bg="#f0f0f0", padx=30, pady=20)
    frame.pack(expand=True)

    tk.Label(frame, text="Hệ thống Giám Sát Tập Trung Học Sinh",
             font=("Helvetica", 14, "bold"), bg="#f0f0f0").pack(pady=(0, 5))

    tk.Label(frame, text="Vui lòng nhập license key để kích hoạt:",
             font=("Helvetica", 11), bg="#f0f0f0").pack(pady=(0, 10))

    entry = tk.Entry(frame, width=50, font=("Courier", 12), justify="center")
    entry.pack(pady=5, ipady=4)
    entry.focus_set()

    btn_frame = tk.Frame(frame, bg="#f0f0f0")
    btn_frame.pack(pady=15)

    # macOS Tkinter không hỗ trợ bg/fg cho tk.Button → dùng ttk với style
    from tkinter import ttk
    style = ttk.Style()
    style.theme_use("clam")
    style.configure("Activate.TButton",
                    font=("Helvetica", 13, "bold"),
                    padding=(20, 8),
                    background="#4361ee",
                    foreground="white")
    style.map("Activate.TButton",
              background=[("active", "#3451de")])
    style.configure("Quit.TButton",
                    font=("Helvetica", 11),
                    padding=(15, 6))

    ttk.Button(btn_frame, text="Kích hoạt", command=on_activate,
               style="Activate.TButton").pack(side=tk.LEFT, padx=10)

    ttk.Button(btn_frame, text="Thoát", command=on_quit,
               style="Quit.TButton").pack(side=tk.LEFT, padx=10)

    # Enter key = activate
    root.bind("<Return>", lambda e: on_activate())
    root.protocol("WM_DELETE_WINDOW", on_quit)
    root.mainloop()

    return activated[0]
