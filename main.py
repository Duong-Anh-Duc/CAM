import tkinter as tk
from tkinter import ttk, messagebox
import subprocess
import os
import threading
import sys

face_proc = None
blink_proc = None
behavior_proc = None
behavior_noyolo_proc = None
app_gui_proc = None
resnet_proc = None
is_dark_mode = False  # Tracks current theme state


def _start_proc(script, args_extra=None):
    """Khởi chạy script con, trả về Popen object."""
    if not os.path.exists(script):
        messagebox.showerror("Lỗi", f"Không tìm thấy script '{script}'.")
        return None
    args = [sys.executable, script] + (args_extra or [])
    try:
        return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        messagebox.showerror("Lỗi", f"Không thể khởi động:\n{e}")
        return None


def run_face_detection(btn_face=None):
    global face_proc
    # Nếu đang chạy → dừng lại
    if face_proc and face_proc.poll() is None:
        face_proc.terminate()
        face_proc = None
        if btn_face:
            btn_face.config(text="Phát hiện khuôn mặt")
        return
    # Bắt đầu chạy
    face_proc = _start_proc("face-try.py")
    if face_proc is None:
        return
    if btn_face:
        btn_face.config(text="DỪNG – Phát hiện khuôn mặt")
    def check_proc():
        try:
            face_proc.communicate()
        except Exception:
            pass
        finally:
            if btn_face:
                btn_face.config(text="Phát hiện khuôn mặt")
    threading.Thread(target=check_proc, daemon=True).start()


def run_blink_detection(btn_blink=None):
    global blink_proc
    # Nếu đang chạy → dừng lại
    if blink_proc and blink_proc.poll() is None:
        blink_proc.terminate()
        blink_proc = None
        if btn_blink:
            btn_blink.config(text="Phát hiện ngủ gật")
        return
    # Bắt đầu chạy
    blink_proc = _start_proc("blinkDetect.py")
    if blink_proc is None:
        return
    if btn_blink:
        btn_blink.config(text="DỪNG – Phát hiện ngủ gật")
    def check_proc():
        try:
            blink_proc.communicate()
        except Exception:
            pass
        finally:
            if btn_blink:
                btn_blink.config(text="Phát hiện ngủ gật")
    threading.Thread(target=check_proc, daemon=True).start()


def run_behavior_detector(btn=None, use_yolo=True):
    """Chạy Behavior Detector tích hợp YOLO + MediaPipe + Dlib"""
    global behavior_proc, behavior_noyolo_proc
    proc_ref = behavior_proc if use_yolo else behavior_noyolo_proc
    orig_text = "Phát hiện hành vi (YOLO + Dlib)" if use_yolo else "Phát hiện hành vi (Chỉ Dlib, không YOLO)"
    # Nếu đang chạy → dừng lại
    if proc_ref and proc_ref.poll() is None:
        proc_ref.terminate()
        if use_yolo:
            behavior_proc = None
        else:
            behavior_noyolo_proc = None
        if btn:
            btn.config(text=orig_text)
        return
    # Bắt đầu chạy
    extra = [] if use_yolo else ["--no-yolo"]
    proc = _start_proc("behavior_detector.py", extra)
    if proc is None:
        return
    if use_yolo:
        behavior_proc = proc
    else:
        behavior_noyolo_proc = proc
    if btn:
        btn.config(text=f"DỪNG – {orig_text}")
    def check_proc():
        try:
            stdout, stderr = proc.communicate()
            if proc.returncode not in (0, None, -15):
                err = stderr.decode(errors='replace') if stderr else "Unknown error"
                if "KeyboardInterrupt" not in err:
                    messagebox.showerror("Lỗi Behavior Detector",
                        f"Thoát với code {proc.returncode}:\n{err[-600:]}")
        except Exception:
            pass
        finally:
            if btn:
                btn.config(text=orig_text)
    threading.Thread(target=check_proc, daemon=True).start()

def _find_python_with_torch():
    """Tìm Python interpreter có thể chạy torch (cross-platform)."""
    import shutil
    import platform

    # Ưu tiên Python hiện tại (nếu đã cài torch)
    try:
        result = subprocess.run(
            [sys.executable, "-c", "import torch; print('ok')"],
            capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            return sys.executable
    except Exception:
        pass

    # Trên Windows thử python, py -3.12, py -3.11, py -3.10
    if platform.system() == "Windows":
        candidates = ["python", "py -3.12", "py -3.11", "py -3.10"]
    else:
        candidates = ["python3.12", "python3.11", "python3.10", "python3"]

    for cmd in candidates:
        parts = cmd.split()
        exe = shutil.which(parts[0])
        if exe:
            try:
                full_cmd = [exe] + parts[1:] + ["-c", "import torch; print('ok')"]
                result = subprocess.run(full_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0:
                    return " ".join([exe] + parts[1:]) if len(parts) > 1 else exe
            except Exception:
                continue

    # Fallback: trả về python hiện tại
    return sys.executable


def run_resnet_detection(btn=None):
    """Chạy phát hiện ngủ gật bằng ResNet50 AI."""
    global resnet_proc
    orig_text = "Phat hien Ngu Gat (ResNet50 AI)"
    if resnet_proc and resnet_proc.poll() is None:
        resnet_proc.terminate()
        resnet_proc = None
        if btn:
            btn.config(text=orig_text)
        return
    py_exe = _find_python_with_torch()
    script = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resnet_detector.py")
    if not os.path.exists(script):
        messagebox.showerror("Loi", f"Khong tim thay: {script}")
        return
    try:
        cmd = py_exe.split() + [script] if " " in py_exe else [py_exe, script]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except Exception as e:
        messagebox.showerror("Loi", f"Khong the khoi dong ResNet detector:\n{e}")
        return
    resnet_proc = proc
    if btn:
        btn.config(text=f"DUNG – {orig_text}")
    def check_proc():
        try:
            stdout, stderr = proc.communicate()
            if proc.returncode not in (0, None, -15):
                err = stderr.decode(errors="replace") if stderr else ""
                if "KeyboardInterrupt" not in err and err.strip():
                    messagebox.showerror("Loi ResNet Detector",
                        f"Thoat voi code {proc.returncode}:\n{err[-600:]}")
        except Exception:
            pass
        finally:
            if btn:
                btn.config(text=orig_text)
    threading.Thread(target=check_proc, daemon=True).start()


def toggle_theme(root, frame, toggle_btn, labels=None):
    global is_dark_mode

    if is_dark_mode:
        bg, fg = "#f0f0f0", "#222222"
        frame.configure(style="Light.TFrame")
        toggle_btn.config(text="Chuyển sang Dark Mode")
        ttk.Style().configure('TButton', background="#ffffff", foreground="#000000")
    else:
        bg, fg = "#2e2e2e", "#dddddd"
        frame.configure(style="Dark.TFrame")
        toggle_btn.config(text="Chuyển sang Light Mode")
        ttk.Style().configure('TButton', background="#444444", foreground="#ffffff")

    root.configure(bg=bg)
    if labels:
        for lbl in labels:
            lbl.configure(bg=bg, fg=fg)

    is_dark_mode = not is_dark_mode

def on_quit(root):
    for proc in (face_proc, blink_proc, behavior_proc, behavior_noyolo_proc, app_gui_proc, resnet_proc):
        if proc and proc.poll() is None:
            proc.terminate()
    root.destroy()

def main():
    root = tk.Tk()
    root.title("Hệ thống giám sát tập trung học sinh")
    root.geometry("580x620")
    root.configure(bg="#f0f0f0")

    style = ttk.Style()
    style.theme_use("clam")

    # Frame styles
    style.configure("Light.TFrame", background="#f0f0f0")
    style.configure("Dark.TFrame", background="#2e2e2e")
    style.configure("TButton",
                    font=('Segoe UI', 12, 'bold'),
                    padding=10,
                    borderwidth=1,
                    relief="raised")
    style.configure("Accent.TButton",
                    font=('Segoe UI', 13, 'bold'),
                    padding=12)

    frame = ttk.Frame(root, padding=20, style="Light.TFrame")
    frame.pack(expand=True)

    # ── Tiêu đề ───────────────────────────────────────────────────
    lbl_title = tk.Label(frame,
        text="Hệ thống giám sát tập trung học sinh",
        font=('Segoe UI', 15, 'bold'),
        bg="#f0f0f0", fg="#222222")
    lbl_title.grid(row=0, column=0, columnspan=2, pady=(0, 10))

    # ── Nút cũ ────────────────────────────────────────────────────
    lbl_basic = tk.Label(frame, text="Basic Detectors",
        font=('Segoe UI', 9), bg="#f0f0f0", fg="#666666")
    lbl_basic.grid(row=1, column=0, columnspan=2)

    btn_face = ttk.Button(frame, text="Phát hiện khuôn mặt")
    btn_face.config(command=lambda: run_face_detection(btn_face))
    btn_face.grid(row=2, column=0, padx=10, pady=8, sticky="ew")

    btn_blink = ttk.Button(frame, text="Phát hiện ngủ gật")
    btn_blink.config(command=lambda: run_blink_detection(btn_blink))
    btn_blink.grid(row=2, column=1, padx=10, pady=8, sticky="ew")

    # ── Dấu phân cách ─────────────────────────────────────────────
    sep = ttk.Separator(frame, orient='horizontal')
    sep.grid(row=3, column=0, columnspan=2, sticky="ew", pady=10)

    # ── Nút Behavior Detector mới ──────────────────────────────────
    lbl_adv = tk.Label(frame, text="Nâng cao - Phát hiện hành vi học sinh (YOLO + Dlib)",
        font=('Segoe UI', 9), bg="#f0f0f0", fg="#333399")
    lbl_adv.grid(row=4, column=0, columnspan=2)

    btn_behavior = ttk.Button(frame,
        text="Phát hiện hành vi (YOLO + Dlib)",
        style="Accent.TButton")
    btn_behavior.config(command=lambda: run_behavior_detector(btn_behavior, use_yolo=True))
    btn_behavior.grid(row=5, column=0, columnspan=2,
                      padx=10, pady=6, sticky="ew")

    btn_behavior_noyolo = ttk.Button(frame,
        text="Phát hiện hành vi (Chỉ Dlib, không YOLO)")
    btn_behavior_noyolo.config(
        command=lambda: run_behavior_detector(btn_behavior_noyolo, use_yolo=False))
    btn_behavior_noyolo.grid(row=6, column=0, columnspan=2,
                             padx=10, pady=4, sticky="ew")

    # Mở giao diện app_gui.py (camera + dashboard tích hợp)
    def open_app_gui():
        global app_gui_proc
        if app_gui_proc and app_gui_proc.poll() is None:
            app_gui_proc.terminate()
            app_gui_proc = None
            btn_gui.config(text="Mở giao diện giám sát (UI Dashboard)")
            return
        p = _start_proc("app_gui.py")
        if p is None:
            return
        app_gui_proc = p
        btn_gui.config(text="DỪNG – Giao diện giám sát")
        def check():
            try:
                p.communicate()
            except Exception:
                pass
            finally:
                btn_gui.config(text="Mở giao diện giám sát (UI Dashboard)")
        threading.Thread(target=check, daemon=True).start()

    btn_gui = ttk.Button(frame,
        text="Mở giao diện giám sát (UI Dashboard)",
        style="Accent.TButton")
    btn_gui.config(command=open_app_gui)
    btn_gui.grid(row=7, column=0, columnspan=2,
                 padx=10, pady=6, sticky="ew")

    # Chú thích hành vi phát hiện
    note = (
        "Phát hiện học sinh: Ngủ gật | Microsleep | Ngáp | "
        "Cúi đầu | Quay đầu | Điện thoại | Nhiều người"
    )
    lbl_note = tk.Label(frame, text=note,
        font=('Segoe UI', 8), bg="#f0f0f0", fg="#555555",
        wraplength=500, justify="center")
    lbl_note.grid(row=8, column=0, columnspan=2, pady=(4, 0))

    sep2 = ttk.Separator(frame, orient='horizontal')
    sep2.grid(row=9, column=0, columnspan=2, sticky="ew", pady=10)

    # ── ResNet50 AI Section ────────────────────────────────────────
    lbl_resnet = tk.Label(frame,
        text="Deep Learning – ResNet50 Transfer Learning (AI Model)",
        font=('Segoe UI', 9), bg="#f0f0f0", fg="#880000")
    lbl_resnet.grid(row=10, column=0, columnspan=2)

    btn_resnet = ttk.Button(frame,
        text="Phat hien Ngu Gat (ResNet50 AI)",
        style="Accent.TButton")
    btn_resnet.config(command=lambda: run_resnet_detection(btn_resnet))
    btn_resnet.grid(row=11, column=0, columnspan=2,
                    padx=10, pady=6, sticky="ew")

    lbl_resnet_note = tk.Label(frame,
        text="Can chay train_resnet.py truoc de tao model | q/ESC=thoat  r=reset",
        font=('Segoe UI', 8), bg="#f0f0f0", fg="#888888")
    lbl_resnet_note.grid(row=12, column=0, columnspan=2, pady=(0, 4))

    sep3 = ttk.Separator(frame, orient='horizontal')
    sep3.grid(row=13, column=0, columnspan=2, sticky="ew", pady=10)

    # ── Controls ──────────────────────────────────────────────────
    theme_labels = [lbl_title, lbl_basic, lbl_adv, lbl_note, lbl_resnet, lbl_resnet_note]
    btn_toggle = ttk.Button(frame, text="Chuyển sang Dark Mode")
    btn_toggle.config(command=lambda: toggle_theme(root, frame, btn_toggle, theme_labels))
    btn_toggle.grid(row=14, column=0, padx=10, pady=4, sticky="ew")

    btn_quit = ttk.Button(frame, text="Thoát",
                          command=lambda: on_quit(root))
    btn_quit.grid(row=14, column=1, padx=10, pady=4, sticky="ew")

    root.protocol("WM_DELETE_WINDOW", lambda: on_quit(root))
    root.mainloop()

if __name__ == "__main__":
    main()