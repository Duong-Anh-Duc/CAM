"""
=============================================================================
Hệ thống giám sát tập trung học sinh  |  Giao diện giám sát
=============================================================================
Tích hợp camera + BehaviorDetector vào 1 cửa sổ Tkinter duy nhất.
  * Video preview realtime (trái)
  * Dashboard cảnh báo + metrics (phải)
  * Điều khiển: Start / Stop / YOLO on-off / Screenshot / Ghi video

Chạy:
    python app_gui.py
=============================================================================
"""

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from PIL import Image, ImageTk
import cv2
import numpy as np
import time
import os
import sys
import threading
from datetime import datetime, timedelta
from collections import Counter, deque

# ── Import BehaviorDetector ───────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from behavior_detector import (
    BehaviorDetector, AlertEvent, FaceDatabase,
    YOLO_AVAILABLE, DLIB_AVAILABLE, AUDIO_LIB,
    SEVERITY_LOW, SEVERITY_MEDIUM, SEVERITY_HIGH, SEVERITY_CRITICAL,
)

# Nhãn tiếng Việt cho dashboard
VN_LABELS = {
    "DROWSY"      : "Ngủ gật",
    "MICROSLEEP"  : "Ngủ sâu",
    "YAWNING"     : "Ngáp",
    "HEAD_DOWN"   : "Cúi đầu",
    "HEAD_UP"     : "Ngẩng đầu",
    "HEAD_TURN"   : "Quay đầu",
    "PHONE_USE"   : "Điện thoại",
    "DISTRACTED"  : "Mất tập trung",
    "NO_FACE"     : "Không thấy mặt",
    "MULTI_PERSON": "Nhiều người",
    "FATIGUE_HIGH": "Mệt mỏi nghiêm trọng",
    "NORMAL"      : "Bình thường",
}


# =============================================================================
# THEME / COLORS
# =============================================================================
LIGHT = {
    "bg":         "#f4f4f8",
    "fg":         "#1e1e2e",
    "panel_bg":   "#ffffff",
    "accent":     "#4361ee",
    "ok":         "#2ecc71",
    "warn":       "#f39c12",
    "danger":     "#e74c3c",
    "critical":   "#c0392b",
    "muted":      "#888888",
    "border":     "#ddd",
    "card_bg":    "#f9f9f9",
}
DARK = {
    "bg":         "#1e1e2e",
    "fg":         "#dcdcdc",
    "panel_bg":   "#2a2a3c",
    "accent":     "#7c8fef",
    "ok":         "#2ecc71",
    "warn":       "#f39c12",
    "danger":     "#e74c3c",
    "critical":   "#c0392b",
    "muted":      "#666666",
    "border":     "#444",
    "card_bg":    "#333346",
}


# =============================================================================
# MAIN APPLICATION
# =============================================================================
class CAMApp:
    """Hệ thống giám sát tập trung học sinh – Tkinter GUI tích hợp camera"""

    WINDOW_TITLE = "Hệ thống giám sát tập trung học sinh"
    VIDEO_W = 720          # chiều rộng video preview
    VIDEO_H = 480          # chiều cao
    FPS_TARGET = 25

    def __init__(self):
        self.root = tk.Tk()
        self.root.title(self.WINDOW_TITLE)
        self.root.minsize(1100, 620)
        self.root.protocol("WM_DELETE_WINDOW", self._on_quit)

        # ── State ─────────────────────────────────────────────────────────
        self.running       = False
        self.cap           = None
        self.detector      = None
        self.use_yolo      = tk.BooleanVar(value=YOLO_AVAILABLE)
        self.show_lm       = tk.BooleanVar(value=True)
        self.dark_mode     = False
        self.theme         = LIGHT
        self.frame_count   = 0
        self.fps           = 0.0
        self._fps_hist     = deque(maxlen=30)  # rolling FPS
        self._fps_t        = time.time()
        self.alert_log: list[AlertEvent] = []
        self._recording    = False
        self._vid_writer   = None
        self._session_start = None
        self._status_clear_time = 0.0

        # ── Build UI ──────────────────────────────────────────────────────
        self._build_toolbar()
        self._build_body()
        self._build_statusbar()
        self._apply_theme()

        # Initial photo placeholder
        self._show_placeholder()

    # =====================================================================
    # UI BUILDING
    # =====================================================================
    def _build_toolbar(self):
        """Thanh công cụ phía trên"""
        self.toolbar = tk.Frame(self.root, height=44)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.btn_start = tk.Button(self.toolbar, text="Bắt đầu giám sát",
                                   command=self._toggle_run, width=22,
                                   font=("Helvetica", 12, "bold"))
        self.btn_start.pack(side=tk.LEFT, padx=8, pady=6)

        self.chk_yolo = tk.Checkbutton(self.toolbar, text="YOLO (person/phone)",
                                       variable=self.use_yolo,
                                       font=("Helvetica", 10))
        self.chk_yolo.pack(side=tk.LEFT, padx=6)

        self.chk_lm = tk.Checkbutton(self.toolbar, text="Landmarks",
                                     variable=self.show_lm,
                                     font=("Helvetica", 10))
        self.chk_lm.pack(side=tk.LEFT, padx=6)

        self.btn_screenshot = tk.Button(self.toolbar, text="Chụp màn hình",
                                        command=self._screenshot,
                                        font=("Helvetica", 10))
        self.btn_screenshot.pack(side=tk.LEFT, padx=6)

        self.btn_record = tk.Button(self.toolbar, text="Ghi video",
                                     command=self._toggle_record,
                                     font=("Helvetica", 10))
        self.btn_record.pack(side=tk.LEFT, padx=6)

        self.btn_register = tk.Button(self.toolbar, text="Đăng ký học sinh",
                                      command=self._register_face,
                                      font=("Helvetica", 10), fg="#4361ee")
        self.btn_register.pack(side=tk.LEFT, padx=6)

        self.btn_theme = tk.Button(self.toolbar, text="Theme",
                                   command=self._toggle_theme, width=5,
                                   font=("Helvetica", 12))
        self.btn_theme.pack(side=tk.RIGHT, padx=8, pady=6)

        self.btn_quit_toolbar = tk.Button(self.toolbar, text="Thoát",
                                          command=self._on_quit,
                                          font=("Helvetica", 11, "bold"))
        self.btn_quit_toolbar.pack(side=tk.RIGHT, padx=4, pady=6)

    def _build_body(self):
        """Nội dung chính: video + dashboard"""
        self.body = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashwidth=4)
        self.body.pack(fill=tk.BOTH, expand=True, padx=4, pady=2)

        # ── Video panel (trái) ────────────────────────────────────────────
        self.video_frame = tk.Frame(self.body, bg="black")
        self.body.add(self.video_frame, stretch="always")

        self.canvas = tk.Label(self.video_frame, bg="black")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # ── Dashboard panel (phải) ────────────────────────────────────────
        self.dash_frame = tk.Frame(self.body, width=340)
        self.body.add(self.dash_frame, stretch="never", width=340)

        # --- Info header ---
        self.lbl_info_title = tk.Label(self.dash_frame,
            text="Dashboard", font=("Helvetica", 14, "bold"),
            anchor="w")
        self.lbl_info_title.pack(fill=tk.X, padx=10, pady=(8, 2))

        # --- Status cards ---
        self.cards_frame = tk.Frame(self.dash_frame)
        self.cards_frame.pack(fill=tk.X, padx=10, pady=4)

        self.lbl_status = tk.Label(self.cards_frame,
            text="Chưa bắt đầu", font=("Helvetica", 12, "bold"),
            anchor="w")
        self.lbl_status.pack(fill=tk.X, pady=2)

        self.lbl_fps = tk.Label(self.cards_frame,
            text="FPS: --", font=("Helvetica", 10), anchor="w")
        self.lbl_fps.pack(fill=tk.X)

        self.lbl_faces = tk.Label(self.cards_frame,
            text="Học sinh phát hiện: 0", font=("Helvetica", 10), anchor="w")
        self.lbl_faces.pack(fill=tk.X)

        self.lbl_yolo_info = tk.Label(self.cards_frame,
            text="YOLO: --  |  Phones: --", font=("Helvetica", 10), anchor="w")
        self.lbl_yolo_info.pack(fill=tk.X)

        self.lbl_session = tk.Label(self.cards_frame,
            text="Thời gian: 00:00", font=("Helvetica", 10), anchor="w")
        self.lbl_session.pack(fill=tk.X)

        sep1 = ttk.Separator(self.dash_frame, orient="horizontal")
        sep1.pack(fill=tk.X, padx=10, pady=6)

        # --- Student metrics ---
        self.lbl_metrics_title = tk.Label(self.dash_frame,
            text="Chi tiết học sinh", font=("Helvetica", 12, "bold"),
            anchor="w")
        self.lbl_metrics_title.pack(fill=tk.X, padx=10, pady=2)

        self.metrics_text = tk.Text(self.dash_frame, height=8,
            font=("Courier", 10), wrap=tk.WORD, state=tk.DISABLED)
        self.metrics_text.pack(fill=tk.X, padx=10, pady=2)

        sep2 = ttk.Separator(self.dash_frame, orient="horizontal")
        sep2.pack(fill=tk.X, padx=10, pady=6)

        # --- Alert log ---
        self.lbl_alert_title = tk.Label(self.dash_frame,
            text="Cảnh báo gần đây", font=("Helvetica", 12, "bold"),
            anchor="w")
        self.lbl_alert_title.pack(fill=tk.X, padx=10, pady=2)

        self.alert_listbox = tk.Listbox(self.dash_frame, height=10,
            font=("Helvetica", 10), selectmode=tk.NONE)
        self.alert_listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=4)

        # --- Buttons dưới ---
        btn_frame = tk.Frame(self.dash_frame)
        btn_frame.pack(fill=tk.X, padx=10, pady=4)

        self.btn_clear_log = tk.Button(btn_frame, text="Xóa log",
                                       command=self._clear_log,
                                       font=("Helvetica", 10))
        self.btn_clear_log.pack(side=tk.LEFT, padx=2)

        self.btn_reset = tk.Button(btn_frame, text="Reset",
                                   command=self._reset_states,
                                   font=("Helvetica", 10))
        self.btn_reset.pack(side=tk.LEFT, padx=2)

    def _build_statusbar(self):
        """Thanh trạng thái dưới cùng"""
        self.statusbar = tk.Label(self.root,
            text="Hệ thống giám sát tập trung học sinh  |  Nhấn 'Bắt đầu giám sát' để chạy",
            anchor="w", font=("Helvetica", 10), padx=8, pady=3)
        self.statusbar.pack(side=tk.BOTTOM, fill=tk.X)

    # =====================================================================
    # THEME
    # =====================================================================
    def _apply_theme(self):
        t = self.theme
        bg, fg = t["bg"], t["fg"]
        panel = t["panel_bg"]

        self.root.configure(bg=bg)
        for w in (self.toolbar, self.body, self.dash_frame,
                  self.cards_frame):
            w.configure(bg=bg)
        for w in (self.lbl_info_title, self.lbl_status, self.lbl_fps,
                  self.lbl_faces, self.lbl_yolo_info, self.lbl_session,
                  self.lbl_metrics_title, self.lbl_alert_title):
            w.configure(bg=bg, fg=fg)

        # Toolbar buttons
        btn_bg = panel
        for w in (self.btn_start, self.btn_screenshot, self.btn_record,
                  self.btn_theme, self.btn_quit_toolbar,
                  self.btn_clear_log, self.btn_reset):
            w.configure(bg=btn_bg, fg=fg, activebackground=t["accent"],
                        activeforeground="#fff", highlightthickness=0,
                        bd=1, relief="flat")

        # Checkbuttons
        for w in (self.chk_yolo, self.chk_lm):
            w.configure(bg=bg, fg=fg, activebackground=bg,
                        selectcolor=bg, highlightthickness=0)

        # Text / Listbox
        self.metrics_text.configure(bg=t["card_bg"], fg=fg,
                                    insertbackground=fg, bd=1,
                                    relief="flat")
        self.alert_listbox.configure(bg=t["card_bg"], fg=fg, bd=1,
                                     relief="flat", highlightthickness=0)
        self.statusbar.configure(bg=t["accent"], fg="#fff")

        self.video_frame.configure(bg="#000")
        self.canvas.configure(bg="#000")

    def _toggle_theme(self):
        self.dark_mode = not self.dark_mode
        self.theme = DARK if self.dark_mode else LIGHT
        self.btn_theme.configure(text="Light" if self.dark_mode else "Dark")
        self._apply_theme()

    # =====================================================================
    # CAMERA + DETECTION LOOP
    # =====================================================================
    def _toggle_run(self):
        if self.running:
            self._stop()
        else:
            self._start()

    def _start(self):
        """Bắt đầu giám sát"""
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Không mở được camera.\n"
                                 "Hãy kiểm tra webcam đã kết nối chưa.")
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Build detector
        self.detector = BehaviorDetector(
            camera_id=0,
            use_yolo=self.use_yolo.get(),
            show_landmarks=self.show_lm.get(),
            show_metrics=True,
            record_output=False,
        )

        self.running = True
        self._session_start = datetime.now()
        self.frame_count = 0
        self.btn_start.configure(text="Dừng giám sát")
        self.lbl_status.configure(text="Đang giám sát...",
                                  fg=self.theme["ok"])
        self.statusbar.configure(text="Đang phân tích hành vi học sinh...")

        self._update_frame()

    def _stop(self):
        """Dừng giám sát"""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        if self._vid_writer:
            self._vid_writer.release()
            self._vid_writer = None
            self._recording = False
            self.btn_record.configure(text="Ghi video")

        self.btn_start.configure(text="Bắt đầu giám sát")
        self.lbl_status.configure(text="Đã dừng",
                                  fg=self.theme["muted"])
        self.statusbar.configure(text="Đã dừng. Nhấn 'Bắt đầu giám sát' để chạy lại.")
        self._show_placeholder()

        if self.detector:
            self.detector.audio.stop()

    def _update_frame(self):
        """Đọc 1 frame, xử lý, hiển thị, rồi schedule frame tiếp"""
        if not self.running or not self.cap:
            return

        ret, frame = self.cap.read()
        if not ret or frame is None:
            self.root.after(30, self._update_frame)
            return

        # Update detector settings dynamically
        self.detector.show_landmarks = self.show_lm.get()

        # Process
        result = self.detector.process_frame(frame)
        processed       = result['frame']
        log_alerts      = result['alerts']           # đã cooldown – cho log
        display_alerts  = result.get('display_alerts', log_alerts)  # realtime – cho banner

        # Rolling FPS (mượt hơn so với instant FPS)
        self.frame_count += 1
        now_t = time.time()
        dt = now_t - self._fps_t
        self._fps_t = now_t
        self._fps_hist.append(dt)
        if self._fps_hist:
            avg_dt = sum(self._fps_hist) / len(self._fps_hist)
            self.fps = 1.0 / max(avg_dt, 1e-6)

        # Record
        if self._recording and self._vid_writer:
            self._vid_writer.write(processed)

        # Convert BGR→RGB→PIL→Tk
        rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        cw = self.video_frame.winfo_width()
        ch = self.video_frame.winfo_height()
        if cw > 10 and ch > 10:
            pil = self._resize_keep_ratio(pil, cw, ch)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

        # Dashboard dùng display_alerts (realtime, không cooldown)
        self._update_dashboard(result, display_alerts)

        # Log dùng log_alerts (đã cooldown)
        for a in log_alerts:
            self.alert_log.append(a)
            self._add_alert_to_list(a)

        # Schedule next
        delay = max(1, int(1000 / self.FPS_TARGET - dt * 1000))
        self.root.after(delay, self._update_frame)

    @staticmethod
    def _resize_keep_ratio(pil_img, max_w, max_h):
        w, h = pil_img.size
        ratio = min(max_w / w, max_h / h)
        nw, nh = int(w * ratio), int(h * ratio)
        return pil_img.resize((nw, nh), Image.LANCZOS)

    def _show_placeholder(self):
        """Hiển thị ảnh placeholder khi chưa chạy camera"""
        placeholder = np.zeros((self.VIDEO_H, self.VIDEO_W, 3), dtype=np.uint8)
        placeholder[:] = (30, 30, 46)
        # Text
        cv2.putText(placeholder, "Hệ thống giám sát tập trung học sinh",
                    (80, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (200, 200, 200), 2, cv2.LINE_AA)
        cv2.putText(placeholder, "Nhấn 'Bắt đầu giám sát' để chạy camera",
                    (100, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (120, 120, 120), 1, cv2.LINE_AA)

        rgb = cv2.cvtColor(placeholder, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=pil)
        self.canvas.imgtk = imgtk
        self.canvas.configure(image=imgtk)

    # =====================================================================
    # DASHBOARD UPDATES
    # =====================================================================
    def _update_dashboard(self, result, alerts):
        """Cập nhật tất cả thông số trên dashboard"""
        t = self.theme

        # FPS
        self.lbl_fps.configure(text=f"FPS: {self.fps:.1f}")

        # Faces
        n_faces = result.get('faces', 0)
        self.lbl_faces.configure(text=f"Học sinh phát hiện: {n_faces}")

        # YOLO info
        n_p = result.get('persons', 0)
        n_ph = result.get('phones', 0)
        self.lbl_yolo_info.configure(
            text=f"YOLO persons: {n_p}  |  Phones: {n_ph}")

        # Session time
        if self._session_start:
            elapsed = datetime.now() - self._session_start
            mm = int(elapsed.total_seconds()) // 60
            ss = int(elapsed.total_seconds()) % 60
            self.lbl_session.configure(text=f"Thời gian: {mm:02d}:{ss:02d}")

        # Status color
        if alerts:
            severity_order = {SEVERITY_LOW: 0, SEVERITY_MEDIUM: 1,
                              SEVERITY_HIGH: 2, SEVERITY_CRITICAL: 3}
            worst = max(alerts, key=lambda a: severity_order.get(a.severity, 0))
            vn = VN_LABELS.get(worst.behavior_type, worst.behavior_type)
            pid_str = f"HS #{worst.person_id + 1}" if worst.person_id >= 0 else ""
            if worst.severity == SEVERITY_CRITICAL:
                self.lbl_status.configure(
                    text=f"NGUY HIỂM  {pid_str} {vn}",
                    fg=t["critical"])
            elif worst.severity == SEVERITY_HIGH:
                self.lbl_status.configure(
                    text=f"CẢNH BÁO  {pid_str} {vn}",
                    fg=t["danger"])
            elif worst.severity == SEVERITY_MEDIUM:
                self.lbl_status.configure(
                    text=f"CHÚ Ý  {pid_str} {vn}",
                    fg=t["warn"])
            else:
                self.lbl_status.configure(text="Đang giám sát...",
                                          fg=t["ok"])
            # Giữ trạng thái hiển thị 12 giây sau cảnh báo cuối
            self._status_clear_time = time.time() + 12.0
        elif time.time() < self._status_clear_time:
            # Vẫn trong thời gian giữ trạng thái – không reset
            pass
        else:
            self.lbl_status.configure(text="Bình thường",
                                      fg=t["ok"])

        # Student metrics
        self._update_metrics_text()

    def _update_metrics_text(self):
        """Cập nhật text box chi tiết từng học sinh"""
        if not self.detector:
            return
        lines = []
        for pid, st in self.detector.behavior.states.items():
            if not st.face_visible and (time.time() - st.last_face_seen > 10):
                continue
            status = "[V] " if st.face_visible else "[X]"
            pname = self.detector.get_person_name(pid) if self.detector else f"Học sinh #{pid + 1}"
            lines.append(f"{status} {pname}")
            lines.append(f"   EAR: {st.smooth_ear:.3f}  "
                         f"MAR: {st.smooth_mar:.3f}")
            lines.append(f"   Pitch: {st.smooth_pitch:.0f}° "
                         f"Yaw: {st.smooth_yaw:.0f}°")
            lines.append(f"   Blinks: {st.total_blinks}  "
                         f"Yawns: {st.yawn_count}")
            lines.append("")

        if not lines:
            lines = ["Chưa phát hiện khuôn mặt..."]

        self.metrics_text.configure(state=tk.NORMAL)
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert(tk.END, "\n".join(lines))
        self.metrics_text.configure(state=tk.DISABLED)

    def _add_alert_to_list(self, alert: AlertEvent):
        """Thêm alert vào listbox với nhãn tiếng Việt + tên học sinh"""
        t = alert.timestamp.strftime("%H:%M:%S")
        prefix = {"LOW": "[OK]", "MEDIUM": "[!!]",
                  "HIGH": "[!!]", "CRITICAL": "[XX]"}.get(alert.severity, "[--]")
        vn = VN_LABELS.get(alert.behavior_type, alert.behavior_type)
        if alert.person_id >= 0 and self.detector:
            pid_str = self.detector.get_person_name(alert.person_id)
        elif alert.person_id >= 0:
            pid_str = f"Học sinh #{alert.person_id + 1}"
        else:
            pid_str = ""
        text = f"{prefix} [{t}] {pid_str} - {vn}: {alert.message}"
        self.alert_listbox.insert(0, text)
        # Giới hạn 200 dòng
        while self.alert_listbox.size() > 200:
            self.alert_listbox.delete(tk.END)

    # =====================================================================
    # ACTIONS
    # =====================================================================
    def _register_face(self):
        """Đăng ký khuôn mặt học sinh live từ camera."""
        if not self.running or not self.detector:
            messagebox.showinfo("Thông báo",
                "Cần bắt đầu giám sát trước khi đăng ký.")
            return

        # Nhập tên
        name = simpledialog.askstring("Đăng ký học sinh",
            "Nhập tên học sinh (nhìn thẳng vào camera):",
            parent=self.root)
        if not name or not name.strip():
            return
        name = name.strip()

        # Chụp frame hiện tại
        if not self.cap or not self.cap.isOpened():
            messagebox.showerror("Lỗi", "Camera chưa mở.")
            return

        ret, frame = self.cap.read()
        if not ret:
            messagebox.showerror("Lỗi", "Không đọc được frame từ camera.")
            return

        # Đăng ký
        success = self.detector.register_face(frame, name)
        if success:
            count = len(self.detector.face_db.entries.get(name, []))
            messagebox.showinfo("Thành công",
                f"Đã đăng ký '{name}' ({count} mẫu).\n"
                f"Có thể đăng ký thêm để tăng độ chính xác.")
            self.statusbar.configure(text=f"Đã đăng ký: {name}")
        else:
            messagebox.showwarning("Không thành công",
                "Không phát hiện khuôn mặt.\n"
                "Hãy nhìn thẳng vào camera và thử lại.")

    def _screenshot(self):
        if not self.running:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu giám sát.")
            return
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        fname = f"screenshot_{ts}.jpg"
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                cv2.imwrite(fname, frame)
                self.statusbar.configure(text=f"Đã lưu: {fname}")

    def _toggle_record(self):
        if not self.running:
            messagebox.showinfo("Thông báo", "Chưa bắt đầu giám sát.")
            return
        if self._recording:
            # Stop recording
            if self._vid_writer:
                self._vid_writer.release()
                self._vid_writer = None
            self._recording = False
            self.btn_record.configure(text="Ghi video")
            self.statusbar.configure(text="Đã dừng ghi video.")
        else:
            # Start recording
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"recording_{ts}.avi"
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fourcc = cv2.VideoWriter_fourcc(*'MJPG')
            self._vid_writer = cv2.VideoWriter(fname, fourcc, 15, (w, h))
            self._recording = True
            self.btn_record.configure(text="Dừng ghi")
            self.statusbar.configure(text=f"Đang ghi video: {fname}")

    def _clear_log(self):
        self.alert_listbox.delete(0, tk.END)
        self.alert_log.clear()

    def _reset_states(self):
        if self.detector:
            self.detector.behavior.states.clear()
            self.detector._last_alert_time.clear()
            self.detector._alert_active.clear()
            self.detector.all_alerts.clear()
            self.detector._alarm_clear_counter = 0
            # Reset face tracker để gán lại person-id từ đầu
            from behavior_detector import FaceTracker
            self.detector.tracker = FaceTracker()
        self._fps_hist.clear()
        self._status_clear_time = 0.0
        self._clear_log()
        self.statusbar.configure(text="Đã reset trạng thái.")

    def _on_quit(self):
        self._stop()
        self.root.destroy()

    # =====================================================================
    # RUN
    # =====================================================================
    def run(self):
        self.root.mainloop()


# =============================================================================
# ENTRY POINT
# =============================================================================
if __name__ == "__main__":
    app = CAMApp()
    app.run()
