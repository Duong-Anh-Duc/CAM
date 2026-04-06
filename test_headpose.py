"""Test head pose với Landmark method (đang dùng trong app).
Bấm phím số trên CỬA SỔ CAMERA để đánh dấu động tác.
Nhấn q để thoát.
"""
import cv2
import numpy as np
import mediapipe as mp
import time

ACTIONS = {
    49: "1-NHIN THANG",   # phím 1
    50: "2-CUI DAU",      # phím 2
    51: "3-NGANG DAU",    # phím 3
    52: "4-QUAY TRAI",    # phím 4
    53: "5-QUAY PHAI",    # phím 5
}

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True,
    min_detection_confidence=0.5, min_tracking_confidence=0.5)

def landmark_pose(lms, w, h):
    nose = lms[4]; fore = lms[10]; chin = lms[152]
    lc = lms[234]; rc = lms[454]
    nose_x, nose_y = nose.x*w, nose.y*h
    face_h = max(chin.y*h - fore.y*h, 1)
    face_w = max(rc.x*w - lc.x*w, 1)
    cy = (fore.y*h + chin.y*h) / 2
    cx = (lc.x*w + rc.x*w) / 2
    pitch = (nose_y - cy) / face_h * 180
    yaw = (nose_x - cx) / face_w * 180
    return max(-60, min(60, pitch)), max(-60, min(60, yaw))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

current_action = "CHUA CHON"
log_f = open("headpose_log.txt", "w")
log_f.write("action,landmark_pitch,landmark_yaw\n")
last_log = 0

print("=" * 55)
print("  TEST LANDMARK HEAD POSE")
print("  CLICK VAO CUA SO CAMERA truoc, roi bam phim:")
print("    1 = Nhin thang")
print("    2 = Cui dau")
print("    3 = Ngang dau")
print("    4 = Quay trai")
print("    5 = Quay phai")
print("    q = Thoat")
print("=" * 55)

while True:
    ret, frame = cap.read()
    if not ret: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    lm_p = lm_y = 0.0
    if result.multi_face_landmarks:
        lms = result.multi_face_landmarks[0].landmark
        lm_p, lm_y = landmark_pose(lms, w, h)

    # Hien thi dong tac hien tai (TO, DO)
    cv2.rectangle(frame, (0, 0), (w, 40), (0, 0, 180), -1)
    cv2.putText(frame, f"DONG TAC: {current_action}", (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Hien thi gia tri
    cv2.putText(frame, f"Pitch: {lm_p:+.1f}", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    cv2.putText(frame, f"Yaw:   {lm_y:+.1f}", (10, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    # Huong dan phia duoi
    cv2.rectangle(frame, (0, h-30), (w, h), (50, 50, 50), -1)
    cv2.putText(frame, "CLICK camera window > bam 1-5 chon dong tac | q=thoat",
                (5, h - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

    cv2.imshow("Head Pose Test", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break
    if key in ACTIONS:
        current_action = ACTIONS[key]
        print(f"\n>>> Dong tac: {current_action}")

    # Log moi 0.3s
    now = time.time()
    if now - last_log >= 0.3 and result.multi_face_landmarks:
        line = f"{current_action},{lm_p:+.1f},{lm_y:+.1f}"
        log_f.write(line + "\n"); log_f.flush()
        print(f"  [{current_action:15s}]  Pitch={lm_p:+7.1f}   Yaw={lm_y:+7.1f}")
        last_log = now

cap.release()
cv2.destroyAllWindows()
log_f.close()
print(f"\nLog saved to headpose_log.txt")
