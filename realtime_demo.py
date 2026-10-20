print("Dang khoi dong, vui long cho...")
from ultralytics import YOLO
import cv2
import time

import sys
import os
import torch

if getattr(sys, 'frozen', False):
    base_path = sys._MEIPASS  # thư mục tạm khi chạy từ file exe
else:
    base_path = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(base_path, "best.pt")
model = YOLO(model_path)
def pick_device():
    if not torch.cuda.is_available():
        return "cpu"
    major, minor = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    if (major, minor) < (7, 5):
        print(f"{name} (CC {major}.{minor}) qua cu, chuyen sang CPU")
        return "cpu"
    try:
        torch.zeros(8, device="cuda") + 1
        torch.cuda.synchronize()
        print(f"Dung GPU: {name} (CC {major}.{minor})")
        return 0
    except Exception:
        print("GPU khong tuong thich, chuyen sang CPU")
        return "cpu"

device = pick_device()
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Khong mo duoc webcam!")
    exit()

print("Dang chay real-time... Nhan phim 'q' de thoat.")

prev_time = time.time()
fps_list = []
infer_ms = []
conf_list = []
frames_total = 0
frames_with_det = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model.predict(frame, conf=0.4, device=device, verbose=False)
    frames_total += 1
    infer_ms.append(results[0].speed["inference"])
    boxes = results[0].boxes
    if len(boxes) > 0:
        frames_with_det += 1
        conf_list.extend(boxes.conf.tolist())
    annotated_frame = results[0].plot()

    curr_time = time.time()
    fps = 1 / (curr_time - prev_time)
    prev_time = curr_time
    fps_list.append(fps)

    cv2.putText(annotated_frame, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Phat hien UAV - Real-time", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if fps_list:
    n = frames_total
    print("\n" + "=" * 42)
    print("        KET QUA REAL-TIME")
    print("=" * 42)
    print(f"Thiet bi          : {'GPU' if device == 0 else 'CPU'}")
    print(f"So khung xu ly    : {n}")
    print(f"FPS trung binh    : {sum(fps_list) / len(fps_list):.2f}")
    print(f"FPS thap / cao    : {min(fps_list):.2f} / {max(fps_list):.2f}")
    print(f"Do tre inference  : {sum(infer_ms) / len(infer_ms):.1f} ms/khung")
    print(f"Ti le co phat hien: {frames_with_det / n * 100:.1f} %")
    if conf_list:
        print(f"Conf trung binh   : {sum(conf_list) / len(conf_list):.3f}")
    print("=" * 42)
    input("Nhan Enter de dong...")