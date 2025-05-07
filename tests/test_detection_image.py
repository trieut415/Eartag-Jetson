import os
import sys
import shutil
import cv2
from ultralytics import YOLO
import termios

def find_project_root():
    cur = os.path.abspath(os.getcwd())
    while True:
        if (os.path.isdir(os.path.join(cur, "resources"))
            and os.path.isfile(os.path.join(cur, "requirements.txt"))):
            return cur
        parent = os.path.dirname(cur)
        if parent == cur:
            raise RuntimeError("Could not locate project root")
        cur = parent

BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")

model_path  = os.path.join(RESOURCES_DIR, "detection_model.pt")
engine_path = os.path.join(RESOURCES_DIR, "detection_model.engine")
test_image  = os.path.join(RESOURCES_DIR, "cow_image_test.jpg")


model = YOLO(model_path)
if not os.path.exists(engine_path):
    print("[INFO] Exporting model to TensorRT engine…")
    export_result = model.export(format="engine")
    exported_engine = export_result.get("engine")
    if exported_engine and exported_engine != engine_path:
        shutil.move(exported_engine, engine_path)
else:
    print(f"[INFO] TensorRT engine already exists at '{engine_path}'. Skipping export.")

trt_model = YOLO(engine_path, task="detect")
results   = trt_model(test_image)
print("Inference completed, plotting…")

annotated = results[0].plot()
resized   = cv2.resize(annotated, (1920, 1080))
win       = "Detected Eartags"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.imshow(win, resized)

termios.tcflush(sys.stdin, termios.TCIFLUSH)
print("\nTo exit, press Ctrl+C.")
while True:
    key = cv2.waitKey(10)
    if key != -1:
        print(f"[DEBUG] got keycode: {key}")
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
