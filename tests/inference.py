import os
import sys
import shutil
import cv2
from ultralytics import YOLO
import termios

# Load & export (unchanged) …
model_path = "./resources/detection_model.pt"
model = YOLO(model_path)
engine_path = "./resources/detection_model.engine"
if not os.path.exists(engine_path):
    print("[INFO] Exporting model to TensorRT engine…")
    export_result = model.export(format="engine")
    exported_engine = export_result.get("engine")
    if exported_engine and exported_engine != engine_path:
        shutil.move(exported_engine, engine_path)
else:
    print(f"[INFO] TensorRT engine already exists at '{engine_path}'. Skipping export.")

trt_model = YOLO(engine_path)
results = trt_model("./resources/cow_image_test.jpg")
print("Inference completed, plotting…")

# Annotate & resize
annotated = results[0].plot()
resized = cv2.resize(annotated, (640, 480))

# Make a named, resizable window and show
win = "Detected Eartags"
cv2.namedWindow(win, cv2.WINDOW_NORMAL)
cv2.imshow(win, resized)

# Flush any stray key events (so previous ESC presses don’t linger)
termios.tcflush(sys.stdin, termios.TCIFLUSH)
print("")
print("To exit, press Ctrl+C to exit.")
while True:
    key = cv2.waitKey(10)  
    if key != -1:
        print(f"[DEBUG] got keycode: {key}")
    if key == 27:
        break

cv2.destroyAllWindows()
