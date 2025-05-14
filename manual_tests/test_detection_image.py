import os
import sys
import cv2
import termios
from ultralytics import YOLO
from eartag_jetson.common.common_utils import (
    get_logger,
    find_project_root,
    export_yolo_to_engine,
)

# ─── SET PATHS ────────────────────────────────────────────────────────────────
BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")
model_path    = os.path.join(RESOURCES_DIR, "detection_model.pt")
engine_path   = os.path.join(RESOURCES_DIR, "detection_model.engine")
test_image    = os.path.join(BASE_DIR, "src", "eartag_jetson", "data_collection", "saved_frames", "frame13.jpg")

# ─── EXPORT TO TENSORRT IF NEEDED ─────────────────────────────────────────────
model_pt = YOLO(model_path)
export_yolo_to_engine(model_pt, engine_path, logger)

# ─── LOAD TRT MODEL & RUN INFERENCE ──────────────────────────────────────────
trt_model = YOLO(engine_path, task="detect")
results = trt_model(test_image)
logger.info("Inference completed, plotting…")

# ─── DISPLAY IMAGE ────────────────────────────────────────────────────────────
annotated = results[0].plot()
resized = cv2.resize(annotated, (1920, 1080))

window_name = "Detected Eartags"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.imshow(window_name, resized)

termios.tcflush(sys.stdin, termios.TCIFLUSH)
logger.info("To exit, press ESC.")

while True:
    key = cv2.waitKey(10)
    if key != -1:
        logger.debug(f"got keycode: {key}")
    if key == 27:  # ESC
        break

cv2.destroyAllWindows()
logger.info("Window closed")

