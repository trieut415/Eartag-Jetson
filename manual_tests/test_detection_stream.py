import os
import cv2
from ultralytics import YOLO
from eartag_jetson.common.common_utils import get_logger, find_project_root, export_yolo_to_engine

logger = get_logger(__name__)

# ─── DEFINE PATHS ───────────────────────────────────────────────────────────────
BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")
model_path    = os.path.join(RESOURCES_DIR, "detection_model.pt")
engine_path   = os.path.join(RESOURCES_DIR, "detection_model.engine")

# ─── EXPORT TO TENSORRT IF NEEDED ────────────────────────────────────────────────
model_pt = YOLO(model_path)
export_yolo_to_engine(model_pt, engine_path, logger)  # uses your shared helper

# ─── LOAD TRT MODEL ──────────────────────────────────────────────────────────────
trt_model = YOLO(engine_path, task="detect")

# ─── SETUP VIDEO CAPTURE ─────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)

# ─── LOG ACTUAL CAPTURED PROPERTIES ─────────────────────────────────────────────
actual_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
logger.info(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")
nominal_fps = cap.get(cv2.CAP_PROP_FPS)
logger.info(f"Camera-reported FPS: {nominal_fps:.2f}")

# ─── DISPLAY SETTINGS ───────────────────────────────────────────────────────────
screen_w, screen_h = 1920, 1080

# ─── INFERENCE LOOP ─────────────────────────────────────────────────────────────
while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to grab frame")
        break

    results = trt_model(frame)
    annotated = results[0].plot()

    # Resize dynamically to fit screen (downscale only)
    h, w = annotated.shape[:2]
    scale = min(1.0, screen_w / w, screen_h / h)
    resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    cv2.imshow("Detection Inference", resized)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Quitting inference loop")
        break

# ─── CLEANUP ─────────────────────────────────────────────────────────────────────
cap.release()
cv2.destroyAllWindows()
logger.info("Released camera and destroyed all windows")
