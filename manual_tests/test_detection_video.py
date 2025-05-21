import os
import cv2
import time
import logging
from ultralytics import YOLO
from eartag_jetson.common.common_utils import get_logger, find_project_root, export_yolo_to_engine

logger = get_logger(__name__)

# ─── SET PATHS ───────────────────────────────────────────────────────────────────
BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")
model_path    = os.path.join(RESOURCES_DIR, "seg_model.pt")
engine_path   = os.path.join(RESOURCES_DIR, "seg_model.engine")
video_path    = os.path.join(RESOURCES_DIR, "cow_video_test.mp4")

# ─── LOAD & EXPORT ENGINE IF NEEDED ──────────────────────────────────────────────
model_pt = YOLO(model_path)
export_yolo_to_engine(model_pt, engine_path, logger)  # Uses your common function

# ─── LOAD TRT ENGINE ─────────────────────────────────────────────────────────────
trt_model = YOLO(engine_path, task="detect")

# ─── VIDEO PROCESSING ────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logger.error(f"Could not open video: {video_path}")
    raise RuntimeError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)
screen_w, screen_h = 480, 480

logger.info(f"Playing '{video_path}' at {fps:.2f} FPS")

while True:
    ret, frame = cap.read()
    if not ret:
        logger.info("End of video reached")
        break

    results = trt_model(frame)
    annotated = results[0].plot()

    h, w = annotated.shape[:2]
    scale = min(1.0, screen_w / w, screen_h / h)
    if scale < 1.0:
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    cv2.imshow("Video Detection", annotated)
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        logger.info("Playback interrupted by user")
        break

cap.release()
cv2.destroyAllWindows()
logger.info("Released video capture and destroyed all windows")
