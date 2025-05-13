import os
import shutil
import cv2
import time
import logging
from ultralytics import YOLO
from eartag_jetson.common.common_utils import get_logger, find_project_root

logger = get_logger(__name__)

# ——— set up absolute paths ———
BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")
model_path    = os.path.join(RESOURCES_DIR, "detection_model.pt")
engine_path   = os.path.join(RESOURCES_DIR, "detection_model.engine")
video_path    = os.path.join(RESOURCES_DIR, "cow_video_test.mp4")
# ————————————————————————

# load & export if needed
model = YOLO(model_path)
if not os.path.exists(engine_path):
    logger.info("Exporting model to TensorRT engine…")
    export_result = model.export(format="engine")
    exported_engine = export_result.get("engine")
    if exported_engine and exported_engine != engine_path:
        shutil.move(exported_engine, engine_path)
        logger.info(f"Moved exported engine to '{engine_path}'")
else:
    logger.info(f"TensorRT engine already exists at '{engine_path}'. Skipping export.")

# load the TensorRT engine
trt_model = YOLO(engine_path, task="detect")

# open video file
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    logger.error(f"Could not open video: {video_path}")
    raise RuntimeError(f"Could not open video: {video_path}")

# grab original fps 
fps = cap.get(cv2.CAP_PROP_FPS) or 30
delay = int(1000 / fps)

# get video resolution for optional resizing
screen_w, screen_h = 480, 480

logger.info(f"Playing '{video_path}' at {fps:.2f} FPS")

while True:
    ret, frame = cap.read()
    if not ret:
        logger.info("End of video reached")
        break

    # run inference
    results = trt_model(frame)

    # annotate
    annotated = results[0].plot()

    # resize if larger than screen
    h, w = annotated.shape[:2]
    scale = min(1.0, screen_w / w, screen_h / h)
    if scale < 1.0:
        annotated = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    # show
    cv2.imshow("Video Detection", annotated)

    # wait according to fps, exit on 'q'
    if cv2.waitKey(delay) & 0xFF == ord('q'):
        logger.info("Playback interrupted by user")
        break

# cleanup
cap.release()
cv2.destroyAllWindows()
logger.info("Released video capture and destroyed all windows")
