import os
import shutil
import cv2
from ultralytics import YOLO
from eartag_jetson.common.common_utils import get_logger, find_project_root

logger = get_logger(__name__)

# Define paths
BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson","resources")
model_path    = os.path.join(RESOURCES_DIR, "detection_model.pt")
engine_path   = os.path.join(RESOURCES_DIR, "detection_model.engine")

# Load and export to TensorRT if needed
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

# Load the TensorRT model
trt_model = YOLO(engine_path, task="detect")

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)

# Check actual resolution and fps
actual_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
logger.info(f"Camera resolution: {int(actual_width)}x{int(actual_height)}")
nominal_fps = cap.get(cv2.CAP_PROP_FPS)
logger.info(f"Camera-reported FPS: {nominal_fps:.2f}")
# Get max screen resolution (modify if different)
screen_w, screen_h = 1920, 1080

while True:
    ret, frame = cap.read()
    if not ret:
        logger.error("Failed to grab frame")
        break

    # Run inference
    results = trt_model(frame)

    # Annotate result
    annotated = results[0].plot()

    # Dynamically resize to fit screen
    h, w = annotated.shape[:2]
    scale_w = screen_w / w
    scale_h = screen_h / h
    scale = min(1.0, scale_w, scale_h)  # Don't upscale
    resized = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    # Display
    cv2.imshow("Detection Inference", resized)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        logger.info("Quitting inference loop")
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
logger.info("Released camera and destroyed all windows")
