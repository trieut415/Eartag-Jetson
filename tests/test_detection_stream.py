import os
import shutil
import cv2
from ultralytics import YOLO

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

# Define paths
BASE_DIR      = find_project_root()
RESOURCES_DIR = os.path.join(BASE_DIR, "resources")
model_path    = os.path.join(RESOURCES_DIR, "detection_model.pt")
engine_path   = os.path.join(RESOURCES_DIR, "detection_model.engine")

# Load and export to TensorRT if needed
model = YOLO(model_path)
if not os.path.exists(engine_path):
    print("[INFO] Exporting model to TensorRT engine…")
    export_result = model.export(format="engine")
    exported_engine = export_result.get("engine")
    if exported_engine and exported_engine != engine_path:
        shutil.move(exported_engine, engine_path)
else:
    print(f"[INFO] TensorRT engine already exists at '{engine_path}'. Skipping export.")

# Load the TensorRT model
trt_model = YOLO(engine_path, task="detect")

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)   

# Check actual resolution
actual_width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"[INFO] Camera resolution: {int(actual_width)}x{int(actual_height)}")

# Get max screen resolution (modify if different)
screen_w, screen_h = 1920, 1080

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
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
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
