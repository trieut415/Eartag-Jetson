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
save_dir      = os.path.join(BASE_DIR, "data_collection")
save_dir      = os.path.join(save_dir, "saved_frames")
os.makedirs(save_dir, exist_ok=True)

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

ret, frame = cap.read()
if not ret:
    print("Failed to capture image.")
else:
    # Run inference
    results = trt_model(frame)

    # Annotate and save result
    annotated = results[0].plot()
    # Auto-increment filename logic
    i = 1
    while True:
        save_path = os.path.join(save_dir, f"frame{i}.jpg")
        if not os.path.exists(save_path):
            break
        i += 1

    cv2.imwrite(save_path, annotated)
    print(f"[INFO] Annotated image saved to {save_path}")


# Clean up
cap.release()
cv2.destroyAllWindows()
