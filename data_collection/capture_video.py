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
video_dir     = os.path.join(BASE_DIR,"data_collection")
video_dir     = os.path.join(video_dir,"saved_videos")
os.makedirs(video_dir, exist_ok=True)

# Auto-increment video filename
i = 1
while True:
    output_path = os.path.join(video_dir, f"video{i}.avi")
    if not os.path.exists(output_path):
        break
    i += 1

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

# Confirm actual resolution
actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps           = int(cap.get(cv2.CAP_PROP_FPS)) or 20
print(f"[INFO] Camera resolution: {actual_width}x{actual_height}, FPS: {fps}")

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')  # or 'MJPG' or 'mp4v' for .mp4
out = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))

# Get screen resolution (for display only)
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

    # Write to video file
    out.write(annotated)

    # Resize for display
    h, w = annotated.shape[:2]
    scale = min(1.0, screen_w / w, screen_h / h)
    display_frame = cv2.resize(annotated, (int(w * scale), int(h * scale)))

    # Display
    cv2.imshow("Detection Inference", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Annotated video saved to: {output_path}")
