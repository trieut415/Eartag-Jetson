import os
import cv2
from eartag_jetson.common.common_utils import find_project_root


# Define paths
BASE_DIR   = find_project_root()
video_dir  = os.path.join(BASE_DIR, "data_collection", "saved_videos")
os.makedirs(video_dir, exist_ok=True)

# Auto-increment video filename
i = 1
while True:
    output_path = os.path.join(video_dir, f"video{i}.avi")
    if not os.path.exists(output_path):
        break
    i += 1

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)  
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)

# Confirm actual resolution and FPS
actual_width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps           = int(cap.get(cv2.CAP_PROP_FPS)) or 20
print(f"[INFO] Camera resolution: {actual_width}x{actual_height}, FPS: {fps}")

# Define video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (actual_width, actual_height))

# Get screen resolution (for display only)
screen_w, screen_h = 1920, 1080

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    # Write raw frame to video file
    out.write(frame)

    # Resize for display
    h, w = frame.shape[:2]
    scale = min(1.0, screen_w / w, screen_h / h)
    display_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))

    # Display
    cv2.imshow("Raw Camera Feed", display_frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"[INFO] Video saved to: {output_path}")
