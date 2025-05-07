import os
import cv2

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
save_dir      = os.path.join(BASE_DIR, "data_collection", "saved_frames")
os.makedirs(save_dir, exist_ok=True)

# Start video capture
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)

ret, frame = cap.read()
if not ret:
    print("Failed to capture image.")
else:
    # Auto-increment filename logic
    i = 1
    while True:
        save_path = os.path.join(save_dir, f"frame{i}.jpg")
        if not os.path.exists(save_path):
            break
        i += 1

    cv2.imwrite(save_path, frame)
    print(f"[INFO] Image saved to {save_path}")

# Clean up
cap.release()
cv2.destroyAllWindows()
