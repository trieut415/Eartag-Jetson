#!/usr/bin/env python3
import cv2
import time

def nothing(x):
    pass

def open_camera(idx, width, height):
    # try default backend by index
    cap = cv2.VideoCapture(idx)
    if cap.isOpened():
        return cap

    # fallback: force V4L2 by index
    cap = cv2.VideoCapture(idx, cv2.CAP_V4L2)
    if cap.isOpened():
        return cap

    # fallback: GStreamer pipeline (replace YUY2 if your camera speaks MJPG)
    pipeline = (
        f'v4l2src device=/dev/video{idx} ! '
        f'video/x-raw,format=YUY2,width={width},height={height} ! '
        'videoconvert ! appsink'
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    return cap

def main():
    TARGET_W, TARGET_H = 4608, 2592
    # pick the index you know exists; try 0 then 1
    for cam_idx in (0, 1):
        cap = open_camera(cam_idx, TARGET_W, TARGET_H)
        if cap.isOpened():
            print(f"✔️  Opened /dev/video{cam_idx}")
            break
    else:
        print("❌ Could not open any camera (tried 0 and 1)")
        return

    # request target resolution & read back
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  TARGET_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_H)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"⚙️  Running at {w}×{h}")

    # UI
    cv2.namedWindow("Tune", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tune", 1280, 720)
    cv2.createTrackbar("Left",  "Tune", 500, w//2, nothing)
    cv2.createTrackbar("Right", "Tune", 500, w//2, nothing)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Frame grab failed; exiting.")
                break

            left  = cv2.getTrackbarPos("Left",  "Tune")
            right = w - cv2.getTrackbarPos("Right", "Tune")

            cv2.line(frame, (left,  0), (left,  h), (0, 0, 255), 2)
            cv2.line(frame, (right, 0), (right, h), (0, 0, 255), 2)

            cv2.imshow("Tune", frame)
            if cv2.waitKey(1) & 0xFF == 27:  # ESC
                break

            time.sleep(1/60)
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
