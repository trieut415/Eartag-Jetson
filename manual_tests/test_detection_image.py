import os
import sys
import shutil
import cv2
import termios
from ultralytics import YOLO
from eartag_jetson.common.common_utils import get_logger, find_project_root

logger = get_logger(__name__)

def main():
    BASE_DIR      = find_project_root()
    RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")

    model_path  = os.path.join(RESOURCES_DIR, "detection_model.pt")
    engine_path = os.path.join(RESOURCES_DIR, "detection_model.engine")
    test_image  = os.path.join(
        BASE_DIR,
        "src", "eartag_jetson", "data_collection",
        "saved_frames", "frame13.jpg"
    )

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

    trt_model = YOLO(engine_path, task="detect")
    results   = trt_model(test_image)
    logger.info("Inference completed, plotting…")

    annotated = results[0].plot()
    resized   = cv2.resize(annotated, (1920, 1080))
    win       = "Detected Eartags"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.imshow(win, resized)

    termios.tcflush(sys.stdin, termios.TCIFLUSH)
    logger.info("To exit, press ESC.")
    while True:
        key = cv2.waitKey(10)
        if key != -1:
            logger.debug(f"got keycode: {key}")
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
