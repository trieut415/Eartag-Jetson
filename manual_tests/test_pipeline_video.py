import cv2
import logging
import csv
import os
from ultralytics import YOLO
from paddleocr import PaddleOCR
from eartag_jetson.common.common_utils import get_logger, find_project_root
from datetime import datetime

def crop_images(image, detections):
    cropped_images = []
    for _, bbox, _ in detections:
        x0, y0, x1, y1 = bbox
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1 = min(image.shape[1], int(x1))
        y1 = min(image.shape[0], int(y1))
        cropped_images.append(image[y0:y1, x0:x1])
    return cropped_images

if __name__ == "__main__":
    # Paths & resources
    BASE_DIR = find_project_root()
    VIDEO_PATH = os.path.join(BASE_DIR, "src", "eartag_jetson", "data_collection", "saved_videos", "video2_crop.avi")  
    RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")
    DET_MODEL = os.path.join(RESOURCES_DIR, "seg_model.pt")
    CLS_MODEL = os.path.join(RESOURCES_DIR, "cls_model.pt")
    OUTPUT_CSV = os.path.join(BASE_DIR, "eartag_results.csv")

    # Initialize models & OCR
    model = YOLO(DET_MODEL)
    cls_model = YOLO(CLS_MODEL)
    logging.getLogger('ppocr').setLevel(logging.ERROR)
    ocr = PaddleOCR(use_angle_cls=True, lang='en', ocr_version='PP-OCRv4', use_space_char=True)
    CLEAR_THRESHOLD = 0.85

    # Open video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")

    # Prepare CSV
    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["frame_count", "timestamp_ms", "region", "yolo_conf", "blur_label", "blur_conf", "ocr_text", "ocr_conf"])

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            timestamp = int(cap.get(cv2.CAP_PROP_POS_MSEC))  # in ms

            # 1. YOLO detection
            results = model(frame)
            detections = []
            for box in results[0].boxes:
                x_min, y_min, x_max, y_max = box.xyxy[0].tolist()
                score = float(box.conf[0])
                class_name = "ear_tag" if int(box.cls[0]) == 0 else f"class_{int(box.cls[0])}"
                detections.append([class_name, (x_min, y_min, x_max, y_max), score])
            detections.sort(key=lambda x: x[1][0])

            # 2. Blur classification & OCR
            crops = crop_images(frame, detections)
            blur_labels = []
            blur_confs = []

            for idx, crop in enumerate(crops, start=1):
                cls_results = cls_model(crop, task='classify', imgsz=128)
                probs = cls_results[0].probs
                class_id = int(probs.top1)
                blur_label = cls_model.names[class_id]
                blur_conf = float(probs.top1conf)

                # Default OCR values
                ocr_text = "N/A"
                ocr_conf = "N/A"

                # Conditional OCR
                if blur_label == "clear" and blur_conf >= CLEAR_THRESHOLD:
                    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                    ocr_results = ocr.ocr(crop_rgb, cls=True)
                    if ocr_results and ocr_results[0]:
                        text, conf = ocr_results[0][0][1]
                        ocr_text = text
                        ocr_conf = round(conf, 2)

                # YOLO conf of this region
                yolo_conf = round(detections[idx-1][2], 2)

                # Write row
                writer.writerow([frame_count, timestamp, idx, yolo_conf, blur_label, round(blur_conf, 2), ocr_text, ocr_conf])

    cap.release()
    print(f"Results saved to {OUTPUT_CSV}")
