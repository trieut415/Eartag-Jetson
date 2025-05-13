import cv2
import logging
import csv
import os
import re
from ultralytics import YOLO
from paddleocr import PaddleOCR
from eartag_jetson.common.common_utils import find_project_root, export_yolo_to_engine
from collections import defaultdict

def main():
    # ─── CONFIG ───────────────────────────────────────────────────────────────────
    BASE_DIR      = find_project_root()
    VIDEO_PATH    = os.path.join(BASE_DIR, "src", "eartag_jetson", "data_collection", "saved_videos", "video4_single_stall_new.avi")
    RESOURCES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")

    PT_MODEL_PATH     = os.path.join(RESOURCES_DIR, "seg_model.pt")
    ENGINE_MODEL_PATH = os.path.join(RESOURCES_DIR, "seg_model.engine")
    SUMMARY_CSV       = os.path.join(BASE_DIR, "eartag_stall_summary.csv")

    # ─── LOGGER ───────────────────────────────────────────────────────────────────
    logger = logging.getLogger(__name__)
    logging.getLogger('ppocr').setLevel(logging.ERROR)

    # ─── EXPORT / LOAD MODEL ──────────────────────────────────────────────────────
    model_pt = YOLO(PT_MODEL_PATH)
    export_yolo_to_engine(model_pt, ENGINE_MODEL_PATH, logger)
    model = YOLO(ENGINE_MODEL_PATH)

    # ─── INIT OCR ─────────────────────────────────────────────────────────────────
    ocr = PaddleOCR(
        use_angle_cls=True,
        lang='en',
        ocr_version='PP-OCRv4',
        use_space_char=True,
    )

    # ─── VIDEO CAPTURE ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")

    # ─── AGGREGATION SETUP ────────────────────────────────────────────────────────
    agg = defaultdict(lambda: {'count': 0, 'x_list': []})

    # ─── SESSION‑BREAK SETUP ──────────────────────────────────────────────────────
    low_detect_streak = 0
    STREAK_THRESHOLD = 50  # break if <4 tags for this many consecutive frames

    # ─── PROCESSING LOOP ──────────────────────────────────────────────────────────
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)
        detections = []
        for box in results[0].boxes:
            x0, y0, x1, y1 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            detections.append((x0, y0, x1, y1, conf))
        detections.sort(key=lambda d: d[0])

        # OCR + per‐frame valid count
        valid_this_frame = 0
        for x0, y0, x1, y1, _ in detections:
            crop = frame[y0:y1, x0:x1]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            ocr_res = ocr.ocr(crop_rgb, cls=True)
            if ocr_res and ocr_res[0]:
                text, _conf = ocr_res[0][0][1]
                if re.fullmatch(r"\d{4}", text):
                    valid_this_frame += 1
                    entry = agg[text]
                    entry['count'] += 1
                    entry['x_list'].append(x0)

        # check for session break
        if valid_this_frame < 4:
            low_detect_streak += 1
            if low_detect_streak >= STREAK_THRESHOLD:
                logger.info(f"Ending session: {low_detect_streak} consecutive frames with <4 tags")
                break
        else:
            low_detect_streak = 0

    cap.release()

    # ─── BUILD SUMMARY ────────────────────────────────────────────────────────────
    summary = []
    for text, data in agg.items():
        avg_x = sum(data['x_list']) / len(data['x_list'])
        summary.append({
            'text': text,
            'frequency': data['count'],
            'avg_x': avg_x
        })
    summary.sort(key=lambda e: (-e['frequency'], e['avg_x']))

    # ─── WRITE OUT CSV ────────────────────────────────────────────────────────────
    with open(SUMMARY_CSV, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['stall_id', 'ocr_text', 'frequency', 'avg_x'])
        for idx, entry in enumerate(summary, start=1):
            writer.writerow([
                idx,
                entry['text'],
                entry['frequency'],
                round(entry['avg_x'], 2)
            ])

    print(f"Stall summary complete. Results saved to {SUMMARY_CSV}")

if __name__ == "__main__":
    main()
