import os
import cv2
import logging
import csv
from collections import defaultdict
from datetime import datetime, timezone
import numpy as np

from eartag_jetson.common.common_utils import find_project_root
from eartag_jetson.pipeline.stall_detector import StallDetector

def main():
    # ─── CONFIG ───────────────────────────────────────────────────────────────────
    BASE_DIR      = find_project_root()
    VIDEO_PATH    = os.path.join(
        BASE_DIR,
        "src", "eartag_jetson", "data_collection", "saved_videos", "video6_single.avi"
    )
    SUMMARY_CSV   = os.path.join(BASE_DIR, "eartag_stall_summary_top10.csv")

    # ─── LOGGER ───────────────────────────────────────────────────────────────────
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)
    logging.getLogger('ppocr').setLevel(logging.ERROR)

    # ─── OPEN VIDEO CAPTURE ───────────────────────────────────────────────────────
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {VIDEO_PATH}")
    logger.info(f"Opened test video '{VIDEO_PATH}'")

    # ─── INSTANTIATE DETECTOR ─────────────────────────────────────────────────────
    #   - min_detections=4 → start when ≥4 tags appear
    #   - streak_threshold=50 is still required by the constructor but not used for video
    detector = StallDetector(
        caps={0: cap},
        api_endpoint="",       # unused in this test
        min_detections=7,
        streak_threshold=50,
    )

    # ─── WAIT FOR THE “MILKING” TO START ───────────────────────────────────────────
    if not detector.wait_for_milking():
        logger.warning("No milking detected → exiting without summary")
        detector.shutdown()
        return

    # ─── RUN THE “MILKING” SESSION ────────────────────────────────────────────────
    agg, end_ts = detector.run_milking_session()
    detector.shutdown()

    # ─── BUILD SUMMARY ──────────────────────────────────────────────────────────────
    summary = []
    for text, data in agg.items():
        median_x = float(np.median(data['x_list']))
        summary.append({
            'text': text,
            'frequency': data['count'],
            'median_x': median_x
        })

    # ─── DROP ENTRIES NEAR THE LEFT / RIGHT EDGES ───────────────────────────────────
    EDGE_MARGIN   = 350            # pixels to ignore at each side
    FRAME_WIDTH   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # returns 0 for some codecs
    if FRAME_WIDTH == 0:           # fallback if the capture can’t report width
        FRAME_WIDTH = 4608         # use the known resolution of your video
    summary = [
        e for e in summary
        if EDGE_MARGIN <= e['median_x'] <= FRAME_WIDTH - EDGE_MARGIN
    ]
    if not summary:                # nothing left → exit early
        print("No valid detections after edge filter")
        return

    # ─── MERGE NEAR-DUPLICATES BY POSITION ──────────────────────────────────────────
    CLOSE_THRESH = 200  # pixels
    summary.sort(key=lambda e: e['median_x'])

    deduped, cluster = [], [summary[0]]
    for entry in summary[1:]:
        if abs(entry['median_x'] - cluster[-1]['median_x']) <= CLOSE_THRESH:
            cluster.append(entry)
        else:
            deduped.append(max(cluster, key=lambda e: (e['frequency'], -e['median_x'])))
            cluster = [entry]
    deduped.append(max(cluster, key=lambda e: (e['frequency'], -e['median_x'])))

    # ─── FILTER TOP N BY FREQUENCY ─────────────────────────────────────────────────
    TOP_N = 4
    top_items = sorted(deduped, key=lambda e: e['frequency'], reverse=True)[:TOP_N]
    top_items.sort(key=lambda e: e['median_x'])

    # ─── WRITE OUT CSV ─────────────────────────────────────────────────────────────
    with open(SUMMARY_CSV, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['stall_id', 'ocr_text', 'frequency', 'median_x'])
        for idx, entry in enumerate(top_items, start=1):
            writer.writerow([
                idx,
                entry['text'],
                entry['frequency'],
                round(entry['median_x'], 2)
            ])

    print(f"Top {TOP_N} stall summary complete. Results saved to {SUMMARY_CSV}")


if __name__ == "__main__":
    main()
