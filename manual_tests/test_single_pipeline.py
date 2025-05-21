#!/usr/bin/env python3
import os
os.environ['GLOG_minloglevel'] = '2'  

import cv2, csv, json, time, logging, serial
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from eartag_jetson.common.common_utils import find_project_root, get_logger, send_over_esp
from eartag_jetson.pipeline.stall_detector import StallDetector


# ─── CONSTANTS ───────────────────────────────────────────────────────────────────
PASSWORD       = "o8vTaJ"
BLE_CODES      = ["MM2502V0003FMT"]   # index matches camera index
SERIAL_PORT    = "/dev/ttyACM0"
BAUD_RATE      = 115200

EDGE_MARGIN    = 350    # px to ignore on each side of the frame
CLOSE_THRESH   = 250    # px for collapsing near‑duplicates
TOP_N          = 4      # max number of stalls to keep
MIN_DETECTIONS = 7      # start session when ≥7 tags visible
STREAK_THRESH  = 50     # unused for video, but ctor requires it

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():

    logger = get_logger(__name__)

    # input video
    base_dir   = find_project_root()
    video_path = os.path.join(
        base_dir, "src", "eartag_jetson", "data_collection", "saved_videos",
        "video6_single.avi"
    )

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")
    logger.info(f"Opened test video '{video_path}'")

    # open serial port first, so we can bail out early if it fails
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # allow USB‑CDC to settle
        logger.info(f"Serial port {SERIAL_PORT} opened @ {BAUD_RATE} bps")
    except SerialException as e:
        logger.error(f"Failed to open serial port: {e}")
        return

    detector = StallDetector(
        caps={0: cap},                 # single source → camera index 0
        api_endpoint="",               # unused in this test
        min_detections=MIN_DETECTIONS,
        streak_threshold=STREAK_THRESH,
    )

    # wait until enough tags appear
    if not detector.wait_for_milking():
        logger.warning("No milking detected → exiting.")
        detector.shutdown()
        ser.close()
        return

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
    CLOSE_THRESH = 250  # pixels
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
    top_items.sort(key=lambda e: e['median_x'])     # left-to-right

    # ⬅️ 1. get the ordered list of tags
    tags_lr = [e["text"] for e in top_items]        # e.g. ["3013", "2784", "2321", "3010"]

    # ─── SEND OVER UART ──────────────────────────────────────────────────────────
    send_over_esp(ser, BLE_CODES, tags_lr, end_ts, logger)

    ser.close()
    logger.info("Serial port closed; done.")

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
