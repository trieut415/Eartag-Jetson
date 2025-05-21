#!/usr/bin/env python3
import os
os.environ['GLOG_minloglevel'] = '2'  

import cv2, time, logging, serial
import numpy as np
from collections import defaultdict
from eartag_jetson.common.common_utils import find_project_root, get_logger, send_over_esp
from eartag_jetson.pipeline.single_detector import StallDetector
from serial import SerialException

# ─── CONSTANTS ───────────────────────────────────────────────────────────────────
PASSWORD       = "o8vTaJ"
BLE_CODES      = ["MM2502V0003FMT"]   # index matches camera index
SERIAL_PORT    = "/dev/ttyACM0"
BAUD_RATE      = 115200

FRAME_WIDTH    = 4608
EDGE_MARGIN    = 350    # px to ignore on each side of the frame
CLOSE_THRESH   = 250    # px for collapsing near‑duplicates
TOP_N          = 4      # max number of stalls to keep
MIN_DETECTIONS = 7      # start session when ≥7 tags visible
STREAK_THRESH  = 50     # unused for video, but ctor requires it

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():

    logger = get_logger(__name__)

    # ─── SETUP LIVE CAPTURE ────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 4608)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2592)
    if not cap.isOpened():
        raise IOError("Cannot open camera stream")
    logger.info("Camera stream opened")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        logger.info(f"Serial port {SERIAL_PORT} opened")
    except SerialException as e:
        logger.error(f"Failed to open serial port: {e}")
        cap.release()
        return

    try:
        while True:
            # ─── NEW DETECTOR FOR EACH SESSION ─────────────────────────────────
            detector = StallDetector(
                caps={0: cap},
                api_endpoint="",
                min_detections=MIN_DETECTIONS,
                streak_threshold=STREAK_THRESH,
            )

            logger.info("Waiting for milking to start…")
            if not detector.wait_for_milking():
                logger.warning("No milking detected; retrying in 5s.")
                detector.shutdown()
                time.sleep(5)
                continue

            logger.info("Milking detected → running session")
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
        
            summary = [
                e for e in summary
                if EDGE_MARGIN <= e['median_x'] <= FRAME_WIDTH - EDGE_MARGIN
            ]
            if not summary:                # nothing left → exit early
                logger.info("No valid detections after edge filter; waiting for next session.")
                continue

            # ─── MERGE NEAR-DUPLICATES BY POSITION ──────────────────────────────────────────
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
            top_items = sorted(deduped, key=lambda e: e['frequency'], reverse=True)[:TOP_N]
            top_items.sort(key=lambda e: e['median_x'])     # left-to-right

            # ⬅️ 1. get the ordered list of tags
            tags_lr = [e["text"] for e in top_items]        # e.g. ["3013", "2784", "2321", "3010"]

            # ─── SEND OVER UART ──────────────────────────────────────────────
            send_over_esp(ser, BLE_CODES, tags_lr, end_ts, logger)
            logger.info(f"Session done: sent {tags_lr}")

            # optional cooldown if needed
            time.sleep(2)

    except KeyboardInterrupt:
        logger.info("Interrupted by user (Ctrl-C); exiting loop.")
    finally:
        ser.close()
        cap.release()
        logger.info("Clean shutdown complete.")


if __name__ == "__main__":
    main()
