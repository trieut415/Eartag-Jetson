#!/usr/bin/env python3
import os
os.environ['GLOG_minloglevel'] = '2'

import cv2, time, logging, serial, multiprocessing
from multiprocessing import get_context
import numpy as np
from collections import defaultdict
from eartag_jetson.common.common_utils import (
    find_project_root, get_logger, send_over_esp
)
from eartag_jetson.pipeline.stall_multi import StallMultiDetector

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BLE_CODES      = ["MM2502V0003FMT", "MM2502V0007FMT"]
PASSWORDS = ["o8vTaJ", "TBoWQU"]
SERIAL_PORT    = "/dev/ttyACM0"
BAUD_RATE      = 115200

#Stall Adjustments
EDGE_MARGIN    = 500
FRAME_WIDTH = 4608         
CLOSE_THRESH   = 250


TOP_N          = 4
MIN_DETECTIONS = 7
STREAK_THRESH  = 50

def process_stream(video_path: str, password: str, ble_code: str):
    logger = get_logger(f"proc-{ble_code}")
    logger.info(f"[{ble_code}] Opening video: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"[{ble_code}] Cannot open video")
        return

    # ─── open serial port once ───────────────────────────────────────────────
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        logger.info(f"[{ble_code}] Serial port opened")
    except serial.SerialException as e:
        logger.error(f"[{ble_code}] Serial error: {e}")
        return

    # ─── instantiate detector ────────────────────────────────────────────────
    detector = StallMultiDetector(
        caps={0: cap},
        api_endpoint="",
        min_detections=MIN_DETECTIONS,
        streak_threshold=STREAK_THRESH,
        logger=logger
    )

    # ─── run detection, then shutdown ──────────────────────────────────────
    try:
        if not detector.wait_for_milking():
            logger.warning(f"[{ble_code}] No milking detected; exiting")
            return

        agg, end_ts = detector.run_milking_session()

    except Exception as e:
        logger.error(f"[{ble_code}] Detection error: {e}", exc_info=True)
        return

    finally:
        detector.shutdown()

    # ─── POST‑PROCESS AGGREGATION ────────────────────────────────────────────────
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
        print("No valid detections after edge filter")
        return

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
    TOP_N = 4
    top_items = sorted(deduped, key=lambda e: e['frequency'], reverse=True)[:TOP_N]
    top_items.sort(key=lambda e: e['median_x'])     # left-to-right
    tags_lr = [e["text"] for e in top_items] 

    # ─── SEND OVER UART ─────────────────────────────────────────────────────
    try:
        send_over_esp(ser, password, ble_code, tags_lr, end_ts, logger)
    except Exception as e:
        logger.error(f"[{ble_code}] Serial write error: {e}")
    finally:
        if ser.is_open:
            ser.close()
            logger.info(f"[{ble_code}] Serial port closed.")

    logger.info(f"[{ble_code}] Done.")



def main():
    base = find_project_root()
    videos = [
        os.path.join(
            base, "src", "eartag_jetson",
            "data_collection", "saved_videos", "video6_single.avi"
        ),
        os.path.join(
            base, "src", "eartag_jetson",
            "data_collection", "saved_videos", "video4_single.avi"
        ),
    ]

    # use spawn context to avoid fork-based segfaults
    ctx = get_context('spawn')
    procs = []
    for vid, pw, code in zip(videos, PASSWORDS, BLE_CODES):
        p = ctx.Process(
            target=process_stream,
            args=(vid, pw, code),
            name=f"proc-{code}"
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    logging.getLogger(__name__).info("All streams finished.")


if __name__ == "__main__":
    main()
