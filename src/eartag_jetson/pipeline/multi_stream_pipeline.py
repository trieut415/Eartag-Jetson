#!/usr/bin/env python3
import os
os.environ['GLOG_minloglevel'] = '2'

import cv2
import time
import logging
import serial
from serial import SerialException
import numpy as np
from multiprocessing import get_context
from eartag_jetson.common.common_utils import get_logger, send_over_esp
from eartag_jetson.pipeline.multi_detector import StallMultiDetector

# ─── CONSTANTS ────────────────────────────────────────────────────────────────
BLE_CODES      = ["MM2502V0003FMT", "MM2502V0007FMT"]
PASSWORDS      = ["o8vTaJ",      "TBoWQU"]
SERIAL_PORT    = "/dev/ttyACM0"
BAUD_RATE      = 115200

FRAME_WIDTH    = 4608         # fallback if cap.get() returns 0
EDGE_MARGIN    = 500          # px to ignore at each side
CLOSE_THRESH   = 250          # px for collapsing near-duplicates
TOP_N          = 4            # max number of stalls to keep
MIN_DETECTIONS = 7            # start session when ≥7 tags visible
STREAK_THRESH  = 50           # ctor requires it (unused for video)

def process_stream(cam_dev: str, password: str, ble_code: str):
    logger = get_logger(f"proc-{ble_code}")
    logger.info(f"[{ble_code}] Opening camera: {cam_dev}")

    cap = cv2.VideoCapture(cam_dev)
    if not cap.isOpened():
        logger.error(f"[{ble_code}] Cannot open camera {cam_dev}")
        return
    logger.info(f"[{ble_code}] Camera stream opened")

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        logger.info(f"[{ble_code}] Serial port opened")
    except SerialException as e:
        logger.error(f"[{ble_code}] Serial error: {e}")
        cap.release()
        return

    try:
        while True:
            detector = StallMultiDetector(
                caps={0: cap},
                api_endpoint="",
                min_detections=MIN_DETECTIONS,
                streak_threshold=STREAK_THRESH,
                logger=logger
            )

            logger.info(f"[{ble_code}] Waiting for milking to start…")
            if not detector.wait_for_milking():
                logger.warning(f"[{ble_code}] No milking detected; retry in 5s")
                detector.shutdown()
                time.sleep(5)
                continue

            logger.info(f"[{ble_code}] Milking detected → running session")
            try:
                agg, end_ts = detector.run_milking_session()
            except Exception as e:
                logger.error(f"[{ble_code}] Session error: {e}", exc_info=True)
                detector.shutdown()
                time.sleep(5)
                continue
            detector.shutdown()

            # build & filter summary
            summary = [
                {'text': t,
                 'frequency': d['count'],
                 'median_x': float(np.median(d['x_list']))}
                for t, d in agg.items()
            ]

            fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or FRAME_WIDTH
            summary = [
                e for e in summary
                if EDGE_MARGIN <= e['median_x'] <= fw - EDGE_MARGIN
            ]
            if not summary:
                logger.info(f"[{ble_code}] No valid detections; next session")
                continue

            # merge near duplicates
            summary.sort(key=lambda e: e['median_x'])
            deduped, cluster = [], [summary[0]]
            for e in summary[1:]:
                if abs(e['median_x'] - cluster[-1]['median_x']) <= CLOSE_THRESH:
                    cluster.append(e)
                else:
                    deduped.append(max(cluster, key=lambda x: (x['frequency'], -x['median_x'])))
                    cluster = [e]
            deduped.append(max(cluster, key=lambda x: (x['frequency'], -x['median_x'])))

            # top-N
            top_items = sorted(deduped, key=lambda e: e['frequency'], reverse=True)[:TOP_N]
            top_items.sort(key=lambda e: e['median_x'])
            tags_lr = [e['text'] for e in top_items]

            # send over UART
            try:
                send_over_esp(ser, password, ble_code, tags_lr, end_ts, logger)
                logger.info(f"[{ble_code}] Sent tags: {tags_lr}")
            except Exception as e:
                logger.error(f"[{ble_code}] Serial write error: {e}")

            time.sleep(2)

    except KeyboardInterrupt:
        logger.info(f"[{ble_code}] Ctrl-C received; exiting loop")
    finally:
        if ser.is_open:
            ser.close()
            logger.info(f"[{ble_code}] Serial port closed")
        cap.release()
        logger.info(f"[{ble_code}] Camera released; done")


def main():
    # auto-detect all /dev/video* devices
    cams = StallMultiDetector.auto_detect_cameras()
    # match lengths to avoid zip dropping
    n = min(len(cams), len(PASSWORDS), len(BLE_CODES))
    cams, pws, codes = cams[:n], PASSWORDS[:n], BLE_CODES[:n]

    ctx = get_context('spawn')
    procs = []
    for cam_dev, pw, code in zip(cams, pws, codes):
        p = ctx.Process(
            target=process_stream,
            args=(cam_dev, pw, code),
            name=f"proc-{code}"
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    logging.getLogger(__name__).info("All streams finished.")


if __name__ == "__main__":
    main()
