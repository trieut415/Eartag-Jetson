#!/usr/bin/env python3
import os, cv2, csv, json, time, logging, serial
import numpy as np
from collections import defaultdict
from datetime import datetime, timezone
from serial import SerialException, SerialTimeoutException

from eartag_jetson.common.common_utils import find_project_root, get_logger
from eartag_jetson.pipeline.stall_detector import StallDetector

# ─── CONSTANTS ───────────────────────────────────────────────────────────────────
PASSWORD       = "o8vTaJ"
BLE_CODES      = ["MM2502V0003FMT", "MM2502V0007FMT"]   # index matches camera index
SERIAL_PORT    = "/dev/ttyACM0"
BAUD_RATE      = 115200

EDGE_MARGIN    = 350    # px to ignore on each side of the frame
CLOSE_THRESH   = 250    # px for collapsing near‑duplicates
TOP_N          = 4      # max number of stalls to keep
MIN_DETECTIONS = 7      # start session when ≥7 tags visible
STREAK_THRESH  = 50     # unused for video, but ctor requires it

# ─── HELPERS ─────────────────────────────────────────────────────────────────────
def send_over_esp(ser, ble_code: str, agg: dict, end_ts: float):
    """
    Build and transmit the JSON payload over an open serial port.
    Only the top 3 ear‑tags (by count) are included.
    """
    # pick top 3 tags by count
    sorted_items = sorted(agg.items(), key=lambda kv: kv[1]["count"], reverse=True)
    ear_tag_str  = ", ".join(tag for tag, _ in sorted_items[:4])

    # ISO‑8601 timestamp in UTC (milliseconds precision)
    iso_time = (
        datetime.fromtimestamp(end_ts, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    payload = {
        "password": PASSWORD,
        "ble_code": ble_code,
        "time": iso_time,
        "ear_tag": ear_tag_str,
    }

    try:
        ser.write((json.dumps(payload) + "\n").encode("utf-8"))
        ser.flush()
        logger.info(f"Sent to ESP32 ({ble_code}): {payload}")
    except (SerialTimeoutException, SerialException) as e:
        logger.error(f"Serial write error: {e}")

# ─── MAIN ────────────────────────────────────────────────────────────────────────
def main():
    # initialise logger early
    global logger
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

    # ─── POST‑PROCESS AGGREGATION ────────────────────────────────────────────────
    # build summary rows
    summary = []
    for tag, data in agg.items():
        median_x = float(np.median(data["x_list"]))
        summary.append({"text": tag, "frequency": data["count"], "median_x": median_x})

    # drop detections near image edges
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 4608
    summary = [e for e in summary
               if EDGE_MARGIN <= e["median_x"] <= frame_w - EDGE_MARGIN]
    if not summary:
        logger.warning("Nothing left after edge filter – nothing to send.")
        ser.close()
        return

    # merge near‑duplicates
    summary.sort(key=lambda e: e["median_x"])
    deduped, cluster = [], [summary[0]]
    for entry in summary[1:]:
        if abs(entry["median_x"] - cluster[-1]["median_x"]) <= CLOSE_THRESH:
            cluster.append(entry)
        else:
            deduped.append(max(cluster, key=lambda e: (e["frequency"], -e["median_x"])))
            cluster = [entry]
    deduped.append(max(cluster, key=lambda e: (e["frequency"], -e["median_x"])))

    # keep at most TOP_N, highest frequency first
    top_items = sorted(deduped, key=lambda e: e["frequency"], reverse=True)[:TOP_N]

    # build a *clean* agg dict using the merged counts
    clean_agg = {
        e["text"]: {"count": e["frequency"]}
        for e in top_items
    }

    # ─── SEND OVER UART ──────────────────────────────────────────────────────────
    send_over_esp(ser, BLE_CODES[0], clean_agg, end_ts)

    ser.close()
    logger.info("Serial port closed; done.")

# -------------------------------------------------------------------------------
if __name__ == "__main__":
    main()
