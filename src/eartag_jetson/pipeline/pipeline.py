import argparse
import time
import json
import serial
from serial import SerialException, SerialTimeoutException
from datetime import datetime, timezone
import logging
import threading
from eartag_jetson.common.common_utils import get_logger
from pipeline.stall_detector import StallDetector

# ——— CONFIG ———
PASSWORD       = "o8vTaJ"
# Map each camera source to a unique BLE code
BLE_CODES      = [
    "MM2502V0003FMT",
    "MM2502V0007FMT"
]
SERIAL_PORT    = "/dev/ttyACM0"
BAUD_RATE      = 115200
MIN_DETECTIONS = 3
STREAK_THRESH  = 50

# Initialize module logger
logger = get_logger(__name__)

def send_over_esp(ser, ble_code, agg, end_ts):
    """
    Build the JSON payload from `agg` (per-camera) and send it over `ser`.
    Only the top 3 ear-tags by count are included.
    """
    try:
        # pick top 3 tags by count
        sorted_items = sorted(
            agg.items(),
            key=lambda kv: kv[1]["count"],
            reverse=True
        )
        top_tags = [tag for tag, data in sorted_items[:3]]
        ear_tag_str = ", ".join(top_tags)

        # current UTC timestamp in ISO8601 w/ milliseconds + Z
        end_dt   = datetime.fromtimestamp(end_ts, timezone.utc)
        iso_time = end_dt.isoformat(timespec="milliseconds").replace("+00:00", "Z")

        payload = {
            "password": PASSWORD,
            "ble_code": ble_code,
            "time": iso_time,
            "ear_tag": ear_tag_str
        }

        msg = json.dumps(payload) + "\n"
        ser.write(msg.encode("utf-8"))
        ser.flush()
        logger.info(f"Sent to ESP32 (BLE {ble_code}): {payload}")

    except (ValueError, TypeError) as e:
        logger.error(f"Payload serialization error: {e}")
    except SerialTimeoutException as e:
        logger.error(f"Serial timeout when writing: {e}")
    except SerialException as e:
        logger.error(f"Serial write error: {e}")

def run_detector(source, ble_code, ser):
    """
    Runs one StallDetector on `source`. Aggregates per-session data,
    then immediately sends top-3 tags via BLE code.
    """
    detector = StallDetector(
        sources=[source],
        api_endpoint="",  # unused
        min_detections=MIN_DETECTIONS,
        streak_threshold=STREAK_THRESH,
    )
    while True:
    if detector.wait_for_milking():
        agg, end_ts = detector.run_milking_session()
        if agg and end_ts:
            send_over_esp(ser, ble_code, agg, end_ts)
    else:
        break
    detector.run()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--auto", action="store_true",
                        help="Auto‑detect cameras (must equal BLE_CODES count)")
    parser.add_argument("--input", nargs="+", default=None,
                        help="Explicit list of video sources")
    args = parser.parse_args()

    # Determine video sources
    if args.auto:
        try:
            sources = StallDetector.auto_detect_cameras()
        except Exception as e:
            logger.error(f"Camera auto-detect failed: {e}")
            return
    elif args.input:
        sources = args.input
    else:
        sources = ["0"]

    if len(sources) != len(BLE_CODES):
        logger.error(
            f"Expected {len(BLE_CODES)} sources for {len(BLE_CODES)} BLE codes, got {len(sources)}"
        )
        return

    # Open serial port
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        time.sleep(2)
        logger.info(f"Opened serial port {SERIAL_PORT} at {BAUD_RATE}bps")
    except SerialException as e:
        logger.error(f"Failed to open serial port: {e}")
        return

    # Launch one detector thread per camera
    threads = []
    for src, code in zip(sources, BLE_CODES):
        t = threading.Thread(
            target=run_detector,
            args=(src, code, ser),
            daemon=False
        )
        t.start()
        threads.append(t)
        logger.info(f"Started detector for source '{src}' with BLE '{code}'")

    # Wait for detectors (they loop until interrupted)
    try:
        for t in threads:
            t.join()
    except KeyboardInterrupt:
        logger.info("Interrupted by user; shutting down.")
    finally:
        if ser and ser.is_open:
            try:
                ser.close()
                logger.info("Serial port closed.")
            except SerialException as e:
                logger.error(f"Error closing serial port: {e}")

if __name__ == "__main__":
    main()
