# src/eartag_jetson/common/common_utils.py
import os
import logging
from pathlib import Path
from typing import Sequence, Union
import shutil
from ultralytics import YOLO
import torch
from colorama import Fore, Style, init
from datetime import datetime, timezone
from serial import SerialException, SerialTimeoutException
import json
import logging
from logging import Logger

init(autoreset=True)

from colorama import Fore, Style
import logging

class ColorFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            "DEBUG": Fore.BLUE,
            "INFO": Fore.GREEN,
            "WARNING": Fore.YELLOW,
            "ERROR": Fore.RED,
            "CRITICAL": Fore.RED + Style.BRIGHT,
        }.get(record.levelname, "")

        fmt = (
            f"{Fore.WHITE}[%(asctime)s]{Style.RESET_ALL} "
            f"{level_color}[%(levelname)s]{Style.RESET_ALL} "
            f"{Fore.YELLOW}[%(processName)s]{Style.RESET_ALL} "
            f"{Fore.CYAN}%(filename)s{Style.RESET_ALL} "
            f"- %(message)s"
        )
        formatter = logging.Formatter(fmt, datefmt="%Y-%m-%d %H:%M:%S")
        return formatter.format(record)

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # clear any old handlers so you don’t double-log
    if logger.hasHandlers():
        logger.handlers.clear()

    handler = logging.StreamHandler()
    handler.setFormatter(ColorFormatter())
    logger.addHandler(handler)

    return logger


def find_project_root(
    marker_files: Union[str, Sequence[str]] = ("pyproject.toml", "setup.py")
) -> str:
    """
    Walk upwards from this file's directory until you find one of the marker files.
    Returns the absolute path to the project root.
    Raises FileNotFoundError if none is found.
    """
    if isinstance(marker_files, str):
        marker_files = (marker_files,)
    cur = Path(__file__).parent.resolve()
    while True:
        for m in marker_files:
            if (cur / m).exists():
                return str(cur)
        if cur.parent == cur:
            break
        cur = cur.parent
    raise FileNotFoundError(f"No project root found (checked for {marker_files})")

def export_yolo_to_engine(model, engine_path, logger=None):
    """
    Export a YOLO model to a TensorRT engine, but only if a GPU is available.
    On CPU‐only machines, it will warn and skip the engine export.

    Args:
        model (YOLO): an ultralytics YOLO model instance
        engine_path (str): path where the .engine file should live
        logger (logging.Logger, optional): your logger
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if not torch.cuda.is_available():
        logger.warning("No GPU detected; skipping TensorRT engine export.")
        return

    if not os.path.exists(engine_path):
        logger.info("Exporting model to TensorRT engine…")
        export_path = model.export(format="engine", device="0")  # returns a string path
        if export_path and export_path != engine_path:
            shutil.move(export_path, engine_path)
            logger.info(f"Moved exported engine to '{engine_path}'")
        else:
            logger.info(f"Engine already at expected path: {engine_path}")
    else:
        logger.info(f"Engine already exists at '{engine_path}', skipping export.")

def send_over_esp(ser, password: str, ble_code: str, tags_lr: list[str], end_ts: float, logger: Logger):
    """
    Build and transmit the JSON payload over an open serial port.
    Only the top 4 ear‑tags (by count) are included.
    """
    ear_tag_str = ", ".join(tags_lr)            # preserve caller’s order

    # ISO‑8601 timestamp in UTC (milliseconds precision)
    iso_time = (
        datetime.fromtimestamp(end_ts, timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )

    SERIAL_PORT    = "/dev/ttyACM0"
    BAUD_RATE      = 115200
    payload = {
        "password": password,
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