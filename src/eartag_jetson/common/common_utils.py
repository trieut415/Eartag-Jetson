# src/eartag_jetson/common/common_utils.py
import logging
from pathlib import Path
from typing import Sequence, Union
import os
import shutil
from ultralytics import YOLO
import torch

def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Return a logger that logs to stdout with a standard format.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        fmt = "%(asctime)s %(name)s %(levelname)s: %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
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

