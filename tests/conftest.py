# tests/conftest.py
import os
import pytest
from eartag_jetson.common.common_utils import find_project_root

def pytest_addoption(parser):
    parser.addoption(
        "--input-image",
        default=None,
        help="path to input image for detection_image tests"
    )
    parser.addoption(
        "--input-video",
        default=None,
        help="path to input video for detection_video tests"
    )

@pytest.fixture(scope="session")
def base_dir():
    """Absolute path to the project root (where setup.py lives)."""
    return find_project_root()

@pytest.fixture(scope="session")
def input_path_image(request, base_dir):
    """
    Return either the user‑provided --input-image or the default frame13.jpg.
    """
    ip = request.config.getoption("input_image")
    if ip:
        return ip
    return os.path.join(
        base_dir,
        "src", "eartag_jetson", "data_collection",
        "saved_frames", "frame13.jpg"
    )

@pytest.fixture(scope="session")
def input_path_video(request, base_dir):
    """
    Return either the user‑provided --input-video or the default cow_video_test.mp4.
    """
    ip = request.config.getoption("input_video")
    if ip:
        return ip
    return os.path.join(
        base_dir,
        "src", "eartag_jetson", "resources",
        "cow_video_test.mp4"
    )
