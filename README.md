# Eartag-Jetson
<img src="src/eartag_jetson/resources/cow_flying.gif" alt="cow flying" width="100%">
This repository contains several pipelines for our Labby x Jetson integration, focused on deploying computer vision models optimized for real-time inference on the Jetson Orin Nano. This guide will be focused on the [Seed Studio J401 Carrier Board](https://www.seeedstudio.com/reComputer-J401-Carrier-Board-for-Jetson-Orin-NX-Orin-Nano-p-5636.html?srsltid=AfmBOooAkb-AXaLjTLzwR_bd8hEQg_XqoU2tUUa1oWQuXeJkGLJrSoZP). 

# Get Started

## 1. Assemble Hardware

To get started with the hardware setup, please refer to the detailed [Hardware Setup Guide](setup_hardware.md).


## 2. Software Setup
After setting up the hardware, you may have to download a browser, and navigate to the repository.

### Clone the Repository
```bash
git clone https://github.com/trieut415/Eartag-Jetson.git
```

### Installation
Run the following script to install all necessary dependancies:
```bash
cd Eartag-Jetson
chmod +x setup_dependancies.sh
./setup_dependancies.sh
```

Activate the virtual environment:
```bash
source labby-eartag/bin/activate
```
All scripts below can be ran with `python3 /path/to/python/file`. For example:
```bash
python3 Eartag-Jetson/src/eartag_jetson/pipeline/pipeline.py
```
---

### File Structure Description
Here is a top level view of the repository. I will describe the purpose of each more in depth under this graph. This will provide you all insights on what each file does, and the relevant parameters to adjust and fine tune. Pipeline will contain the primary files that will be used, however descriptions for all can be found below.
```bash
Eartag-Jetson/
├── dashboard/                        # Streamlit or other UI components for visualization, tuning, or configuration
├── manual_tests/                     # Standalone scripts or test cases for manually validating features (e.g., OCR, detection)
└── src/
    └── eartag_jetson/                # Main Python package containing all application logic
        ├── common/                   # Shared utility functions (e.g., logging, file I/O, helpers)
        ├── data_collection/          # Code and output related to collecting data
        │   ├── saved_frames/         # Extracted image frames from videos for annotation or inspection
        │   └── saved_videos/         # Raw or processed video files used for detection, training, or debugging
        ├── pipeline/                 # Core detection and processing logic (e.g., stall detector, session management)
        └── resources/                # Static assets (e.g., models, images, fonts) loaded at runtime
```

---

#### `dashboard/`
Houses user interface components for configuring, visualizing, and tuning the system, currently this only contains `tune.py` 

- `tune.py`:
    - Streams a live feed from a camera, and has sliders on the bottom for adjusting the frame edge thresholds to be considered, then displays a vertical line. This is used to determine an appropriate X boundary for ignoring cow ear tags on the very edges of the camera stream. Currently the camera resolution resolution we are using is 4608x2592. Here, 4608 is the width, so sometimes the camera will pick up extra ear tags from the next stall over, which are not in the current ROI.
    - To solve this, adjust the slider until an appropriate value is found that properly captures the cows of interest, but excludes the cows on the edge. After an appropriate threshold is determined, navigate to `Eartag-Jetson/src/eartag-jetson/pipeline/multi_stream_pipeline.py` and change `EDGE_MARGIN`. This will exclude OCR results from 500 PX on both right and left side of the frame. Adjust this to be more exclusive/strict than inclusive in the case that the edge cows may move their head into the frame.

---

#### `manual_tests/`
Holds standalone Python scripts used for debugging or verifying specific features of the system. Examples include:
- Running OCR or object detection on test images/videos
- Validating serial communication
- Manually invoking pipeline steps
1. `test_detection_image.py`
    - Runs a quick ear tag detection script without OCR on an image and displays the results.
2. `test_detection_video.py`
   - Runs a quick ear tag detection script without OCR on a video and displays the results.
3. `test_detection_stream.py`
    - Runs a quick ear tag detection script without OCR on an image and displays the results.
4. `test_multi_pipeline.py`
    - Runs the full end-to-end pipeline on multiple input videos simultaneously, and transmits it over UART using the ESP32.
5. `test_pipeline_no_cls_video.py`
    - Runs a pipeline test with no classification of blurriness after detection of ear tags. This was a preliminary test to see if classifying whether or not a photo was blurry would contribute to PaddleOCR model performance, since running OCR on blurry images is wasteful. Still could be implemented, but currently would rather have more data samples since this is an additional layer of filtering. 
6. `test_pipeline_video_cls.py`
    - Runs a pipeline test with classification of blurriness after detection of ear tags. This was a preliminary test to see if classifying whether or not a photo was blurry would contribute to PaddleOCR model performance, since running OCR on blurry images is wasteful. Still could be implemented, but currently would rather have more data samples since this is an additional layer of filtering.  
7. `test_single_pipeline.py`
    - Tests the end-to-end pipeline on a single input video.

---


#### `src/`
Contains all the source code for the Eartag-Jetson project. This directory is structured as a Python package and is the root for all application logic.

---

#### `src/eartag_jetson/`
The main Python package for the project. All functional modules—such as data collection, detection pipelines, and utilities—are organized under this namespace. This makes the project importable as a package.

---

#### `src/eartag_jetson/common/`
A utility module that contains shared functions and helpers used throughout the codebase. This may include:
- Logging setup
- Path resolution functions
- ESP communication helpers
- General-purpose utilities for consistent, reusable code
---

#### `src/eartag_jetson/data_collection/`
Contains tools and scripts for collecting image and video data from the camera pipeline. This data is often used for debugging, training models, or validating system performance. Scripts here may include logic to save frames during detection events or archive full video sessions.

1. `capture_image.py`:
    - Takes a image and saves it to saved_frames.
2. `capture_video.py`:
    - Takes a image and saves it to saved_videos.

---

#### `src/eartag_jetson/data_collection/saved_frames/`
A subdirectory that stores individual image frames captured from video streams. These are typically extracted during events of interest (e.g., a cow entering a stall) for later analysis, annotation, or OCR evaluation.

---

#### `src/eartag_jetson/data_collection/saved_videos/`
Stores video recordings captured during data collection runs. These may include full milking sessions, test videos, or clips used for debugging detection and OCR performance.

---

#### `src/eartag_jetson/pipeline/`
Contains the core processing logic for the application. This includes:
- Stall detection logic
- YOLO object detection and PaddleOCR integration
- Aggregation of detection results
- Session control (e.g., starting and ending milking sessions)

1. `pipeline.py`:
    - Contains the end to end pipeline that detects ear tags and at the end of a milking session uploads results to the Labby backend.

**Parameters to adjust:**

| Variable             | Meaning                                                                                                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EDGE_MARGIN = 500`  | Ignores detections within 500 pixels from the **left or right edge** of the frame to avoid partial or spurious readings.                                                                     |
| `FRAME_WIDTH = 4608` | Specifies the full **horizontal resolution** of the input video frame in pixels. Used for boundary checks with `EDGE_MARGIN`.                                                                |
| `CLOSE_THRESH = 250` | Defines the maximum pixel distance in the **x-direction** to consider two detections as **"close duplicates"** that should be merged (helps reduce OCR noise or overlapping bounding boxes). |


| Variable             | Meaning                                                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TOP_N = N`          | Only the **top N most frequent tags** (after filtering and merging) are retained in the final summary. If there are 4 cows, N = 4.                                          |
| `MIN_DETECTIONS = N` | N amount of ear tag detections must be present before calling `run_milking_session`. This is tuneable, I use ~75% so for 4 cows (8 eartags), so N = 6.                           |
| `STREAK_THRESH = N` | Minimum number of **consecutive frames** where a tag must persist to be considered part of an active milking session — helps detect entry/start. Currently using N=50 with a 10fps video -> 5s of low detections.|

2. `stall_detector.py`:
    - Contains the stall detector class for a single stream milking session
        - Relevant methods:
            - `wait_for_milking`
                - Waits until enough tags have been detected to start a milking session
            - `run_milking_session`
                - Continuously runs detection and OCR until the end of a milking session, as defined by `min_detections`. 
                  A session is considered ended when the number of detections drops below a percentage (`end_threshold_ratio`) 
                  of the peak detection count, and this drop is sustained for a minimum timeout duration (`end_timeout` seconds). The idea of this is if insufficient ear tags have been detected for some period of time, then it will end and upload results.

3. `stall_multi.py`:
    - Contains the stall detector class for a multi stream milking session
        - Relevant methods:
            - `wait_for_milking`
                - Waits until enough tags have been detected to start a milking session
            - `run_milking_session`
                - Continuously runs detection and OCR until the end of a milking session, as defined by `min_detections`. 
                  A session is considered ended when the number of detections drops below a percentage (`end_threshold_ratio`) 
                  of the peak detection count, and this drop is sustained for a minimum timeout duration (`end_timeout` seconds). The idea of this is if insufficient ear tags have been detected for some period of time, then it will end and upload results.
---

#### `src/eartag_jetson/resources/`
Holds static assets required at runtime by the application. This includes:
- Trained model files (`.pt`, `.onnx`, `.engine`) for classification, segmentation, and OCR
- Reference images and GIFs for documentation
- Fonts (e.g., `times_new_roman.ttf`) for use in rendering overlays or OCR post-processing

These assets are loaded as needed by the application and are kept separate from code for clarity and maintainability.














