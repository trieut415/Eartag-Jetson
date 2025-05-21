# Eartag-Jetson
This repository contains several pipelines for our Labby x Jetson integration, focused on deploying computer vision models optimized for real-time inference on the Jetson Orin Nano. This guide will be focused on the [Seed Studio J401 Carrier Board](https://www.seeedstudio.com/reComputer-J401-Carrier-Board-for-Jetson-Orin-NX-Orin-Nano-p-5636.html?srsltid=AfmBOooAkb-AXaLjTLzwR_bd8hEQg_XqoU2tUUa1oWQuXeJkGLJrSoZP). 

## Assemble Hardware

To get started with the hardware setup, please refer to the detailed [Hardware Setup Guide](setup_hardware.md).


## Software Setup

### Clone the Repository
```bash
git clone https://github.com/trieut415/Eartag-Jetson.git
```

### Installation
Run the following script to install, and start the pipeline:
```bash
cd Eartag-Jetson
chmod +x setup_dependancies.sh
./setup_dependancies.sh
```
This should install all necessary dependancies to run the pipeline.

### File Structure Description
Here is a top level view of the repository. I will describe the purpose of each more in depth under this graph.
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
#### `dashboard/`
Houses user interface components for configuring, visualizing, and tuning the system, currently this only contains `tune.py` 

- `tune.py`:
    - Streams a live feed from a camera, and has sliders on the bottom for adjusting the frame edge thresholds to be considered, then displays a vertical line. This is used to determine an appropriate X boundary for ignoring cow ear tags on the very edges of the camera stream. Currently the camera resolution resolution we are using is 4608x2592. Here, 4608 is the width, so sometimes the camera will pick up extra ear tags from the next stall over, which are not in the current ROI. To solve this, adjust the slider until an appropriate value is found that properly captures the cows of interest, but excludes the cows on the edge. After an appropriate threshold is determined, navigate to `Eartag-Jetson/src/eartag-jetson/pipeline/multi_stream_pipeline.py` and change `EDGE_MARGIN`. This will exclude OCR results from 500 PX on both right and left side of the frame. 
- 















Parameters to adjust:

| Variable             | Meaning                                                                                                                                                                                      |
| -------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `EDGE_MARGIN = 500`  | Ignores detections within 500 pixels from the **left or right edge** of the frame to avoid partial or spurious readings.                                                                     |
| `FRAME_WIDTH = 4608` | Specifies the full **horizontal resolution** of the input video frame in pixels. Used for boundary checks with `EDGE_MARGIN`.                                                                |
| `CLOSE_THRESH = 250` | Defines the maximum pixel distance in the **x-direction** to consider two detections as **"close duplicates"** that should be merged (helps reduce OCR noise or overlapping bounding boxes). |


| Variable             | Meaning                                                                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `TOP_N = N`          | Only the **top N most frequent tags** (after filtering and merging) are retained in the final summary. If there are 4 cows, N = 4.                                          |
| `MIN_DETECTIONS = N` | N amount of ear tag detections must be present before calling `run_milking_session` N .                           |
| `STREAK_THRESH = N` | Minimum number of **consecutive frames** where a tag must persist to be considered part of an active milking session — helps detect entry/start. Currently using N=50 with a 10fps video -> 5s of low detections.|
