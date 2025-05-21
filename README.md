# Eartag-Jetson
This repository contains several pipelines for our Labby x Jetson integration, focused on deploying computer vision models optimized for real-time inference on the Jetson Orin Nano. This guide will be focused on the [Seed Studio J401 Carrier Board](https://www.seeedstudio.com/reComputer-J401-Carrier-Board-for-Jetson-Orin-NX-Orin-Nano-p-5636.html?srsltid=AfmBOooAkb-AXaLjTLzwR_bd8hEQg_XqoU2tUUa1oWQuXeJkGLJrSoZP). This process will 

## Assemble Hardware
1. Install fan onto the Orin Nano computing module, the metal mounting bracket is to be placed on the bottom of the module, and the fan on the top. A video can be found [here](https://www.youtube.com/watch?v=PKADTqkG538).
2. Unscrew the two computing module mounting screws on the carrier board. They should be on the opposite side of the I/O ports, one is  next to the Control and UART Header, and the other is located next to the CAN Header. See here: ![J401](src/eartag_jetson/resources/j401.png)
3. Slide in the computing module into the 260 PIN SODIMM until it clicks into place, you may have to move the side clips out of the way before you slide the Orin Nano module in.
4. Install storage on the bottom of the board in the M.2 KEY M slot
 
## Flash Jetpack OS onto board (Original guide here)
You can feel free to follow the original guide, but the condensed steps for this specific setup, Jetpack 6.2 w/ the J401 are here.
1. To do this, you need a Ubuntu Host Computer running Ubuntu 22.04, for this I set up my Windows computer to dual-boot into Ubuntu22.04. There are many youtube videos on how to do this, here is [one](https://www.youtube.com/watch?v=mXyN1aJYefc&t=1s).
2. Here is a quick overview of what to do, but I will lay it out step by step after.
![Recovery Mode](src/eartag-jetson/resources/j401_set_recovery.gif).
3. After you have set up dual boot, boot into it. 
4. Connect a female to female jumper pin to force the device into forced recovery mode by connected the GND and FC Rec pins shown ![here](src/eartag-jetson/resources/jumper.png).
5. 

## Setup

### Clone the Repository
```bash
git clone https://github.com/trieut415/labby-hailo.git
```

### Installation
Run the following script to install, and start the pipeline:
```bash
cd labby-hailo
chmod +x labby-hailo-setup.sh
./labby-hailo-setup.sh
```
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
