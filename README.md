# Eartag-Jetson

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
