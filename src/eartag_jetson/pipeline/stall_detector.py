# pipeline/stall_detector.py
import os
import glob
import cv2
import logging
import re
import time
from collections import defaultdict
from datetime import datetime
from ultralytics import YOLO
from paddleocr import PaddleOCR
from eartag_jetson.common.common_utils import (
    find_project_root,
    export_yolo_to_engine,
    get_logger
)

API_ENDPOINT     = "https://your.api/endpoint"

class StallDetector:
    def __init__(
        self,
        *,
        sources: list[str] | None = None,
        caps: dict[int, cv2.VideoCapture] | None = None,
        api_endpoint: str,
        min_detections: int,
        streak_threshold: int,
    ):
        """
        Either pass:
          • sources=["0","1"] (or file paths) and `caps`=None,  
            and this will call _init_captures(sources),  
          • or pass caps={0: cap0, 1: cap1} and sources=None
            to skip opening cameras internally.
        """
        self.logger = get_logger("multi_cam_detector")
        self.api_endpoint = api_endpoint
        self.min_detections = min_detections
        self.streak_threshold = streak_threshold

        # initialize models & OCR
        self.model, self.ocr = self._init_models()

        # decide whether to use pre‑opened captures or open new ones
        if caps is not None:
            self.caps = caps
            self.logger.info(f"Using {len(caps)} pre‑opened captures")
        else:
            assert sources is not None, "Must provide `sources` if no `caps` given"
            self.caps = self._init_captures(sources)

    def _init_models(self):
        BASE_DIR = find_project_root()
        RES_DIR = os.path.join(BASE_DIR, "src", "eartag_jetson", "resources")
        pt_path = os.path.join(RES_DIR, "seg_model.pt")
        eng_path = os.path.join(RES_DIR, "seg_model.engine")

        logging.getLogger("ppocr").setLevel(logging.ERROR)
        self.logger.info("Loading YOLO model...")
        model_pt = YOLO(pt_path, task="detect")
        export_yolo_to_engine(model_pt, eng_path, self.logger)
        model = YOLO(eng_path, task="detect")

        self.logger.info("Initializing PaddleOCR...")
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            ocr_version="PP-OCRv4",
            use_space_char=True,
            drop=0.9,
        )
        return model, ocr

    def _init_captures(self, sources):
        caps = {}
        for idx, src in enumerate(sources):
            src_obj = int(src) if isinstance(src, str) and src.isdigit() else src
            cap = cv2.VideoCapture(src_obj)
            if not cap.isOpened():
                self.logger.error(f"Cannot open camera {src}")
                raise IOError(f"Cannot open {src}")
            self.logger.info(f"Camera {idx} opened: {src}")
            caps[idx] = cap
        return caps

    @staticmethod
    def auto_detect_cameras() -> list[str]:
        pattern = "/dev/v4l/by-path/*-video-index0"
        links = glob.glob(pattern)
        if not links:
            raise RuntimeError("No cameras found via by-path detection")
        devs = [os.path.realpath(lnk) for lnk in links if os.path.realpath(lnk).startswith("/dev/video")]
        return sorted(set(devs), key=lambda d: int(d.split("video")[-1]))

    def detect_and_aggregate(self, frame, agg):
        results = self.model(frame)
        dets = sorted(
            [
                (*map(int, box.xyxy[0][:2].tolist()), *map(int, box.xyxy[0][2:].tolist()), float(box.conf[0]))
                for box in results[0].boxes
            ],
            key=lambda x: x[0]
        )
        self.logger.info(f"YOLO found {len(dets)} boxes")

        valid = 0
        for idx, (x0, y0, x1, y1, conf) in enumerate(dets):
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 0, 0), 2)
            cv2.putText(frame, f"{conf:.2f}", (x0, y0-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

            crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            res = self.ocr.ocr(crop, cls=True)

            if res and res[0]:
                text, _ = res[0][0][1]
                self.logger.debug(f"OCR on box {idx}: '{text}'")
                if re.fullmatch(r"\d{4}", text):
                    valid += 1
                    entry = agg[text]
                    entry["count"] += 1
                    entry["x_list"].append(x0)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(frame, text, (x0, y0-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    self.logger.info(f"Accepted tag {text} at x={x0}")

        return valid

    def wait_for_milking(self) -> bool:
        self.logger.info("Waiting for milking to start...")
        while True:
            for cam_id, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Camera {cam_id} stream ended")
                    #change to continue for stream
                    return False
                v = self.detect_and_aggregate(frame, defaultdict(lambda: {"count":0,"x_list":[]}))
                # display_frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
                # cv2.imshow(f"Cam{cam_id}", display_frame)

                if v >= self.min_detections:
                    self.logger.info(f"Milking started on camera {cam_id}")
                    return True
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

    def run_milking_session(self):
        """
        Runs one milking session, returns (agg, end_timestamp).
        End is detected via relative drop and sustained timeout.
        """
        self.logger.info("Running milking session…")
        agg = defaultdict(lambda: {"count":0,"x_list":[]})
        peak_seen = 0
        end_threshold_ratio = 0.4
        end_timeout = 4.0
        end_start = None

        while True:
            ret, frame = next(iter(self.caps.values())).read()
            if not ret:
                #finite video, change to continue 
                self.logger.info("End of video stream reached")
                break
            v = self.detect_and_aggregate(frame, agg)
            peak_seen = max(peak_seen, v)

            if peak_seen >= self.min_detections:
                if v < peak_seen * end_threshold_ratio:
                    if end_start is None:
                        end_start = time.time()
                        self.logger.debug(f"Below {end_threshold_ratio*100:.0f}% of peak; starting end timer")
                else:
                    end_start = None

                if end_start and (time.time() - end_start) >= end_timeout:
                    self.logger.info("Milking session ended (relative drop sustained)")
                    break

            source_id = list(self.caps.keys())[0]
            # display_frame = cv2.resize(frame, None, fx=0.3, fy=0.3)
            # cv2.imshow(f"Cam_{source_id}", display_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return None, None

        return agg, end_start

    def shutdown(self):
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()
        self.logger.info("StallDetector shutdown complete.")
