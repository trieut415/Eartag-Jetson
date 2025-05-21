# pipeline/stall_mult.py

import os, glob, cv2, logging, re, time, multiprocessing
from collections import defaultdict
from ultralytics import YOLO
import threading
from paddleocr import PaddleOCR
from concurrent.futures import ThreadPoolExecutor
from eartag_jetson.common.common_utils import (
    find_project_root, export_yolo_to_engine, get_logger
)

class StallMultiDetector:
    def __init__(
        self,
        *,
        sources: list[str] | None = None,
        caps: dict[int, cv2.VideoCapture] | None = None,
        api_endpoint: str,
        min_detections: int,
        streak_threshold: int,
        logger: logging.Logger | None = None,
    ):
        self.logger         = logger or get_logger("multi_cam_detector")
        self.api_endpoint   = api_endpoint
        self.min_detections = min_detections
        self.streak_threshold = streak_threshold

        # ─── load models ─────────────────────────────────────────────────────
        self.model, self.ocr = self._init_models()
        # ─── OCR lock ─────────────────────────────────────────────────────────
        # ensure only one thread ever calls into PaddleOCR at a time
        self.ocr_lock = threading.Lock()
        # ─── dynamic OCR thread‐pool ─────────────────────────────────────────
        cpu_total = multiprocessing.cpu_count()
        reserved  = 1
        workers   = max(1, cpu_total - reserved)
        self.logger.info(f"Starting OCR pool: {workers}/{cpu_total} cores")
        self.ocr_executor = ThreadPoolExecutor(max_workers=workers)

        # ─── set up captures ─────────────────────────────────────────────────
        if caps is not None:
            self.caps = caps
            self.logger.info(f"Using {len(caps)} pre-opened captures")
        else:
            assert sources, "Must give sources or caps"
            self.caps = self._init_captures(sources)

    def _init_models(self):
        base = find_project_root()
        res  = os.path.join(base, "src", "eartag_jetson", "resources")
        pt   = os.path.join(res, "seg_model.pt")
        eng  = os.path.join(res, "seg_model.engine")

        logging.getLogger("ppocr").setLevel(logging.ERROR)
        self.logger.info("Loading YOLO…")
        model_pt = YOLO(pt, task="detect")
        export_yolo_to_engine(model_pt, eng, self.logger)
        model = YOLO(eng, task="detect")

        self.logger.info("Init PaddleOCR…")
        ocr = PaddleOCR(
            use_angle_cls=True,
            lang="en",
            ocr_version="PP-OCRv4",
            use_space_char=True,
            drop=0.9,
        )
        return model, ocr
        
    def _safe_ocr(self, crop):
        """Wrap self.ocr.ocr with a lock to prevent concurrent C++ calls."""
        with self.ocr_lock:
            return self.ocr.ocr(crop)

    def _init_captures(self, sources):
        caps = {}
        for i, src in enumerate(sources):
            src_obj = int(src) if isinstance(src, str) and src.isdigit() else src
            cap = cv2.VideoCapture(src_obj)
            if not cap.isOpened():
                raise IOError(f"Cannot open {src}")
            caps[i] = cap
            self.logger.info(f"Camera {i} → {src}")
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
        res = self.model(frame)
        dets = sorted(
            [
                (
                    *map(int, box.xyxy[0][:2].tolist()),
                    *map(int, box.xyxy[0][2:].tolist()),
                    float(box.conf[0])
                )
                for box in res[0].boxes
            ],
            key=lambda x: x[0]
        )
        self.logger.info(f"YOLO → {len(dets)} boxes")

        tasks = []
        for (x0, y0, x1, y1, conf) in dets:
            crop = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)
            fut  = self.ocr_executor.submit(self._safe_ocr, crop)
            tasks.append((x0, y0, x1, y1, fut))

        valid = 0
        for idx, (x0, y0, x1, y1, fut) in enumerate(tasks):
            out = fut.result()
            if not out or not out[0]:
                continue

            text, confidence = out[0][0][1]

            self.logger.debug(f"OCR on box {idx}: '{text}' ({confidence:.2f})")
            if re.fullmatch(r"\d{4}", text):
                valid += 1
                entry = agg[text]
                entry["count"]  += 1
                entry["x_list"].append(x0)
                self.logger.info(f"Accepted tag {text} at x={x0}")

        return len(dets), valid


    def wait_for_milking(self) -> bool:
        self.logger.info("Waiting for milking…")
        while True:
            for cid, cap in self.caps.items():
                ret, frame = cap.read()
                if not ret:
                    self.logger.warning(f"Stream {cid} ended")
                    return False
                boxes, valid = self.detect_and_aggregate(
                    frame,
                    defaultdict(lambda: {"count":0,"x_list":[]})
                )
                if boxes >= self.min_detections:
                    self.logger.info(f"Milking on cam {cid}")
                    return True
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return False

    def run_milking_session(self):
        self.logger.info("Running session…")
        agg = defaultdict(lambda: {"count":0,"x_list":[]})
        peak, end_start = 0, None
        while True:
            # here we just pick one cap (you can extend to multi-cap)
            ret, frame = next(iter(self.caps.values())).read()
            if not ret:
                self.logger.info("End of stream")
                break
            boxes, valid = self.detect_and_aggregate(frame, agg)
            peak = max(peak, boxes)

            if peak >= self.min_detections:
                if boxes < peak * 0.4 and end_start is None:
                    end_start = time.time()
                elif boxes >= peak * 0.4:
                    end_start = None
                if end_start and (time.time() - end_start) >= 4.0:
                    self.logger.info("Session ended")
                    break

            if cv2.waitKey(1) & 0xFF == ord('q'):
                return {}, None

        return agg, end_start

    def shutdown(self):
        self.ocr_executor.shutdown(wait=True)
        for cap in self.caps.values():
            cap.release()
        cv2.destroyAllWindows()
        self.logger.info("Shutdown complete.")
