# src/eartag_jetson/cli.py

import argparse

def handle_detect(args):
    if args.source == "image":
        from eartag_jetson.data_collection.capture_image import main as detect_image
        detect_image()
    elif args.source == "video":
        from eartag_jetson.data_collection.capture_video import main as detect_video
        detect_video()
    else:
        from eartag_jetson.data_collection.stream import detect_stream
        detect_stream(camera_index=int(args.input))

def handle_collect(args):
    from eartag_jetson.data_collection.collect import collect_data
    collect_data(
        mode=args.mode,
        duration=args.duration,
        outdir=args.outdir
    )

def handle_pipeline(args):
    from eartag_jetson.pipeline.runner import run_pipeline
    run_pipeline(
        input=args.input,
        output=args.output,
        steps=args.steps
    )

def handle_test(args):
    import pytest
    # Pick the right test file
    test_file = f"tests/test_{args.suite}.py"

    # Build pytest args
    pytest_args = [test_file]
    if args.input_image:
        pytest_args.append(f"--input-image={args.input_image}")
    if args.input_video:
        pytest_args.append(f"--input-video={args.input_video}")

    # Run pytest on just that file, passing through the input flags
    pytest.main(pytest_args)

def build_parser():
    p = argparse.ArgumentParser(prog="eartag", description="Eartag‑Jetson CLI")
    subs = p.add_subparsers(dest="cmd", required=True)

    # --- detect ---
    d = subs.add_parser("detect", help="run detection")
    d.add_argument(
        "--source", "-s",
        choices=["image","video","stream"],
        required=True,
        help="what to run detection on"
    )
    d.add_argument(
        "--input", "-i",
        help="file path (image/video) or camera index (for stream)"
    )
    d.set_defaults(func=handle_detect)

    # --- collect ---
    c = subs.add_parser("collect", help="capture frames or video")
    c.add_argument(
        "--mode", choices=["frames","video"],
        default="frames",
        help="collect as individual frames or one video"
    )
    c.add_argument(
        "--duration", "-d",
        type=int, default=60,
        help="seconds to record"
    )
    c.add_argument(
        "--outdir", "-o",
        default="saved_frames",
        help="where to store the captures"
    )
    c.set_defaults(func=handle_collect)

    # --- pipeline ---
    p_run = subs.add_parser("pipeline", help="run full detect→OCR pipeline")
    p_run.add_argument(
        "--input", "-i",
        required=True,
        help="input file or directory"
    )
    p_run.add_argument(
        "--output", "-o",
        default="results.json",
        help="where to dump JSON results"
    )
    p_run.add_argument(
        "--steps",
        nargs="+",
        default=["detect","ocr"],
        help="which pipeline stages to run"
    )
    p_run.set_defaults(func=handle_pipeline)

    # --- test ---
    t = subs.add_parser("test", help="run a single pytest suite")
    t.add_argument(
        "suite",
        choices=["detection_image", "detection_video", "stream"],
        help="which test to run (maps to tests/test_<suite>.py)"
    )
    t.add_argument(
        "--input-image",
        help="override default image path for detection_image tests"
    )
    t.add_argument(
        "--input-video",
        help="override default video path for detection_video tests"
    )
    t.set_defaults(func=handle_test)

    return p

def main():
    args = build_parser().parse_args()
    args.func(args)
