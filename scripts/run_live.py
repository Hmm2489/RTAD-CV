"""Live webcam attention detection demo. Press q to quit."""

import argparse
import time

import cv2

from rtad.capture import VideoStream
from rtad.pipeline import AttentionPipeline
from rtad.utils import load_config
from rtad.viz import draw_overlay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--classifier", choices=["rules", "ml"], default=None)
    parser.add_argument("--camera", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.classifier:
        cfg["classifier"]["type"] = args.classifier
    if args.camera is not None:
        cfg["camera"]["index"] = args.camera

    pipeline = AttentionPipeline(cfg)
    prev = time.perf_counter()
    fps = 0.0

    with VideoStream(**cfg["camera"]) as stream:
        while True:
            frame, _ = stream.read()
            if frame is None:
                continue

            result = pipeline.process(frame)

            now = time.perf_counter()
            fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
            prev = now

            draw_overlay(frame, result.prediction, result.features, result.landmarks, fps, pipeline.latency.last("total"))
            cv2.imshow("RTAD", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    pipeline.close()
    cv2.destroyAllWindows()
    print(pipeline.latency.summary())


if __name__ == "__main__":
    main()
