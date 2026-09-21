"""Run landmarks + feature extraction over recorded clips and write a feature CSV."""

import argparse
from pathlib import Path

import cv2
import pandas as pd

from rtad.features import FeatureExtractor
from rtad.landmarks import FaceMeshDetector
from rtad.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--out", default="data/processed/features.csv")
    args = parser.parse_args()

    cfg = load_config()
    rows = []

    for video in sorted(Path(args.raw).glob("*.mp4")):
        labels = pd.read_csv(Path(args.labels) / f"{video.stem}.csv").set_index("frame_idx")["label"]
        detector = FaceMeshDetector(**cfg["face_mesh"])
        window = int(cfg["features"]["window_seconds"] * cfg["camera"]["fps"])
        extractor = FeatureExtractor(cfg["thresholds"]["ear_closed"], window)

        cap = cv2.VideoCapture(str(video))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            lm = detector.detect(frame)
            if lm is not None and frame_idx in labels.index:
                feats = extractor.extract(lm, frame.shape[:2])
                rows.append({"session": video.stem, "frame_idx": frame_idx, **feats, "label": labels[frame_idx]})
            frame_idx += 1
        cap.release()
        detector.close()
        print(f"{video.name}: {frame_idx} frames")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out, index=False)
    print(f"Wrote {len(rows)} rows -> {args.out}")


if __name__ == "__main__":
    main()
