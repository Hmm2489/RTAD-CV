"""Evaluate detection accuracy and inference latency on the benchmark dataset.

Replays each recorded clip through the full AttentionPipeline (including smoothing)
and compares against frame labels.
"""

import argparse
import json
from pathlib import Path

import cv2
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from rtad.pipeline import AttentionPipeline
from rtad.utils import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", default="data/raw")
    parser.add_argument("--labels", default="data/labels")
    parser.add_argument("--classifier", choices=["rules", "ml"], default="ml")
    parser.add_argument("--out", default="results/eval.json")
    args = parser.parse_args()

    cfg = load_config()
    cfg["classifier"]["type"] = args.classifier

    y_true, y_pred = [], []
    latency_by_session = {}

    for video in sorted(Path(args.raw).glob("*.mp4")):
        labels = pd.read_csv(Path(args.labels) / f"{video.stem}.csv").set_index("frame_idx")["label"]
        pipeline = AttentionPipeline(cfg)
        cap = cv2.VideoCapture(str(video))
        frame_idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            result = pipeline.process(frame)
            if frame_idx in labels.index:
                y_true.append(labels[frame_idx])
                y_pred.append(result.prediction.state.value)
            frame_idx += 1
        cap.release()
        latency_by_session[video.stem] = pipeline.latency.summary()
        pipeline.close()

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    labels_order = ["attentive", "distracted", "drowsy", "no_face"]
    cm = confusion_matrix(y_true, y_pred, labels=labels_order).tolist()

    print(classification_report(y_true, y_pred, zero_division=0))
    print(json.dumps(latency_by_session, indent=2))

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"report": report, "confusion_matrix": {"labels": labels_order, "matrix": cm},
                   "latency_ms": latency_by_session}, f, indent=2)
    print(f"Saved -> {args.out}")


if __name__ == "__main__":
    main()
