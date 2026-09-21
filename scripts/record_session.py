"""Record benchmark clips for the self-recorded dataset.

While recording, press keys to mark the ground-truth state:
    a = attentive, d = distracted, s = drowsy (sleepy), q = stop
Saves data/raw/<subject>_<timestamp>.mp4 and data/labels/<same>.csv (frame_idx, label).
"""

import argparse
import csv
import time
from pathlib import Path

import cv2

KEY_TO_LABEL = {ord("a"): "attentive", ord("d"): "distracted", ord("s"): "drowsy"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--subject", required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    name = f"{args.subject}_{time.strftime('%Y%m%d_%H%M%S')}"
    video_path = Path("data/raw") / f"{name}.mp4"
    label_path = Path("data/labels") / f"{name}.csv"

    cap = cv2.VideoCapture(args.camera)
    w, h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))

    label = "attentive"
    frame_idx = 0
    with open(label_path, "w", newline="") as f:
        out = csv.writer(f)
        out.writerow(["frame_idx", "label"])
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            writer.write(frame)
            out.writerow([frame_idx, label])
            frame_idx += 1

            preview = frame.copy()
            cv2.putText(preview, f"REC  label={label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imshow("record", preview)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            label = KEY_TO_LABEL.get(key, label)

    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"Saved {frame_idx} frames -> {video_path}, labels -> {label_path}")


if __name__ == "__main__":
    main()
