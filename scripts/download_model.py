"""Download the MediaPipe FaceLandmarker model into models/."""

import urllib.request
from pathlib import Path

URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
OUT = Path("models/face_landmarker.task")


def main():
    if OUT.exists():
        print(f"Already present: {OUT}")
        return
    OUT.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {URL}")
    urllib.request.urlretrieve(URL, OUT)
    print(f"Saved -> {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
