# RTAD — Real-Time Attention Detection

Real-time engagement detection from live webcam video using OpenCV and MediaPipe FaceMesh.
Facial landmarks and gaze direction are turned into a compact feature vector, and a lightweight
classifier scores the viewer's attention state as **attentive**, **distracted**, or **drowsy**.

## Pipeline

```
webcam ─► VideoStream ─► FaceMeshDetector ─► FeatureExtractor ─► Classifier ─► TemporalSmoother ─► Overlay / log
           (capture)      (landmarks)         (EAR, MAR, head      (rules or     (EMA + hysteresis)
                                               pose, gaze)          sklearn)
```

## Layout

```
configs/            YAML config (thresholds, camera, model paths)
src/rtad/
  capture/          threaded webcam reader
  landmarks/        MediaPipe FaceMesh wrapper + landmark index constants
  features/         eye (EAR/blink/PERCLOS), mouth (MAR/yawn), head pose, gaze, extractor
  classifier/       rule-based baseline, ML classifier, temporal smoothing
  viz/              OpenCV overlay drawing
  utils/            config loading, latency timing
  pipeline.py       end-to-end per-frame pipeline
scripts/
  download_model.py     fetch the FaceLandmarker model
  run_live.py           live demo
  record_session.py     record labelled benchmark clips
  extract_features.py   video + labels -> feature CSV
  train_classifier.py   train the lightweight classifier
  evaluate.py           accuracy + latency report on the benchmark
data/{raw,processed,labels}   self-recorded benchmark (git-ignored)
models/             trained classifier artifacts
results/            evaluation reports
tests/              unit tests
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
python scripts/download_model.py   # fetches models/face_landmarker.task
```

> Uses the MediaPipe **Tasks API** (`FaceLandmarker`). The legacy `mp.solutions.face_mesh`
> was removed in MediaPipe 1.0.

## Usage

```bash
python scripts/run_live.py                      # live demo (rule-based classifier)
python scripts/record_session.py --subject s01  # record benchmark clips with labels
python scripts/extract_features.py              # build data/processed/features.csv
python scripts/train_classifier.py              # train models/classifier.joblib
python scripts/run_live.py --classifier ml      # live demo with trained model
python scripts/evaluate.py                      # accuracy + latency report
```

## Roadmap

- [ ] FaceMesh landmark extraction + overlay
- [ ] EAR / MAR / head pose / gaze features
- [ ] Rule-based baseline classifier
- [ ] Benchmark recording tool + labelling protocol
- [ ] Train lightweight classifier (logistic regression / small GBM)
- [ ] Temporal smoothing
- [ ] Evaluation: per-class accuracy, F1, confusion matrix, per-stage latency (p50/p95)
