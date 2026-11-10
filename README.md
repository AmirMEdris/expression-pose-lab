# Expression + Pose Lab

A modern continuation of [Amir Edris's 2020 facial-expression capstone](https://github.com/AmirMEdris/Facial-Expression-Recognition).

The first milestone combines the recovered expression baseline with MediaPipe body-pose landmarks. It runs locally on images, videos, or an explicitly selected camera. The original capstone repository is preserved.

## Quick start (Windows / Python 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe app.py --image path\to\photo.jpg --output results\photo.png
```

On macOS/Linux, use `.venv/bin/python` instead. Those platforms have not yet been verified for this project.

`requirements-lock.txt` records the full tested Windows environment (including test tools). Use it instead of `requirements.txt` when you want the exact dependency versions used for validation.

```powershell
# Body pose only (does not load the expression model)
.\.venv\Scripts\python.exe app.py --image photo.jpg --pose-only

# Process a video and save an annotated MP4 + per-frame JSONL
.\.venv\Scripts\python.exe app.py --video input.mp4 --output results\video.mp4

# Open camera 0; Q or Escape closes it
.\.venv\Scripts\python.exe app.py --camera 0

# Run fast tests
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest
```

Camera capture is never started by importing modules or installing the project. Images and videos are processed on the local machine. Downloads happen only when you run the model downloader or install packages. Video files are saved without audio; camera recordings use nominal capture FPS and are not guaranteed to preserve wall-clock timing.

## What changed

- Replaced the missing `tf_pose` dependency with MediaPipe Pose Landmarker (lite).
- Preserved the original baseline architecture and grayscale 48×48 /255 preprocessing.
- Added standalone commands, pinned direct dependencies, verified model downloads, and tests.
- Batched expression inference across detected faces.
- Added an `uncertain` display label below a configurable model-score threshold (default 0.6).
- Kept raw class probabilities and pose visibility/presence in machine-readable output.
- Added resource cleanup and actionable input/model errors.

## Model provenance and limits

Expression weights come from `model_filter.h5` in original commit `514316f2e5dce38a0e8ab35b50807f44fc57a2a4`. This is an early baseline, **not the missing FinalModel**. The original README credits the baseline architecture to Ashadullah Shawon. We have not retrained it or established new accuracy results.

Pose weights are the version-1 MediaPipe Pose Landmarker lite bundle, downloaded from Google's model hosting. See [Google's task documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). Models are downloaded separately and retain their respective terms; this repository does not claim ownership of third-party models or imagery.

Expression labels describe model classifications of visible faces, not a person's internal emotional state. Scores are uncalibrated model probabilities. Pose and expression detections are independent: this version does not match faces to bodies, assign stable person IDs, or infer emotions from posture. The Haar face detector remains sensitive to lighting and head angle. Landmarks may be inaccurate for occluded people; only sufficiently visible/present points are drawn.

## Next milestones to build together

1. **Reliable evaluation:** restore FER2013; preserve its documented partitions; use validation for tuning and an untouched test set for final reporting. Report macro F1, per-class recall, and a correctly oriented confusion matrix.
2. **Better face detection:** measure failures on real camera footage, then compare a modern detector against the Haar baseline.
3. **Video behavior:** benchmark CPU latency/FPS, add stable tracking and temporal smoothing, and test camera capture on the target device.
4. **Training:** compare controlled architecture/augmentation changes and retain experiment configurations, seeds, metrics, and checkpoints.
5. **Interface:** decide together whether the next interface should be a desktop app or a browser demo.

See `VALIDATION.md` for the exact checks performed on the first milestone.
