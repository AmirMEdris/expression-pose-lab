# Expression Lab

**Live expressions, body pose, and a window into the neural network.**

![Actual feature maps and gradient-ascent output on a synthetic pattern](docs/neural-gallery.png)

## Use the desktop app

Open **Expression Lab.exe** from the portable Windows folder. Keep `_internal` beside the executable. Python and a terminal are not needed for the packaged app. For a source checkout, run `python desktop.py` in the configured environment.

1. Choose **Camera**, **Screen**, or **Open image**.
2. Turn on **Expression predictions** or **Body pose** when you want those overlays.
3. Choose **Live view**, **Dream**, **Feature maps**, or **Explain prediction** under Explore the model.
4. **Save face crops** saves detected faces. **Save neural artwork** exports the visualization. Nothing is recorded automatically.

The default view gives the input the available space. The neural panel appears only when you choose a visualization. Training is a separate developer tool, not an app button. **Import model** accepts a compatible `.keras` checkpoint and its matching `.json` metadata. **Model results** opens the evaluation report attached to the loaded model, verified against the checkpoint hash; it never asks you to browse for an arbitrary report.

### Understand what you see

| View or control | Meaning |
| --- | --- |
| Live view | Your input with optional expression and pose overlays. |
| Feature maps | Responses from 16 filters. Brighter means stronger response **within that tile**. Each tile has its own scale; these are not class probabilities or an explanation of a prediction. |
| Dream | Multiscale gradient ascent through the loaded expression model’s convolutional features. The same weights operate on a larger image; classification still uses 48×48 faces. |
| Dream area | Whole frame (default) or Face only. Face only uses the largest detected face, falling back to the whole frame when none is found. |
| Expand artwork | Gives the artwork the full available viewport instead of showing it beside the live input. Screen input plus Whole frame processes your screen scene. |
| Dream intensity | How far pixels change during each update. More is not more accurate. |
| Pattern depth, in View settings | Fine detail / Shapes / Abstract select early / middle / late convolution layers. These are navigation labels, not claims about a layer's exact semantic meaning. |
| Dream processing, in View settings | Quick / Balanced / Detailed perform 1 / 3 / 6 pixel-update steps per refresh. More steps take more time; they **do not retrain the model**. |
| Preserve input colors | Your model changes luminance; original chroma is preserved. The classifier still sees grayscale. |
| Explain prediction | Grad-CAM for the largest detected face using the loaded classifier. Choose a class; warm areas show relative positive support, not certainty. |
| Dream resolution, in View settings | Quick / Balanced / Detailed cap the longest edge at 256 / 384 / 512 pixels without stretching. CPU is capped at 256. |

### What is trained—and what is not

The app still defaults to the **recovered historical baseline**, not the missing final capstone model. The new residual candidate completed 50 epochs and scored **62.15% test accuracy and 58.36% macro F1**, improving over the compact candidate below on the same deduplicated FER2013 partition. It has 1,773,735 parameters versus the recovered baseline’s 2,787,015 (about 36% fewer). Its checkpoint was selected by validation loss. This is a comparison with the compact experiment, not proof of superiority to the lost FinalModel or a live-camera benchmark. A new compact candidate was trained from scratch, but its run stopped after 27 of 30 requested epochs. Evaluation of its saved best validation-loss checkpoint produced **59.1% test accuracy and 53.7% macro F1** on 3,572 deduplicated FER2013 test images. This is a development candidate, not a demonstrated improvement or a production-quality claim.

The historical baseline scored 92.8% on those images, but its original training-split provenance is unknown. That number may reflect prior exposure and **must not be presented as clean held-out accuracy**. A second spatial candidate collapsed during training and is not shipped as a recommended model. Live-camera accuracy remains unmeasured. See [dataset protocol](DATASET.md) and the checkpoint-specific reports for the full context.

### GPU acceleration

The Windows GPU path uses Keras with PyTorch CUDA. Expression inference and DeepDream gradients were verified on an RTX 5070 Laptop GPU (8 GB). Full-float32 predictions agreed with the TensorFlow CPU path within 0.000002 on a deterministic eight-input check. MediaPipe pose remains on CPU. GPU availability does not guarantee that every small visualization is faster; capture, face detection, rendering, and transfer overhead still matter.

For a fresh NVIDIA source environment, install `requirements-gpu.txt` instead of `requirements.txt`. The default backend chooses PyTorch when installed and otherwise TensorFlow. A CUDA-enabled PyTorch installation falls back to CPU when no compatible GPU is available. Set `KERAS_BACKEND=tensorflow` before launch to explicitly use the TensorFlow installation. The portable GPU bundle is large because it includes the CUDA runtime.

Dream and prediction explanations both use the loaded expression classifier. No unrelated ImageNet model drives the artwork. Dream clones the convolutional feature graph with exactly the same weights and flexible image dimensions, then uses coarse-to-fine ascent and jitter. Its source-color option restores chroma after luminance optimization. Applying 48×48-trained filters at larger sizes is an artistic extrapolation, not a higher-resolution accuracy claim. Truly learned color features would require labeled RGB training data.

Grad-CAM uses class gradients from the loaded model, without changing its weights or softmax. It is a coarse, relative localization of positive support for a class, not a causal proof or a measure of emotion. No face means no explanation heatmap. See the [Keras Grad-CAM reference](https://keras.io/examples/vision/grad_cam/).

Capture is capped at 30 FPS; artwork updates independently with bounded latest-frame queues. Actual speed depends on settings and hardware. Static-image face detections are reused until inference settings change; pose loads only when enabled. CPU Dream resolution is capped at 256 pixels on the longest edge.

### Training and automatic analytics

Training remains separate from the everyday app:

```powershell
python train.py path\to\fer2013.zip --architecture residual --epochs 50 --output experiments\new-run
```

Each completed run automatically generates learning curves, a confusion matrix (rows=true, columns=predicted), per-class precision/recall/F1, calibration diagnostics, JSON metrics, and saved predictions in an offline HTML report. No manual plotting step is required. Import the resulting `best.keras` with `best.json` to use it in the app; its report is discovered automatically. Architectures include residual, compact, spatial, the recovered baseline, and a reconstruction of the original final architecture. They all train from scratch. The large original model needs a smaller batch size, for example `--batch-size 16`.

The residual candidate uses four skip-connected convolution blocks (32/64/128/192 channels), a spatial dense head, dropout, AdamW, gradient clipping, and square-root class weights capped at 3. Its default learning rate is 0.0003. These are training hypotheses, not an accuracy guarantee. Checkpoints are selected by validation loss, and both the best and latest checkpoints are preserved. Class weighting can be changed with `--weighting none|sqrt|balanced`; record these choices when comparing runs.

---

A modern continuation of [Amir Edris's 2020 facial-expression capstone](https://github.com/AmirMEdris/Facial-Expression-Recognition).

This continuation combines the recovered expression baseline with MediaPipe body-pose landmarks. It runs locally on images, videos, camera frames, or screen capture. The original capstone repository is preserved.

## Quick start (Windows / Python 3.13)

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe download_models.py
.\.venv\Scripts\python.exe app.py --image path\to\photo.jpg --output results\photo.png
```

On macOS/Linux, use `.venv/bin/python` instead. Those platforms have not yet been verified for this project.

`requirements-lock.txt` records the earlier TensorFlow CPU environment. `requirements-gpu-lock.txt` records the desktop GPU development environment. CUDA wheels use the PyTorch index specified in `requirements-gpu.txt`.

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

Camera/screen capture is never started by importing modules or installing the project. Images and videos are processed on the local machine. Downloads happen only when you run the model downloader or install packages. Video files are saved without audio. Explicit live recordings (`--output`) use a fixed 960x720 canvas at nominal 30 FPS and are not guaranteed to preserve wall-clock timing.

### Camera startup on Windows

After setup, double-click `launch_camera.cmd` to start camera 0 with the restored original controls. Nothing is saved automatically; Space explicitly saves detected face crops. Model loading can take 20–40 seconds. Click the main live window before pressing keys; Q or Escape closes it.

Camera capture tries DirectShow, Media Foundation, and automatic selection, requiring an actual frame before accepting a backend. If all fail, close other camera apps and try a normal desktop launch. A restricted execution environment may be unable to access a camera that works in Windows Camera. For another device, run `python app.py --camera 1` with the project environment activated.

### Original live controls

| Key | Action |
| --- | --- |
| A | Switch camera / screen; release the camera while viewing the screen |
| P | Toggle expression predictions |
| E | Toggle pose estimation |
| I | Show/hide instructions |
| F | Show/hide FPS and face count |
| Space | Save detected face crops in `results/captures` and open crop previews |
| Esc / Q | Exit and release capture resources |

Like the original notebook, normal live startup shows face boxes, instructions, and FPS with predictions and pose OFF. The improved pose coverage check still applies when E is enabled. Pose tracking resets when sources change. Failed switches retain the working source and show an error in the panel. Face crops now persist as PNG files instead of only being appended to the notebook's in-memory list. Use `--capture-dir` to choose a destination.

Start directly with the primary screen using `python app.py --screen`. Screen mode captures the full primary display rather than the old notebook's fixed `[900:, :1000]` slice. For a crop, use `--screen-region LEFT TOP WIDTH HEIGHT`, for example `python app.py --screen --screen-region 0 900 1000 600`. Keep the preview outside that region to avoid recursive screen capture. Large frames are reduced to 1280 pixels wide for processing; JSON face boxes refer to that processed frame. Live recording output is opt-in with `--output results/live.mp4`.

## What changed

- Replaced the missing `tf_pose` dependency with MediaPipe Pose Landmarker (lite).
- Preserved the original baseline architecture and grayscale 48×48 /255 preprocessing.
- Added standalone commands, pinned direct dependencies, verified model downloads, and tests.
- Batched expression inference across detected faces.
- Added an `uncertain` display label below a configurable model-score threshold (default 0.6).
- Kept raw class probabilities and pose visibility/presence in machine-readable output.
- Added resource cleanup and actionable input/model errors.

## Pose display and model comparison

The default display now withholds a skeleton unless all 12 major joints (shoulders, elbows, wrists, hips, knees, ankles) are inside the image and meet the visibility/presence threshold. This deliberately favors full-body input. Cropped or occluded people may be omitted even when some joints are accurate.

This is a **coverage check, not proof of correct alignment**. Wrong but confident skeletons can still pass. Raw landmarks remain in JSON under `poses`; `pose_quality` explains which estimates were withheld. The image footer reports withheld estimates. Use `--show-partial-poses` only to inspect raw, potentially wrong estimates.

```powershell
# Download the optional stronger models
.\.venv\Scripts\python.exe download_models.py --pose-model all
.\.venv\Scripts\python.exe app.py --image photo.jpg --pose-only --pose-model full
.\.venv\Scripts\python.exe app.py --video input.mp4 --pose-only --pose-model heavy --output results\heavy.mp4
```

`--landmark-threshold` adjusts the coverage threshold (default 0.5). Lite remains the default: full and heavy did not fix the crowded group-photo failure. All three performed better on an isolated person in the inspected samples. See `POSE_COMPARISON.md` for evidence and limitations.

## Model provenance and limits

Expression weights come from `model_filter.h5` in original commit `514316f2e5dce38a0e8ab35b50807f44fc57a2a4`. This is an early baseline, **not the missing FinalModel**. The original README credits the baseline architecture to Ashadullah Shawon. Its descriptive evaluation is subject to the unknown training-exposure caveat above.

Pose weights are the version-1 MediaPipe Pose Landmarker lite bundle, downloaded from Google's model hosting. See [Google's task documentation](https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker). Models are downloaded separately and retain their respective terms; this repository does not claim ownership of third-party models or imagery.

Expression labels describe model classifications of visible faces, not a person's internal emotional state. Scores are uncalibrated model probabilities. Pose and expression detections are independent: this version does not match faces to bodies, assign stable person IDs, or infer emotions from posture. The Haar face detector remains sensitive to lighting and head angle. Landmarks may be inaccurate for occluded people; only sufficiently visible/present points are drawn.

## Next milestones to build together

1. **External evaluation:** test on a separate, consented camera dataset and audit near-duplicates and identity overlap. FER2013 recovery, exact deduplication, and automatic reports are implemented.
2. **Better face detection:** measure failures on real camera footage, then compare a modern detector against the Haar baseline.
3. **Video behavior:** benchmark CPU latency/FPS, add stable tracking and temporal smoothing, and test camera capture on the target device.
4. **Training:** compare controlled architecture/augmentation changes and retain experiment configurations, seeds, metrics, and checkpoints.
5. **RGB expressions:** obtain a suitable labeled color dataset, adapt preprocessing and metadata, then compare against the grayscale classifier on a controlled evaluation.

See `VALIDATION.md` for the exact checks performed on the first milestone.
