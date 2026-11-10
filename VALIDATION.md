# Validation — 2026-11-10

Tested locally on Windows with Python 3.13.7 and CPU inference. Exact package versions are recorded in requirements-lock.txt.

- Dependency consistency: pip check passed.
- Unit tests: 8 passed (preprocessing scale/shape, empty input, label order, uncertain output, malformed probabilities).
- Model downloader: both pinned artifacts passed SHA-256 verification.
- Combined image test: recovered capstone ABBA sample produced 4 expression detections and 1 pose with 33 finite landmarks. All four expression probability vectors matched the previously recovered baseline within absolute tolerance 1e-5.
- Blank image: no faces and no poses; output saved successfully.
- Combined video path: a three-frame clip made by repeating the group image decoded and wrote 3 output frames, with strictly increasing timestamps and JSONL output. Pose detection missed all people in this compressed crowded clip.
- Pose video path: a three-frame clip made by repeating Google's full-body sample returned 1 pose on every frame, at timestamps 0, 200, and 400 ms. The run processed those frames in 0.19 seconds after initialization. This is a smoke test, not a real-time or moving-video benchmark.
- Combined image overlay was visually checked.

Sample sources (downloaded only for local validation, not committed):
- ABBA: original capstone commit 514316f2e5dce38a0e8ab35b50807f44fc57a2a4, abba.png.
- Full-body sample: https://storage.googleapis.com/mediapipe-assets/pose.jpg

Not yet verified: live webcam capture, moving-person tracking, multi-person pose accuracy, model accuracy/calibration, training, other operating systems. A maximum-person setting does not guarantee all people will be detected. Crowded and partially visible bodies remain a known limitation.

The original tf_pose implementation was replaced; it was not restored or tested. The expression model remains the recovered early baseline, not FinalModel.

## Follow-up: pose quality review

The initial group-photo skeleton is a qualitative failure; successful execution did not establish correct alignment. See POSE_COMPARISON.md for the lite/full/heavy comparison and actual moving-footage test. The updated default withholds incomplete bodies, with explicit diagnostic access to raw estimates. The test suite now contains 18 passing unit tests.

## Desktop camera launch

The Python process launched through the assistant execution environment could not open camera indices 0-2 through DirectShow or Media Foundation. Windows Camera displayed a live feed. After the user ran launch_camera.cmd from the normal Windows desktop, the user confirmed the live camera window opened; its application window was also observed in the desktop inventory. The assistant did not inspect the live model overlay before that window closed. Camera-frame acquisition/display is user-confirmed; model accuracy on that feed is not verified.

Added explicit backend fallback, first-frame validation, cleanup on failed attempts, and a desktop launcher. 20 unit tests pass, including mocked camera-backend fallback and all-backend failure cleanup. No camera recording was saved by the launcher.

## Desktop, visualization, GPU and evaluation work

- 34 fast tests pass, covering original live controls, source switching, crop capture, RGB/BGR conversion, metrics orientation, dataset deduplication, and report-to-model checksum matching.
- GPU verified: PyTorch 2.11.0+cu128, Keras 3.15.1, RTX 5070 Laptop GPU. CUDA matrix multiplication, expression inference, activation maps, and DeepDream gradients passed.
- CPU/GPU recovered-model comparison: eight deterministic inputs, maximum probability difference 1.55e-6, all predicted labels identical with TF32 disabled. This is numerical compatibility, not an accuracy benchmark.
- Source smoke test passed for model loading, Tk initialization, visualization generation, and MediaPipe blank-image inference. The GPU-capable app continues to run pose estimation on CPU.
- Compact candidate: saved best validation-loss checkpoint from an interrupted 27/30-epoch run. Validation accuracy 56.36%, macro F1 50.40%; test accuracy 59.07%, macro F1 53.72%. The test partition has 3,572 images after exact-pixel deduplication. No claim of live-camera accuracy or superiority to the historical model.
- Spatial candidate failed to learn reliably; it is not the recommended checkpoint.
- Recovered baseline evaluation: 92.81% accuracy, 92.21% macro F1 on this test partition, **not a verified held-out result** because its original training provenance is unknown. Its bundled report states this caveat explicitly.
- The app defaults to the recovered baseline. Results are attached to model hashes; missing or mismatched reports produce an explanation instead of a file picker or unrelated metrics.
- The consumer UI has no training controls. Internal layer names are replaced with explained depth choices; iteration counts live behind explained processing presets. Feature maps and Dream default to grayscale. Added Dream colors are explicitly optional artistic mapping.

## Model-driven Dream, Grad-CAM, and residual training

- 42 fast tests cover metrics, dataset handling, source switching, preview aspect ratios, class-specific heatmap weighting, and zero/negative evidence handling.
- Dream uses only the loaded expression classifier’s convolution weights, cloned to accept larger images. Source chroma can be restored after luminance ascent; no separate ImageNet network is used. Classification preprocessing remains 48×48 grayscale.
- Grad-CAM targets class logits where the supported model head exposes them, with a probability-gradient fallback for other heads. It leaves classification weights and predictions unchanged, supports selecting a class, and labels its relative, coarse nature. Missing faces never receive a fabricated heatmap.
- Capture is capped at 30 FPS, visualization work uses a separate bounded queue, still-image face inference is cached, and pose loads lazily. These resource controls are not a guaranteed frame-rate claim.
- The residual run completed all 50 epochs with 1,773,735 parameters. Selected-checkpoint test accuracy: 62.1501%; macro F1: 58.3602%. The earlier compact candidate scored 59.0705% and 53.7231% on the same 3,572-image partition. This is not an independent external benchmark; the old baseline’s unknown training exposure still prevents a clean comparison with it.
- Privacy audit of tracked files and history found only the synthetic documentation gallery as added media. No camera photo, face crop, recording, training image, or attached screenshot is committed.

## Held-frame development

- Stronger mode holds the first input while progressing through 362 coarse-to-fine updates, combining the three deepest convolution layers of the loaded expression model. The lighter live effect remains selectable.
- Full development completed on the public historical sample; the completed result stayed unchanged when a different input was submitted. Reset clears the held frame. The full source check took about 30 seconds including initialization on the verified GPU, with a 0.15-second median chunk time; this is not a guaranteed camera FPS.
- Synthetic-input smoke checks verify held-frame isolation and reset. Visual comparisons show larger distortions than the prior live settings; subjective resemblance to faces is not an accuracy measurement. Public-sample previews remain local and untracked.
