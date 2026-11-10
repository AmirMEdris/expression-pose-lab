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
