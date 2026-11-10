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
