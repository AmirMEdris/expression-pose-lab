# Pose comparison — 2026-11-10

## Finding

Returning landmarks was an execution check, not an accuracy check. The initial group-photo demo is a pose failure. Lite, full, and heavy each detected one pose but visually combined or misplaced joints across overlapping people. The full variant assigned high visibility even to estimated knees and ankles below the image boundary. Increasing model size or confidence thresholds alone does not resolve that failure.

The isolated full-body photo and a real 42-frame dance GIF produced much more plausible alignment. We visually inspected all three variants on the still images and five evenly spaced frames per variant from the dance clip (frames 0, 10, 20, 30, 41). This is qualitative inspection, not an accuracy benchmark; no manually labeled joint ground truth was available.

## Motion comparison

Source: https://raw.githubusercontent.com/tensorflow/tfjs-models/master/pose-detection/assets/dance_input.gif (the input clip used in TensorFlow's MoveNet tutorial). Frames were decoded with Pillow and passed to MediaPipe VIDEO mode using the GIF frame durations. This contains actual subject movement, unlike the earlier repeated-image smoke test. Maximum people was 1 for this single-person clip; image comparisons used 4.

| Variant | Frames with a pose | Frames passing full-body coverage | Median inference time | 95th percentile |
| --- | --- | --- | --- | --- |
| Lite | 42/42 | 42/42 | 21.6 ms | 24.3 ms |
| Full | 42/42 | 42/42 | 30.4 ms | 33.1 ms |
| Heavy | 42/42 | 42/42 | 100.1 ms | 115.7 ms |

Times are from this Windows CPU run, exclude model initialization, expression classification, drawing, file encoding, and camera capture, and are not guaranteed webcam FPS. Detection/coverage counts are not joint accuracy scores. Lite remains the default because these limited observations do not justify a blanket upgrade to heavy.

## Implemented response

- Added full/heavy selection and checksum-verified downloads.
- Withhold a default overlay if any major body joint is outside the image or below the visibility/presence threshold.
- Report withheld estimates and their missing joints; retain raw landmarks for investigation.
- Provide an explicitly labeled diagnostic override for partial skeletons.
- Added unit tests for incomplete, out-of-frame, nonfinite, and low-visibility landmarks.

All three group-photo estimates fail the coverage check; all three isolated full-body estimates pass. That prevents this particular misleading overlay. It does **not** establish that every displayed pose is correct, and it will suppress some useful partial-body results.

## Real next solution for groups

Evaluate a detector-and-pose system designed for multiple overlapping people, including person-box quality, per-person crops, identity association across frames, and labeled joint error. Cropping alone cannot recover occluded limbs and can still mix neighbors. Until that evaluation exists, the supported demo scenario should be one well-separated, fully visible person, with crowded scenes treated as experimental.

Inputs were downloaded for local testing and are not committed. The group sample comes from the original capstone; the isolated still comes from https://storage.googleapis.com/mediapipe-assets/pose.jpg. Model variants are Google's version-1 MediaPipe bundles. No webcam was opened.
