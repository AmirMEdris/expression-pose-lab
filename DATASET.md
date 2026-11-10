# FER2013 recovery and evaluation protocol

Dataset source: [Manas Sambare's FER2013 distribution on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013), retrieved November 10, 2026. Original task: [Challenges in Representation Learning: Facial Expression Recognition](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge).

The downloaded archive contains 35,887 grayscale 48×48 images. Although it has only `train/` and `test/` folders, the test filenames retain `PublicTest_` and `PrivateTest_`. The loader reconstructs the original three partitions: training images, PublicTest validation, PrivateTest final evaluation. The data is not committed or bundled with the app. Obtain it from the source and review its terms before use or redistribution.

Decoded pixel hashes remove exact duplicate images within and across partitions. Priority is final test, validation, then training. The retained counts are 26,942 / 3,520 / 3,572 respectively. Removed counts are 1,767 / 69 / 17. This changes the official benchmark sample counts and must be disclosed in comparisons. No near-duplicate or identity-disjoint audit has been performed.

The image archive can be used directly; decoding it avoids tens of thousands of individual filesystem operations. CSV inputs require `emotion,pixels,Usage` with the original partitions. Folder datasets without original filename prefixes use a reproducible, class-stratified 15% validation split from training, leaving the supplied test partition untouched.

Training uses only the training partition and training-only horizontal flips/translation. Class weights are calculated from training labels. Validation loss selects checkpoints, early stopping, and learning-rate reductions. Final test labels do not tune these decisions. Every run writes its dataset fingerprint, seed, split counts, class counts, augmentation, runtime versions, checkpoint hash, learning curves, and per-example final predictions.

The recovered historical baseline's original training provenance is incomplete. Any evaluation of those old weights is descriptive; it cannot establish a clean, leakage-free baseline comparison. Dataset accuracy also does not establish live-camera performance or a person's internal emotional state.
