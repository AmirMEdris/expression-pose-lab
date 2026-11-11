# CanvasNet RGB

A custom expression CNN trained from random initialization on RGB faces. The same exported checkpoint drives expression probabilities, filter inspection, Grad-CAM, and DeepDream. There is no pretrained backbone or separate art network.

The first run completed 28 epochs, with epoch 20 selected by validation macro F1. Its held-out ExpW results are **68.23% accuracy / 51.62% macro F1** on 6,472 faces. The previous FER2013 grayscale residual scores **47.14% / 33.22%** on those same images. Training data and resolution differ, and cross-dataset exposure for that prior model is unaudited, so this does not isolate the architecture’s contribution. Disgust (8.7% F1) and fear (32.7% F1) remain weak. The app export agrees with the PyTorch checkpoint within 1.2e-7 maximum probability difference on the eight-example export check.

## Architecture

Input is 224 × 224 RGB in [0, 1], normalized to [-1, 1] inside the model. There are **5,045,407 trainable parameters**. Standard residual blocks are combined into this project's design; this is not a claim of a new research architecture.

| Stage | Feature size | Channels | Blocks | Spatial dropout |
| --- | --- | --- | --- | --- |
| Stem | 224 → 112 → 56 | 24, 32 | Two 3×3 convolutions, each followed by normalization, ReLU, average pooling | — |
| Fine structure | 56 × 56 | 48 | Two residual blocks, two 3×3 convolutions each | 5% |
| Mid-level structure | 28 × 28 | 96 | Two residual blocks | 10% |
| Broad structure | 14 × 14 | 192 | Two residual blocks | 15% |
| Deep structure | 7 × 7 | 320 | Two residual blocks | 20% |
| Expression head | 192 + 320 pooled features → 192 → 7 | — | Two-scale concatenation, dense ReLU, softmax output | 35% head dropout |

Each deeper stage begins with average pooling. Residual projections use 1×1 convolutions when channel counts change. Batch normalization uses epsilon 1e-5. The middle feature branch has 15% spatial dropout before its global average. Dropout is disabled for validation, inference, and Dream.

The two-scale head makes both intermediate and deep features contribute to expression predictions. Gradual downsampling retains several spatial grids to explore. RGB input permits learned color sensitivity. These choices create opportunities for richer visualizations; they do not guarantee beautiful Dreams, recognizable parts, or improved accuracy.

## Dream design

The RGB path optimizes all three input channels through identical classifier weights. It develops a held input over 112 / 224 / output-resolution scales, with 160 total updates. Mixed mode targets the feature outputs of stages 2–4. One-filter mode retains exact convolution-layer and channel selection.

Small translations, total-variation regularization, and gradients smoothed at multiple radii suppress some high-frequency noise. Optional **Keep image structure** bounds each RGB channel within 0.18 of the source in normalized units. These are explicit image priors, not learned features. Turning the constraint off permits stronger transformations. **Show color in Dream** displays the learned RGB changes; switching it off displays grayscale.

The grayscale models retain their previous luminance-only path and original-chroma restoration. Displaying that path in color does not make those models color-aware.

Every ten training epochs, `preview_canvas.py` uses the currently selected checkpoint to render the same four validation examples with identical settings, constrained and unconstrained. The local `dream-progress.html` lets the viewer compare actual results across checkpoints. Checkpoint selection remains validation macro F1; no fabricated beauty score is used. Dataset labels and geometric support boxes do not establish that a filter is an eye or mouth detector.

Background: [Feature Visualization](https://distill.pub/2017/feature-visualization/) discusses frequency regularization, transformation robustness, and the tradeoffs of image priors. Stronger priors may make results easier to recognize while adding assumptions to the visualization.

## ExpW protocol

The [authors' project page](https://mmlab.ie.cuhk.edu.hk/projects/socialrelation/index.html) supplies the archive and seven-class annotations. This experiment keeps original RGB, uses a 10% face-box margin, and excludes unreadable, nearly constant, or native face boxes smaller than 64 pixels on either side. Resizing small faces to 224 pixels does not recover missing detail. The audit records native size percentiles and measured color fraction.

Original source images, exact image hashes, exact resized crop hashes, and crop pHashes within Hamming distance 3 form groups. Identical crops with conflicting labels are excluded; other exact duplicates are reduced to one example. Stratified group folds create a fixed approximate 80/10/10 train/validation/test split. This is a **custom split, not an official ExpW benchmark**. The groups prevent detected source/duplicate leakage, but cannot guarantee identity separation or detect every near-duplicate. There are no identity labels.

Training uses CUDA bfloat16 autocast with float32 weights, AdamW, three warmup epochs and cosine decay, modest geometric/color augmentation, 5% label smoothing, square-root class weights capped at 3 before normalization, and gradient clipping. Maximum 40 epochs; early stopping is allowed after epoch 20 with eight epochs without validation macro-F1 improvement. Best and latest checkpoints are retained. The final test is evaluated only after checkpoint selection, in float32. No claim of better expression performance is made before evaluation.

The final report includes learning curves, confusion matrix, per-class precision/recall/F1, calibration diagnostics, and dataset audit. Export parity checks PyTorch probabilities against the Keras checkpoint consumed by the desktop app. The `.keras` hash binds its preprocessing metadata and report to those exact weights.

`rgb_filter_explorer.py` ranks real validation examples for every convolution channel and finds weaker controls matched in global brightness and contrast. It saves lossless 224×224 RGB evidence in a local assets directory, loaded only when a filter is selected. Unlike the grayscale explorer, this report does not run three-seed Dream tests for every filter; its instructions connect the exact layer/filter IDs to live Dream. The report and checkpoint hashes are verified by the app; the manifest also records individual PNG hashes for auditing.

## Run and monitor

Install the CUDA environment from `requirements-training.txt`. Obtain the official archive with `download_expw.py`, extract with `extract_expw.py`, and prepare with `prepare_expw.py`. The downloader records sizes and SHA-256 locally; these are not author-published checksums. Extraction validates entry paths and CRCs. An optional local `data/tools/7zr.exe` from [7-Zip](https://www.7-zip.org/download.html) speeds extraction; otherwise Python handles the archive.

```powershell
python download_expw.py
python extract_expw.py
python prepare_expw.py
python train_canvas.py
# In another window:
python training_progress.py experiments/canvas-rgb-v1
# After training, generate real filter evidence for the selected RGB checkpoint:
python rgb_filter_explorer.py experiments/canvas-rgb-v1/best.keras data/expw --output experiments/canvas-rgb-v1/best.filters.html
```

The monitor shows stages, batch progress, loss, validation accuracy/F1, GPU memory, and links to evolving Dreams and the final report. Closing it leaves training running. `python train_canvas.py --resume` continues the latest completed epoch with the same configuration. A crash inside an epoch restarts that epoch; worker augmentation order is not guaranteed identical after resume.

Data, checkpoints, generated dataset galleries, and camera captures remain local and ignored by Git. No explicit redistribution license was found in the supplied ExpW material, so this project does not redistribute the dataset or its trained weights. Source builds work with the recovered baseline; a locally installed `models/expression-rgb.keras` plus matching metadata becomes the default color model, while **Original model** remains available.
