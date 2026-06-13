# How it was built

The longer version — the decisions behind the model, and the parts of medical image segmentation that don't show up in a tutorial.

## Framing the problem

Contouring means drawing the boundary of a structure on a scan. I framed it as binary semantic segmentation: for every pixel of an MRI slice, decide tumor or not-tumor, then the contour is just the edge of that region.

I went with 2D slice-wise segmentation rather than full 3D volumes for three reasons. The LGG dataset ships as 2D slices to begin with; a 2D U-Net trains in well under an hour on one consumer GPU; and a 2D model drops straight into an interactive web demo later — one image in, one overlay out. For a portfolio piece that needs to *show* something, that last point mattered.

## The data, and the trap inside it

The dataset is Buda et al.'s *Brain MRI segmentation* — the lower-grade glioma collection from TCGA, ~110 patients and ~3,900 slices. Each slice is a 256×256 TIFF with three channels (pre-contrast, FLAIR, post-contrast MRI sequences), paired with a binary mask where a radiologist outlined the FLAIR abnormality.

The trap: adjacent slices of the same brain look almost identical. If you shuffle all the slices and split randomly, near-duplicate slices land in both train and test, the model effectively sees the test data during training, and your Dice score comes out beautiful and meaningless. So the split happens at the **patient** level — every patient's slices go entirely to train, validation, or test, never spread across them. It's the single most important correctness decision in the project, and it's why the test Dice (0.914) lands right on top of the validation Dice (0.916) instead of collapsing on data the model has actually never seen. `split_by_patient` in `src/data.py` does this with a seeded shuffle of patient IDs — no scikit-learn dependency, just deterministic grouping.

The other data reality is class imbalance. Most slices contain little or no tumor, and tumor pixels are a small minority everywhere else. A model that predicts "all background" scores high pixel accuracy and is clinically worthless. The fix is partly in the loss (below) and partly in evaluation — I report Dice and IoU, never pixel accuracy, because accuracy would flatter the model for doing nothing.

## The model

A from-scratch U-Net in `src/model.py` — I implemented it directly rather than importing a prebuilt library so the repo actually demonstrates the architecture.

The shape is the classic "U": an encoder that halves spatial resolution and doubles channels at each stage, a bottleneck, and a decoder that upsamples back to full resolution. The skip connections are the whole point — they carry high-resolution spatial detail from the encoder across to the matching decoder stage, which is what lets the network produce a sharp boundary instead of a blurry blob. Each block is two conv layers with batch norm and ReLU; the whole thing is about 31M parameters.

One detail worth calling out: if an input dimension isn't divisible by 16, the pooling and upsampling can leave a one-pixel size mismatch at a skip connection. `forward()` checks for that and interpolates the upsampled tensor to the skip's exact size before concatenating — cheap insurance against a crash on oddly-sized inputs.

## Loss and metrics

The loss is BCE plus Dice, weighted evenly (`BCEDiceLoss` in `src/losses.py`). Binary cross-entropy gives stable pixel-wise gradients everywhere; the soft Dice term optimizes region overlap directly and is far more robust to the imbalance. It's a well-worn pairing for medical segmentation, and it works here. Every loss and metric function takes raw logits and applies the sigmoid internally, so there's no way for a caller to accidentally activate twice.

For evaluation I use hard Dice and IoU at a 0.5 threshold — Dice is the harmonic-mean-flavored overlap (mathematically the per-pixel F1 score), IoU is its stricter cousin.

## Training

Adam at 1e-4, `ReduceLROnPlateau` watching validation Dice (halve the rate after five stagnant epochs), mixed precision on GPU for speed and memory, and augmentation through albumentations — flips, 90° rotations, shift/scale/rotate, grid distortion, brightness and contrast, applied identically to image and mask. Checkpoint the best validation Dice, log every epoch to `metrics.csv`. There's also a `--smoke-test` mode that runs the entire loop on tiny synthetic tensors so the pipeline can be verified in seconds without downloading anything.

The training curve is unremarkable in the best way: validation Dice climbs into the high 0.8s by epoch 25, the learning-rate drops kick in around epoch 31, and it settles into the low 0.9s, peaking at epoch 45.

## Inference and the contour

`src/inference.py` loads a checkpoint, runs the forward pass, applies sigmoid and a threshold, resizes the mask back to the input resolution, and hands off to `overlay_mask` in `src/utils.py`, which traces the boundary with `cv2.findContours` and lays it over the original with a translucent fill. `predict_overlay` returns the overlay and the tumor area fraction. That's the exact function the web demo calls — the model code doesn't get reimplemented for the front end.

## Where I'd take it next

- A pretrained ResNet or EfficientNet encoder (via `segmentation-models-pytorch`) would likely buy a few Dice points over the from-scratch encoder.
- Test-time augmentation and a boundary-aware loss (Tversky or focal) for the harder, smaller lesions.
- 3D context — moving to BraTS volumes with a 3D or 2.5D U-Net.
- Calibration and uncertainty (MC-dropout or an ensemble) to flag low-confidence contours, which is what you'd actually want before this went anywhere near a clinical workflow.

A reminder on that last point: this is research and education only — not a medical device, and not for clinical use.
