# How it was built

A deeper engineering walkthrough of the brain-MRI tumor contouring model — the
decisions, the architecture, and the things that bite you in medical image
segmentation.

## 1. Problem framing

"Contouring" in radiology means delineating a structure's boundary. We frame it
as **binary semantic segmentation**: for every pixel of an MRI slice, predict
tumor (1) vs. not-tumor (0). The boundary contour is then just the edge of the
predicted region.

We chose 2D slice-wise segmentation (not 3D volumes) because:

- The LGG dataset is distributed as 2D `.tif` slices.
- 2D U-Nets train fast on a single consumer/Colab GPU.
- A 2D model is trivial to demo in a browser later (one image in, one overlay out).

## 2. The data

**Dataset:** *Brain MRI segmentation* (Buda et al., 2019), sourced from The
Cancer Imaging Archive (TCGA) lower-grade glioma collection. ~110 patients,
~3,900 slices total.

Each slice is a 256×256, 3-channel TIFF where the channels are three MRI
sequences (pre-contrast, FLAIR, post-contrast). The paired `*_mask.tif` is a
binary expert annotation of the FLAIR abnormality (the tumor).

### Pitfall: data leakage across slices

Adjacent slices of the same brain look almost identical. If you split slices
randomly, near-duplicate slices land in both train and test, and your reported
Dice is inflated. We split **by patient** (`split_by_patient` in `src/data.py`):
whole patients go to exactly one of train/val/test. This is the single most
important correctness decision in the project.

### Pitfall: class imbalance

Most slices contain little or no tumor; tumor pixels are a small minority. A
naive model that predicts "all background" scores high pixel accuracy and zero
clinical value. Two mitigations:

1. Keep negative (no-tumor) slices in training so the model learns normal
   anatomy — but pair a Dice-based loss with BCE so the rare positive pixels
   still drive the gradient.
2. Evaluate with **Dice** and **IoU**, not pixel accuracy.

## 3. The model — U-Net

We implement U-Net (Ronneberger et al., 2015) from scratch in `src/model.py`
rather than importing a prebuilt library, so the repo demonstrates the
architecture.

```
Input 3x256x256
  └─ DoubleConv 64  ─────────────skip────────────┐
       ↓ maxpool                                 │
     DoubleConv 128 ───────────skip──────────┐   │
       ↓ maxpool                             │   │
     DoubleConv 256 ─────────skip────────┐   │   │
       ↓ maxpool                         │   │   │
     DoubleConv 512 ───────skip──────┐   │   │   │
       ↓ maxpool                     │   │   │   │
     Bottleneck 1024                 │   │   │   │
       ↑ upconv + concat ───────────┘   │   │   │
     DoubleConv 512                      │   │   │
       ↑ upconv + concat ───────────────┘   │   │
     DoubleConv 256                          │   │
       ↑ upconv + concat ───────────────────┘   │
     DoubleConv 128                              │
       ↑ upconv + concat ───────────────────────┘
     DoubleConv 64
       ↓ 1x1 conv
Output 1x256x256 (logits)
```

- **Encoder** halves spatial resolution and doubles channels at each stage,
  building increasingly abstract features.
- **Skip connections** carry high-resolution spatial detail from encoder to
  decoder, which is what lets U-Net produce sharp boundaries.
- **Decoder** upsamples with transpose convolutions and fuses the skips.
- BatchNorm after each conv stabilizes training; ~31M parameters total.

A small but important detail: if an input dimension isn't divisible by 16, the
pooling/upsampling can produce a one-pixel size mismatch at a skip connection.
`forward()` guards against this by interpolating the upsampled tensor to the
skip's exact size before concatenation.

## 4. Loss and metrics

`src/losses.py`:

- **BCEWithLogits** — stable, pixel-wise; good gradients everywhere.
- **Soft Dice loss** — directly optimizes region overlap; robust to imbalance.
- We sum them 0.5/0.5 (`BCEDiceLoss`). This pairing is a well-worn default for
  medical segmentation.
- **Metrics:** hard Dice coefficient and IoU on thresholded (0.5) predictions.

All loss/metric functions take raw logits and apply the sigmoid internally, so
callers can never accidentally double-activate.

## 5. Training setup

`src/train.py`:

- Optimizer: Adam, lr 1e-4.
- Scheduler: `ReduceLROnPlateau` on validation Dice (halve LR after 5 stagnant
  epochs).
- Mixed precision (`torch.amp`) on GPU for speed/memory; automatically disabled
  on CPU.
- Augmentation (`albumentations`): flips, 90° rotations, shift/scale/rotate,
  grid distortion, brightness/contrast — applied identically to image and mask.
- Checkpointing: save `best_model.pt` whenever validation Dice improves; log
  every epoch to `metrics.csv`.
- A `--smoke-test` mode runs the whole loop on tiny synthetic tensors with no
  dataset, so the pipeline can be verified in seconds on any machine.

Expected performance for this dataset/architecture is roughly **0.85–0.90 Dice**
on held-out patients after ~50 epochs (fill in your actual number after the run).

## 6. Inference and contouring

`src/inference.py` loads a checkpoint, runs a forward pass, applies
sigmoid + threshold, resizes the mask back to the input resolution, and
`src/utils.overlay_mask` draws the boundary contour (via `cv2.findContours`)
with a translucent fill. `predict_overlay` returns the overlay plus the tumor
area fraction. This is the exact function the Gradio demo will call.

## 7. What I'd do next

- **Stronger backbone:** swap the from-scratch encoder for a pretrained ResNet/
  EfficientNet encoder (`segmentation-models-pytorch`) for a few extra Dice
  points.
- **Test-time augmentation** and boundary-aware losses (e.g. Tversky / focal).
- **3D context:** move to BraTS volumes with a 3D U-Net or 2.5D stacking.
- **Calibration & uncertainty:** MC-dropout or ensembles to flag low-confidence
  contours — important for any clinical-adjacent use.

> **Disclaimer:** research/education only. Not a medical device and not for
> clinical use.
