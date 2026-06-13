# Brain MRI Tumor Contouring — U-Net (LGG FLAIR)

A deep-learning model that **contours (segments) tumors in brain-MRI slices**,
trained on the lower-grade glioma (LGG) FLAIR dataset. Built from scratch in
PyTorch with a clean, reproducible training pipeline and a one-click Colab
notebook.

> ⚕️ **Disclaimer:** This is a research / educational project. It is **not** a
> medical device and must not be used for diagnosis or clinical decisions.

![python](https://img.shields.io/badge/python-3.10%2B-blue)
![pytorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![license](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Given a brain-MRI slice, the model predicts a per-pixel tumor mask and draws the
tumor **contour** over the image:

```
   MRI slice  ──►  U-Net  ──►  tumor probability  ──►  threshold  ──►  contour overlay
 (3×256×256)                     (1×256×256)                          + area estimate
```

The three input channels are three MRI sequences (pre-contrast / FLAIR /
post-contrast); the output is a binary tumor mask.

### Example output

Held-out **test** slices — radiologist ground truth (red) vs. model prediction (green):

![sample predictions](assets/sample_predictions.png)

### Training curves

![training curves](assets/training_curves.png)

## Dataset

[**Brain MRI segmentation**](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation)
(Buda et al.), from the TCGA lower-grade glioma collection — ~110 patients,
~3,900 slices of 256×256 TIFFs with expert binary tumor masks. The data is
downloaded separately (see Quickstart) and is **not** committed to the repo.

## Method (in one paragraph)

A classic **U-Net** (encoder–decoder with skip connections, ~31M parameters)
implemented from scratch. Trained with a combined **BCE + Dice** loss to handle
the heavy class imbalance, Adam + `ReduceLROnPlateau`, mixed-precision, and
augmentation (flips/rotations/distortions). Critically, the train/val/test split
is **patient-level** to prevent near-duplicate adjacent slices from leaking
across splits. Metrics: **Dice** and **IoU**. See
[`docs/how-it-was-built.md`](docs/how-it-was-built.md) for the full walkthrough.

## Results

U-Net trained for 50 epochs on a single GPU. The checkpoint is selected by best
**validation Dice** (epoch 45). The split is patient-level, so the test set is
patients the model never saw during training.

| Split | Dice | IoU |
|-------|------|-----|
| Validation | **0.9160** | 0.8845 |
| Test | _pending_ | _pending_ |

<sub>Metrics are mean per-slice Dice / IoU at a 0.5 threshold.</sub>

## Repository structure

```
brain-mri-unet/
├── src/
│   ├── model.py        # U-Net architecture (from scratch)
│   ├── data.py         # dataset, patient-level split, augmentation
│   ├── losses.py       # BCE+Dice loss, Dice/IoU metrics
│   ├── train.py        # training loop (CLI), AMP, checkpointing
│   ├── inference.py    # load checkpoint → predict → contour overlay
│   └── utils.py        # seeding, checkpoints, overlay rendering
├── scripts/download_data.py   # Kaggle dataset download helper
├── notebooks/train_colab.ipynb # one-click end-to-end training on Colab
├── app/                # interactive Gradio demo (next iteration — stubbed)
├── docs/how-it-was-built.md    # deep-dive write-up
└── assets/             # generated figures for this README
```

## Quickstart

### Option A — Google Colab (recommended, free GPU)

Open `notebooks/train_colab.ipynb` in Colab, set the runtime to **GPU**, and run
the cells top to bottom. It installs deps, downloads the data, trains, plots
curves, and exports the weights.

### Option B — Local

```bash
# 1. Install
pip install -r requirements.txt

# 2. Download the dataset (needs a Kaggle API token at ~/.kaggle/kaggle.json)
python scripts/download_data.py --out data

# 3. Train
python -m src.train --data-dir data/lgg-mri-segmentation \
    --epochs 50 --batch-size 16 --lr 1e-4 --out runs/exp1

# 4. Predict on a single slice
python -m src.inference --weights runs/exp1/best_model.pt \
    --image path/to/slice.tif --out prediction.png
```

### Verify the pipeline without any data

A synthetic smoke test exercises the full training loop in seconds on CPU:

```bash
python -m src.train --smoke-test --out runs/smoke
```

## Roadmap

- [x] Reproducible U-Net training pipeline + Colab notebook
- [x] Inference + contour-overlay API
- [ ] **Interactive Gradio web demo on Hugging Face Spaces** (upload a slice,
      threshold slider, live contour overlay) — see [`app/README.md`](app/README.md)
- [ ] Pretrained-encoder backbone for higher Dice
- [ ] ONNX export for in-browser inference
- [ ] 3D / BraTS extension

## References

- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical
  Image Segmentation.* MICCAI 2015.
- Buda, Saha, Mazurowski. *Association of genomic subtypes of lower-grade
  gliomas with shape features automatically extracted by a deep learning
  algorithm.* Computers in Biology and Medicine, 2019.

## License

MIT — see [LICENSE](LICENSE).
