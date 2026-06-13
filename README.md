# Brain MRI Tumor Segmentation (U-Net)

A U-Net that contours lower-grade glioma in brain MRI — built from scratch in PyTorch, scoring **0.914 Dice on held-out patients the model never saw during training**.

I do clinical ML for my day job, but I'd never built an image model end-to-end and I wanted to actually understand the pieces instead of importing them. So I picked the one task in radiology that I find genuinely cool: taking a scan and turning it into a region you can point at. The model takes a brain MRI slice, predicts which pixels are tumor, and draws the outline back onto the image.

If you've never touched segmentation, here's the one-sentence version: instead of classifying a *whole* image ("tumor / no tumor"), you classify *every pixel* — and the outline of all the "tumor" pixels is the contour. Drawing those contours by hand is slow, so a model that proposes one is a nice time-saver for the person reviewing the scan.

### Example output

Held-out test slices — radiologist ground truth in red, model prediction in green:

![sample predictions](assets/sample_predictions.png)

![training curves](assets/training_curves.png)

## Results

Trained for 50 epochs on a single GPU; the saved checkpoint is the best epoch by validation Dice.

| Split | Dice | IoU |
|-------|------|-----|
| Validation | 0.916 | 0.885 |
| Test | 0.914 | 0.885 |

**Dice** is the number to watch here — it's the overlap between the predicted region and the real one (0 = no overlap, 1 = perfect). Mathematically it's the F1 score computed per pixel, which makes it a fair scoring method when the thing you're looking for is small. **IoU** is the same idea but stricter.

The part I actually care about is that validation and test are basically tied (0.916 vs 0.914). That gap staying near zero is what tells me the network learned what a tumor looks like instead of memorizing the training slices — more on why that's not guaranteed below.

## How it works

A standard U-Net — a ~31M-parameter encoder/decoder. The encoder shrinks the image down while pulling out features; the decoder blows it back up to a full-resolution mask. The trick is the **skip connections** that wire each encoder stage straight across to its matching decoder stage — they carry the fine spatial detail that the downsampling throws away, which is what keeps the predicted edge crisp instead of a smeared blob. Input is the three co-registered MRI sequences (pre-contrast, FLAIR, post-contrast); output is a single binary tumor mask.

Three decisions did most of the work:

- **Patient-level splitting.** This is the one that bit me conceptually. Neighboring MRI slices of the same brain look almost identical, so if you shuffle every slice and split randomly, near-duplicates land in both train and test — the model "sees" the test set during training and your Dice comes out beautiful and meaningless. The fix is to split by *patient*: every slice from one person goes entirely to train, val, or test. It's why the test number is trustworthy.
- **BCE + Dice loss.** Tumor pixels are a tiny minority of every slice, so a plain pixel-wise loss is happy to predict "all background" and still score high. Pairing binary cross-entropy with a Dice term keeps those rare positive pixels actually driving the gradient.
- **Augmentation** — flips, rotations, grid distortion, brightness/contrast — to stretch ~110 patients into enough variety that the network can't just overfit them.

The full architecture walkthrough, and the things that tripped me up, are in [`docs/how-it-was-built.md`](docs/how-it-was-built.md).

## What I took away

- Most of the "accuracy" in medical imaging is decided before the model — in how you split the data and which metric you trust. I could've had a great-looking 0.95 Dice that meant nothing if I'd split slices randomly, and I wouldn't have known.
- Pixel accuracy is a trap when the target is rare. Dice/IoU are the honest numbers.
- Building the U-Net by hand instead of importing one made the skip connections click in a way no diagram did — they're literally the difference between a sharp boundary and a blurry guess.

Contouring (outlining a structure on a scan) is something radiologists and radiation-oncology teams do by hand a lot, so this is the kind of task where a model that hands over a first draft is genuinely useful — keep the human in the loop, let the model do the tedious first pass.

## Dataset

[Brain MRI segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) (Buda et al.), from the TCGA lower-grade glioma collection — ~110 patients, ~3,900 slices of 256×256 TIFFs with expert binary tumor masks. The data is pulled separately (see below) and is not committed to the repo.

## Run it yourself

**Colab (free GPU)** — open [`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb), set the runtime to GPU, and run top to bottom. It installs dependencies, loads the dataset, trains, plots curves, draws prediction overlays, and exports the weights.

**Locally:**

```bash
pip install -r requirements.txt

# dataset (needs a Kaggle API token at ~/.kaggle/kaggle.json)
python scripts/download_data.py --out data

# train
python -m src.train --data-dir data/lgg-mri-segmentation \
    --epochs 50 --batch-size 16 --lr 1e-4 --out runs/exp1

# predict on one slice
python -m src.inference --weights runs/exp1/best_model.pt \
    --image path/to/slice.tif --out prediction.png
```

To sanity-check the pipeline with no data at all, `python -m src.train --smoke-test --out runs/smoke` runs the whole loop on synthetic tensors in a few seconds.

## What's in here

```
src/
  model.py        U-Net architecture
  data.py         dataset, patient-level split, augmentation
  losses.py       BCE+Dice loss, Dice/IoU metrics
  train.py        training loop (CLI), mixed precision, checkpointing
  inference.py    load checkpoint -> predict -> contour overlay
  utils.py        seeding, checkpoints, overlay rendering
scripts/          Kaggle download helper
notebooks/        end-to-end Colab training notebook
docs/             architecture and design write-up
app/              interactive demo (in progress)
```

## Roadmap

- [x] U-Net training pipeline, patient-level evaluation, Colab notebook
- [x] Inference + contour-overlay API
- [ ] Interactive web demo — upload a slice, get the contour back (Gradio on Hugging Face Spaces)
- [ ] Pretrained-encoder backbone for a few more Dice points
- [ ] ONNX export for in-browser inference

## A note on scope

Research and education only. This is not a medical device and nothing here is cleared for clinical use.

## References

- Ronneberger, Fischer, Brox. *U-Net: Convolutional Networks for Biomedical Image Segmentation.* MICCAI 2015.
- Buda, Saha, Mazurowski. *Association of genomic subtypes of lower-grade gliomas with shape features automatically extracted by a deep learning algorithm.* Computers in Biology and Medicine, 2019.

## License

MIT — see [LICENSE](LICENSE).
