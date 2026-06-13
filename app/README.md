# Interactive demo

A small Gradio app that runs the trained U-Net on a brain-MRI slice and draws the
predicted tumor contour back onto it. Upload your own slice or pick a bundled
example, nudge the decision threshold, and you get the overlay plus the tumor-area
fraction. It imports `predict_overlay` straight from `src/inference.py` — the
model code isn't rewritten for the front end.

The weights (`best_model.pt`, 124 MB) aren't in git. The app pulls them once at
startup from a Hugging Face **model repo**, so the Space itself stays small.

## Run it locally

From the repo root, with the project deps plus Gradio installed:

```bash
pip install -r app/requirements.txt
# point the app at the local checkpoint so it skips the download
WEIGHTS_PATH=best_model.pt python app/app.py
```

That serves it at `http://127.0.0.1:7860`.

## Deploy to Hugging Face Spaces

Two repos: a **model repo** holds the weights, a **Space** runs the app and
downloads them.

**1. Push the weights to a model repo** (one time):

```bash
pip install huggingface_hub
huggingface-cli login
huggingface-cli upload shibdaddev/brain-mri-lgg-unet best_model.pt best_model.pt --repo-type model
```

If you name the repo something other than `shibdaddev/brain-mri-lgg-unet`, set
`HF_MODEL_REPO` in the Space's settings to match.

**2. Create a Gradio Space** (`shibdaddev/brain-mri-tumor-contouring`, SDK: Gradio)
and put these at its root:

```
app.py            <- this folder's app.py
requirements.txt  <- this folder's requirements.txt
src/              <- copy of the repo's src/ package
examples/         <- a few .png slices (optional but nice)
```

The Space installs `requirements.txt`, runs `app.py`, pulls the weights from the
model repo on first boot, and serves the link.

## Example slices

Drop 3–4 test-set slices (`.png`) into `examples/` so people without their own
MRI can click straight through. Grab them from the dataset — any image from a
patient in the held-out test split is a fair demo.
