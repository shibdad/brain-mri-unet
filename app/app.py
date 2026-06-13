"""Gradio demo for the brain-MRI U-Net.

Upload an MRI slice (or pick a bundled example) and get back the slice with the
predicted tumor contour drawn on it, plus the tumor-area fraction. All the model
work is reused from `src.inference` — nothing about the network is reimplemented
here.

The weights aren't committed to git (124 MB); they're pulled once at startup from
a Hugging Face model repo. Override either piece with env vars:

    HF_MODEL_REPO   model repo to pull best_model.pt from (default below)
    WEIGHTS_PATH    use a local checkpoint instead of downloading (handy for dev)
"""
from __future__ import annotations

import glob
import os
import sys

import gradio as gr

# Make `src` importable whether this file sits at the repo's app/ dir (dev) or at
# the root of a Hugging Face Space alongside a copied src/ (deploy).
_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE, os.path.dirname(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from src.inference import load_model, predict_overlay  # noqa: E402

DEVICE = "cpu"  # HF Spaces free tier is CPU-only; the model is small enough.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "shibdaddev/brain-mri-lgg-unet")


def _resolve_weights() -> str:
    local = os.environ.get("WEIGHTS_PATH")
    if local:
        return local
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.pt")


MODEL = load_model(_resolve_weights(), DEVICE)


def segment(image, threshold):
    if image is None:
        return None, "Upload a slice or pick an example to start."
    overlay, area = predict_overlay(MODEL, image, DEVICE, threshold)
    pct = area * 100
    note = "no tumor detected" if area == 0 else f"{pct:.2f}% of the slice flagged as tumor"
    return overlay, note


_examples = sorted(glob.glob(os.path.join(_HERE, "examples", "*.png")))
examples = [[p, 0.5] for p in _examples] or None

demo = gr.Interface(
    fn=segment,
    inputs=[
        gr.Image(type="numpy", label="Brain MRI slice"),
        gr.Slider(0.1, 0.9, value=0.5, step=0.05, label="Decision threshold"),
    ],
    outputs=[
        gr.Image(label="Predicted contour (green)"),
        gr.Text(label="Tumor area"),
    ],
    examples=examples,
    title="Brain MRI Tumor Contouring (U-Net)",
    description=(
        "A from-scratch U-Net that outlines lower-grade glioma in brain MRI. "
        "It scored 0.914 Dice on held-out patients it never saw during training. "
        "The slider is the probability cutoff for calling a pixel 'tumor' — "
        "raise it to be stricter. Research and education only; not for clinical use."
    ),
    allow_flagging="never",
)

if __name__ == "__main__":
    demo.launch()
