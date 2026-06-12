"""Inference utilities: load a trained U-Net and produce tumor contours.

These functions are written to be imported directly by the future Gradio app
(`app/app.py`) as well as run from the command line:

    python -m src.inference --weights runs/exp1/best_model.pt \
        --image sample.tif --out prediction.png
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np
import torch

from .model import UNet
from .utils import load_checkpoint, overlay_mask

IMAGE_SIZE = 256


def load_model(weights_path: str, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Instantiate U-Net and load trained weights; returns an eval-mode model."""
    device = torch.device(device)
    model = UNet(in_channels=3, out_channels=1)
    load_checkpoint(model, weights_path, map_location=str(device))
    model.to(device).eval()
    return model


def _read_rgb(image_path: str) -> np.ndarray:
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path!r}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


@torch.no_grad()
def predict_mask(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> np.ndarray:
    """Predict a binary tumor mask for an RGB image (any size).

    Returns a uint8 mask {0,1} at the image's ORIGINAL resolution.
    """
    device = torch.device(device)
    orig_h, orig_w = image.shape[:2]

    resized = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32) / 255.0
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).to(device)

    logits = model(tensor)
    prob = torch.sigmoid(logits)[0, 0].cpu().numpy()
    prob = cv2.resize(prob, (orig_w, orig_h))
    return (prob > threshold).astype(np.uint8)


def predict_overlay(
    model: torch.nn.Module,
    image: np.ndarray,
    device: str | torch.device = "cpu",
    threshold: float = 0.5,
) -> tuple[np.ndarray, float]:
    """Return (overlay_rgb_uint8, tumor_area_fraction)."""
    mask = predict_mask(model, image, device, threshold)
    overlay = overlay_mask(image, mask)
    area_fraction = float(mask.mean())  # fraction of pixels flagged as tumor
    return overlay, area_fraction


def main() -> None:
    p = argparse.ArgumentParser(description="Run U-Net inference on one MRI slice.")
    p.add_argument("--weights", required=True, help="Path to trained .pt checkpoint.")
    p.add_argument("--image", required=True, help="Path to an MRI slice (.tif/.png/.jpg).")
    p.add_argument("--out", default="prediction.png", help="Where to save the overlay PNG.")
    p.add_argument("--threshold", type=float, default=0.5)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.weights, device)
    image = _read_rgb(args.image)
    overlay, area = predict_overlay(model, image, device, args.threshold)

    cv2.imwrite(args.out, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"Saved overlay -> {args.out} | tumor area fraction: {area:.4f}")


if __name__ == "__main__":
    main()
