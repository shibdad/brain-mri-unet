"""Small shared helpers: reproducibility, checkpointing, and overlay rendering."""
from __future__ import annotations

import os
import random

import cv2
import numpy as np
import torch


def seed_everything(seed: int = 42) -> None:
    """Seed Python, NumPy and PyTorch RNGs for reproducible runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_checkpoint(model: torch.nn.Module, path: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"model_state": model.state_dict()}
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(model: torch.nn.Module, path: str, map_location: str = "cpu") -> dict:
    """Load weights into `model` in place; return the full checkpoint dict."""
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model_state"] if "model_state" in ckpt else ckpt
    model.load_state_dict(state)
    return ckpt


def overlay_mask(
    image: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    alpha: float = 0.25,
) -> np.ndarray:
    """Draw the mask boundary (contour) + a translucent fill over `image`.

    Args:
        image: HxWx3 uint8 RGB.
        mask:  HxW array; nonzero = tumor.
        color: contour/fill color in RGB.
    Returns:
        HxWx3 uint8 RGB with the tumor contour drawn on top.
    """
    image = np.ascontiguousarray(image).copy()
    binary = (mask > 0).astype(np.uint8)

    # Translucent fill so the region is visible without hiding the anatomy.
    if binary.any():
        fill = image.copy()
        fill[binary == 1] = color
        image = cv2.addWeighted(fill, alpha, image, 1 - alpha, 0)

    # Crisp boundary contour on top.
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, contours, -1, color, thickness)
    return image
