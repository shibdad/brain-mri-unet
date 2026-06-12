"""Dataset, patient-level splitting, and augmentation for the LGG MRI data.

Dataset: "Brain MRI segmentation" (Buda et al.), Kaggle `mateuszbuda/lgg-mri-
segmentation`. Layout after download:

    <root>/kaggle_3m/
        TCGA_CS_4941_19960909/
            TCGA_CS_4941_19960909_1.tif        # 3-channel image (pre/FLAIR/post)
            TCGA_CS_4941_19960909_1_mask.tif   # 1-channel binary tumor mask
            ...
        TCGA_CS_4942_19970222/
            ...

Each patient is one folder; every slice has a paired `*_mask.tif`. We split by
**patient** (not by slice) so that slices from the same brain never straddle the
train/val/test boundary — otherwise the metrics are optimistically biased.
"""
from __future__ import annotations

import glob
import os
import random
from dataclasses import dataclass

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Pairing image and mask files
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    image_path: str
    mask_path: str
    patient_id: str


def find_samples(root: str) -> list[Sample]:
    """Recursively collect (image, mask, patient_id) triples under `root`.

    `root` may point either at the dataset top level or directly at `kaggle_3m`.
    """
    mask_paths = sorted(glob.glob(os.path.join(root, "**", "*_mask.tif"), recursive=True))
    if not mask_paths:
        raise FileNotFoundError(
            f"No '*_mask.tif' files found under {root!r}. "
            "Point --data-dir at the extracted dataset (it should contain a "
            "'kaggle_3m' folder)."
        )

    samples: list[Sample] = []
    for mask_path in mask_paths:
        image_path = mask_path.replace("_mask.tif", ".tif")
        if not os.path.exists(image_path):
            continue
        patient_id = os.path.basename(os.path.dirname(mask_path))
        samples.append(Sample(image_path, mask_path, patient_id))
    return samples


def split_by_patient(
    samples: list[Sample],
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    seed: int = 42,
) -> tuple[list[Sample], list[Sample], list[Sample]]:
    """Group-aware split: whole patients are assigned to a single split.

    Implemented with the standard library / numpy only (no scikit-learn) by
    shuffling the unique patient IDs and slicing them into three groups.
    """
    patients = sorted({s.patient_id for s in samples})
    rng = random.Random(seed)
    rng.shuffle(patients)

    n = len(patients)
    n_test = max(1, int(round(n * test_frac)))
    n_val = max(1, int(round(n * val_frac)))
    test_ids = set(patients[:n_test])
    val_ids = set(patients[n_test : n_test + n_val])

    train, val, test = [], [], []
    for s in samples:
        if s.patient_id in test_ids:
            test.append(s)
        elif s.patient_id in val_ids:
            val.append(s)
        else:
            train.append(s)
    return train, val, test


# ---------------------------------------------------------------------------
# Augmentation (albumentations imported lazily so the module stays importable
# even where albumentations is not installed, e.g. for unit tests)
# ---------------------------------------------------------------------------

IMAGE_SIZE = 256


def get_train_transforms(image_size: int = IMAGE_SIZE):
    import albumentations as A

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
            A.GridDistortion(p=0.2),
            A.RandomBrightnessContrast(p=0.3),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ]
    )


def get_eval_transforms(image_size: int = IMAGE_SIZE):
    import albumentations as A

    return A.Compose(
        [
            A.Resize(image_size, image_size),
            A.Normalize(mean=(0.0, 0.0, 0.0), std=(1.0, 1.0, 1.0), max_pixel_value=255.0),
        ]
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class LGGDataset(Dataset):
    """PyTorch dataset yielding (image, mask) tensors.

    image: float32 (3, H, W), normalized to [0, 1]
    mask : float32 (1, H, W), binary {0., 1.}

    `transform` is any albumentations Compose (or callable returning a dict with
    'image' and 'mask'). If None, images are simply resized and scaled to [0, 1].
    """

    def __init__(self, samples: list[Sample], transform=None, image_size: int = IMAGE_SIZE):
        self.samples = samples
        self.transform = transform
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]

        # cv2 loads BGR; convert to RGB so channel order matches the TIFF layout.
        image = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(s.mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # binarize {0,1}

        if self.transform is not None:
            out = self.transform(image=image, mask=mask)
            image, mask = out["image"], out["mask"]
        else:
            image = cv2.resize(image, (self.image_size, self.image_size))
            mask = cv2.resize(mask, (self.image_size, self.image_size))
            image = image.astype(np.float32) / 255.0

        image = torch.from_numpy(image).permute(2, 0, 1).float()  # HWC -> CHW
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # HW -> 1HW
        return image, mask


def build_datasets(root: str, image_size: int = IMAGE_SIZE, seed: int = 42):
    """Convenience: find + split + wrap into train/val/test datasets."""
    samples = find_samples(root)
    train_s, val_s, test_s = split_by_patient(samples, seed=seed)
    train_ds = LGGDataset(train_s, get_train_transforms(image_size), image_size)
    val_ds = LGGDataset(val_s, get_eval_transforms(image_size), image_size)
    test_ds = LGGDataset(test_s, get_eval_transforms(image_size), image_size)
    return train_ds, val_ds, test_ds
