"""Training entry point for the LGG U-Net.

Examples
--------
Real training (on Colab / a GPU box, after downloading the dataset):

    python -m src.train --data-dir data/lgg-mri-segmentation \
        --epochs 50 --batch-size 16 --lr 1e-4 --out runs/exp1

Quick CPU smoke test with synthetic tensors (no dataset needed):

    python -m src.train --smoke-test --out runs/smoke
"""
from __future__ import annotations

import argparse
import csv
import os
import time

import torch
from torch.utils.data import DataLoader, TensorDataset

from .losses import BCEDiceLoss, dice_coef, iou_score
from .model import UNet
from .utils import save_checkpoint, seed_everything


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train U-Net for LGG MRI segmentation.")
    p.add_argument("--data-dir", type=str, default=None, help="Path to extracted dataset root.")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--image-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=str, default="runs/exp", help="Output dir for checkpoints/logs.")
    p.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run 2 epochs on tiny synthetic data to verify the pipeline (no dataset).",
    )
    return p.parse_args()


def make_synthetic_loaders(batch_size: int, image_size: int):
    """Tiny random dataset used only to exercise the training loop end to end."""
    n = 8
    images = torch.rand(n, 3, image_size, image_size)
    # Targets: a centered square so the loss has a learnable signal.
    masks = torch.zeros(n, 1, image_size, image_size)
    c = image_size // 2
    masks[:, :, c - 20 : c + 20, c - 20 : c + 20] = 1.0
    ds = TensorDataset(images, masks)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True)
    return loader, loader  # reuse for "val" in the smoke test


def run_epoch(model, loader, criterion, optimizer, scaler, device, train: bool):
    model.train(train)
    total_loss = total_dice = total_iou = 0.0
    n_batches = 0

    for images, masks in loader:
        images, masks = images.to(device), masks.to(device)
        use_amp = device.type == "cuda"

        with torch.set_grad_enabled(train):
            with torch.autocast(device_type=device.type, enabled=use_amp):
                logits = model(images)
                loss = criterion(logits, masks)

            if train:
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        total_loss += loss.item()
        total_dice += dice_coef(logits.float(), masks)
        total_iou += iou_score(logits.float(), masks)
        n_batches += 1

    return total_loss / n_batches, total_dice / n_batches, total_iou / n_batches


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    os.makedirs(args.out, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    if args.smoke_test:
        train_loader, val_loader = make_synthetic_loaders(args.batch_size, args.image_size)
        epochs = 2
    else:
        if not args.data_dir:
            raise SystemExit("--data-dir is required unless --smoke-test is set.")
        # Imported here so albumentations is only required for real training.
        from .data import build_datasets

        train_ds, val_ds, _test_ds = build_datasets(args.data_dir, args.image_size, args.seed)
        print(f"Train slices: {len(train_ds)} | Val slices: {len(val_ds)}")
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
        )
        epochs = args.epochs

    model = UNet(in_channels=3, out_channels=1).to(device)
    criterion = BCEDiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    log_path = os.path.join(args.out, "metrics.csv")
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(
            ["epoch", "train_loss", "train_dice", "val_loss", "val_dice", "val_iou", "lr", "secs"]
        )

    best_dice = 0.0
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_dice, _ = run_epoch(model, train_loader, criterion, optimizer, scaler, device, True)
        va_loss, va_dice, va_iou = run_epoch(model, val_loader, criterion, optimizer, scaler, device, False)
        scheduler.step(va_dice)
        secs = time.time() - t0
        lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss {tr_loss:.4f} dice {tr_dice:.4f} | "
            f"val loss {va_loss:.4f} dice {va_dice:.4f} iou {va_iou:.4f} | "
            f"lr {lr:.2e} | {secs:.1f}s"
        )
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, f"{tr_loss:.4f}", f"{tr_dice:.4f}", f"{va_loss:.4f}",
                 f"{va_dice:.4f}", f"{va_iou:.4f}", f"{lr:.2e}", f"{secs:.1f}"]
            )

        if va_dice > best_dice:
            best_dice = va_dice
            save_checkpoint(
                model,
                os.path.join(args.out, "best_model.pt"),
                extra={"epoch": epoch, "val_dice": va_dice, "val_iou": va_iou},
            )
            print(f"  -> saved best_model.pt (val dice {va_dice:.4f})")

    save_checkpoint(model, os.path.join(args.out, "last_model.pt"), extra={"epoch": epochs})
    print(f"Done. Best val Dice: {best_dice:.4f}. Artifacts in {args.out}/")


if __name__ == "__main__":
    main()
