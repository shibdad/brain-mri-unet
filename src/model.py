"""U-Net architecture for 2D brain-MRI tumor segmentation.

A from-scratch implementation of the classic U-Net (Ronneberger et al., 2015)
with batch normalization. Kept dependency-free (pure PyTorch) so the repository
demonstrates the architecture end to end rather than importing a prebuilt model.

Input : (N, in_channels=3, H, W)  -- the 3 MRI channels (pre / FLAIR / post)
Output: (N, out_channels=1, H, W) -- raw logits (apply sigmoid for probabilities)
"""
from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv -> BatchNorm -> ReLU) x 2, the basic U-Net building block."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet(nn.Module):
    """U-Net with a configurable channel pyramid.

    Args:
        in_channels: number of input image channels (3 for this dataset).
        out_channels: number of output mask channels (1 for binary tumor).
        features: channel widths of the encoder stages. The decoder mirrors them.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 1,
        features: tuple[int, ...] = (64, 128, 256, 512),
    ) -> None:
        super().__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder: progressively widen channels, halving spatial size via pool.
        prev = in_channels
        for feat in features:
            self.downs.append(DoubleConv(prev, feat))
            prev = feat

        # Bottleneck at the bottom of the "U".
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Decoder: transpose-conv upsample, concat skip connection, then DoubleConv.
        for feat in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(feat * 2, feat, kernel_size=2, stride=2)
            )
            self.ups.append(DoubleConv(feat * 2, feat))

        # 1x1 conv maps to the output channel count (logits).
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]  # reverse for the decoder

        # self.ups holds [upconv, doubleconv, upconv, doubleconv, ...]
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)  # transpose conv
            skip = skip_connections[idx // 2]

            # Guard against odd input sizes where pooling rounds down: if the
            # upsampled tensor and the skip differ by a pixel, align them.
            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x, size=skip.shape[-2:], mode="bilinear", align_corners=False
                )

            x = torch.cat((skip, x), dim=1)
            x = self.ups[idx + 1](x)  # double conv

        return self.final_conv(x)


if __name__ == "__main__":
    # Quick shape sanity check.
    model = UNet(in_channels=3, out_channels=1)
    dummy = torch.randn(2, 3, 256, 256)
    out = model(dummy)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"output shape: {tuple(out.shape)}  |  parameters: {n_params:,}")
