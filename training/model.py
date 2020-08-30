from unet_parts import DoubleConv, Down, OutConv, Up

import torch.nn as nn


class UNet(nn.Module):
    """Docs."""

    def __init__(self, n_channels, n_classes, bilinear=True):
        """Docs."""
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.up1 = Up(768, 256, bilinear=bilinear, mid_channels=256)
        self.up2 = Up(384, 128, bilinear=bilinear, mid_channels=256)
        self.up3 = Up(192, 64, bilinear=bilinear, mid_channels=64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """Docs."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        return logits
