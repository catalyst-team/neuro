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


MeshNet_38_or_64_kwargs = [
    {
        "in_channels": -1,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 21,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 21,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 21,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 2,
        "stride": 1,
        "dilation": 2,
    },
    {
        "in_channels": 21,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 4,
        "stride": 1,
        "dilation": 4,
    },
    {
        "in_channels": 21,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 8,
        "stride": 1,
        "dilation": 8,
    },
    {
        "in_channels": 21,
        "kernel_size": 3,
        "out_channels": 21,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 21,
        "kernel_size": 1,
        "out_channels": -1,
        "padding": 0,
        "stride": 1,
        "dilation": 1,
    },
]

MeshNet_68_kwargs = [
    {
        "in_channels": -1,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 71,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 71,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 2,
        "stride": 1,
        "dilation": 2,
    },
    {
        "in_channels": 71,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 4,
        "stride": 1,
        "dilation": 4,
    },
    {
        "in_channels": 71,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 8,
        "stride": 1,
        "dilation": 8,
    },
    {
        "in_channels": 71,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 16,
        "stride": 1,
        "dilation": 16,
    },
    {
        "in_channels": 71,
        "kernel_size": 3,
        "out_channels": 71,
        "padding": 1,
        "stride": 1,
        "dilation": 1,
    },
    {
        "in_channels": 71,
        "kernel_size": 1,
        "out_channels": -1,
        "padding": 0,
        "stride": 1,
        "dilation": 1,
    },
]


def conv_w_bn_before_act(*args, **kwargs):
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.BatchNorm3d(kwargs["out_channels"]),
        nn.ReLU(inplace=True),
    )


def init_weights(model):
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)):
            nn.init.xavier_normal_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.)


class MeshNet(nn.Module):
    def __init__(self, n_channels, n_classes, large=True):
        if large:
            params = MeshNet_68_kwargs
        else:
            params = MeshNet_38_or_64_kwargs

        super(MeshNet, self).__init__()
        params[0]["in_channels"] = n_channels
        params[-1]["out_channels"] = n_classes
        layers = [
            conv_w_bn_before_act(**block_kwargs)
            for block_kwargs in params[:-1]
        ]
        layers.append(nn.Conv3d(**params[-1]))
        self.model = nn.Sequential(*layers)
        init_weights(self.model,)

    def forward(self, x):
        x = self.model(x)
        return x
