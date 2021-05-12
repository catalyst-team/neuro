import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        """
        Docs.
        """
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Docs."""
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        """
        Docs.
        """
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2), DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        """Docs."""
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(
        self, in_channels, out_channels, mid_channels=None, bilinear=True
    ):
        """
        Docs.
        """
        super().__init__()

        # if bilinear,
        # use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode="trilinear", align_corners=True
            )
            if mid_channels:
                self.conv = DoubleConv(
                    in_channels, out_channels, mid_channels=mid_channels,
                )
            else:
                self.conv = DoubleConv(
                    in_channels, out_channels // 2, in_channels // 2
                )
        else:
            self.up = nn.ConvTranspose3d(
                in_channels, in_channels // 2, kernel_size=2, stride=2
            )
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        """
        Docs.
        """
        x1 = self.up(x1)
        # input is CHW
        diffZ = torch.tensor([x2.size()[2] - x1.size()[2]])
        diffY = torch.tensor([x2.size()[3] - x1.size()[3]])
        diffX = torch.tensor([x2.size()[4] - x1.size()[3]])

        x1 = F.pad(
            x1,
            [
                diffX // 2,
                diffX - diffX // 2,
                diffY // 2,
                diffY - diffY // 2,
                diffZ // 2,
                diffY - diffZ // 2,
            ],
        )
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """
    Docs.
    """

    def __init__(self, in_channels, out_channels):
        """
        Docs.
        """
        super(OutConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        """
        Docs.
        """
        return self.conv(x)


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


def conv_w_bn_before_act(dropout_p=0, *args, **kwargs):
    return nn.Sequential(
        nn.Conv3d(*args, **kwargs),
        nn.BatchNorm3d(kwargs["out_channels"]),
        nn.ReLU(inplace=True),
        nn.Dropout3d(dropout_p),
    )


def init_weights(model):
    for m in model.modules():
        if isinstance(
            m, (nn.Conv2d, nn.Conv3d, nn.ConvTranspose2d, nn.ConvTranspose3d)
        ):
            nn.init.xavier_normal_(
                m.weight, gain=nn.init.calculate_gain("relu")
            )
            nn.init.constant_(m.bias, 0.0)


class MeshNet(nn.Module):
    def __init__(self, n_channels, n_classes, large=True, dropout_p=0):
        if large:
            params = MeshNet_68_kwargs

        else:
            params = MeshNet_38_or_64_kwargs

        super(MeshNet, self).__init__()
        params[0]["in_channels"] = n_channels
        params[-1]["out_channels"] = n_classes
        layers = [
            conv_w_bn_before_act(dropout_p=dropout_p, **block_kwargs)
            for block_kwargs in params[:-1]
        ]
        layers.append(nn.Conv3d(**params[-1]))
        self.model = nn.Sequential(*layers)
        init_weights(self.model,)

    def forward(self, x):
        x = self.model(x)
        return x
