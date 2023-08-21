import torch.nn as nn
import torch


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.double_conv(x)
        x_pooled = self.maxpool(x)
        return x, x_pooled


class DecoderBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, up_mode="bilinear", align_corners=False
    ):
        super().__init__()
        assert up_mode in ("nearest", "bilinear")
        self.upsample = nn.Upsample(
            scale_factor=2, mode=up_mode, align_corners=align_corners
        )
        self.double_conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.upsample(x)
        x = torch.cat([x, x_skip], dim=1)
        x = self.double_conv(x)
        return x


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder1 = EncoderBlock(1, 64)
        self.encoder2 = EncoderBlock(64, 128)
        self.encoder3 = EncoderBlock(128, 256)
        self.encoder4 = EncoderBlock(256, 512)

        self.decoder1 = DecoderBlock(512 + 256, 256)
        self.decoder2 = DecoderBlock(256 + 128, 128)
        self.decoder3 = DecoderBlock(128 + 64, 64)

        self.final_conv = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        x1, x1_pooled = self.encoder1(x)
        x2, x2_pooled = self.encoder2(x1_pooled)
        x3, x3_pooled = self.encoder3(x2_pooled)
        x4, _ = self.encoder4(x3_pooled)

        x = self.decoder1(x4, x3)
        x = self.decoder2(x, x2)
        x = self.decoder3(x, x1)

        output = self.final_conv(x)
        return output

    def get_name(self):
        return "unet"
