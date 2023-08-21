import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, p=0.0, bt=True):
        super().__init__()
        self.dropout_p = p
        if not mid_channels:
            mid_channels = out_channels
        if bt == True:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )
        # self.dropout = nn.Dropout(p)

    def forward(self, x):
        x = self.double_conv(x)
        return x


class EncoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=0, bt=True):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, dropout, bt)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.double_conv(x)
        x_pooled = self.maxpool(x)
        return x, x_pooled


class AttentionGate(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=1, padding=0), nn.BatchNorm2d(out_c)
        )
        self.Ws = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=1, padding=0), nn.BatchNorm2d(out_c)
        )
        self.relu = nn.ReLU(inplace=True)
        self.output = nn.Sequential(
            nn.Conv2d(out_c, out_c, kernel_size=1, padding=0), nn.Sigmoid()
        )

    def forward(self, g, s):
        Wg = self.Wg(g)
        Ws = self.Ws(s)
        out = self.relu(Wg + Ws)
        out = self.output(out)
        return out * s


class DecoderBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        up_mode="bilinear",
        align_corners=False,
        dropout=0,
        bt=True,
    ):
        super().__init__()
        assert up_mode in ("nearest", "bilinear")
        self.up = nn.Upsample(scale_factor=2, mode=up_mode, align_corners=align_corners)
        self.ag = AttentionGate(in_channels, out_channels)
        self.c1 = DoubleConv(in_channels[0] + out_channels, out_channels, dropout, bt)

    def forward(self, x, s):
        x = self.up(x)
        s = self.ag(x, s)
        x = torch.cat([x, s], axis=1)  # type: ignore
        x = self.c1(x)
        return x


class A_UNET(nn.Module):
    def __init__(self, dropout=0, bt=True):
        super().__init__()

        self.e1 = EncoderBlock(1, 64, dropout, bt)
        self.e2 = EncoderBlock(64, 128, dropout, bt)
        self.e3 = EncoderBlock(128, 256, dropout, bt)

        self.b1 = DoubleConv(256, 512)

        self.d1 = DecoderBlock([512, 256], 256, dropout=dropout, bt=bt)
        self.d2 = DecoderBlock([256, 128], 128, dropout=dropout, bt=bt)
        self.d3 = DecoderBlock([128, 64], 64, dropout=dropout, bt=bt)

        self.output = nn.Conv2d(64, 1, kernel_size=1)

    def forward(self, x):
        s1, p1 = self.e1(x)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)

        b1 = self.b1(p3)

        d1 = self.d1(b1, s3)
        d2 = self.d2(d1, s2)
        d3 = self.d3(d2, s1)

        output = self.output(d3)
        return output

    def get_name(self):
        return "attention_unet"
