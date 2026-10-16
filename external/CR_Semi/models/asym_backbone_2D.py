import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv2(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        self.norm_relu_1 = nn.Sequential(
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_relu_2 = nn.Sequential(
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_relu_1(x)

        x = self.conv_2(x)
        x = self.norm_relu_2(x)

        return x


class double_conv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=3,
            padding=2,
            dilation=2
        )
        self.norm_relu_1 = nn.Sequential(
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.norm_relu_2 = nn.Sequential(
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_relu_1(x)

        x = self.conv_2(x)
        x = self.norm_relu_2(x)

        return x


class double_conv_in(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_ch,
            out_ch,
            kernel_size=5,
            padding=2
        )
        self.norm_relu_1 = nn.Sequential(
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Conv2d(
            out_ch,
            out_ch,
            kernel_size=3,
            padding=1,
            stride=1
        )
        self.norm_relu_2 = nn.Sequential(
            nn.InstanceNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_relu_1(x)

        x = self.conv_2(x)
        x = self.norm_relu_2(x)

        x = self.pool(x)

        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = double_conv_in(in_ch, out_ch)

    def forward(self, x):
        return self.conv(x)


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.pool = nn.MaxPool2d(2)
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)

        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv = double_conv2(in_ch, out_ch)

    def forward(self, x1, x2):

        # Upsample decoder feature to exactly the skip-connection size.
        # This is safer than assuming H and W are perfectly divisible.
        x1 = F.interpolate(
            x1,
            size=x2.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        x = torch.cat(
            [x2, x1],
            dim=1
        )

        return self.conv(x)


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_ch,
            in_ch // 2,
            kernel_size=3,
            padding=1
        )

        self.norm_relu = nn.Sequential(
            nn.InstanceNorm2d(in_ch // 2),
            nn.ReLU(inplace=True)
        )

        self.conv_2 = nn.Conv2d(
            in_ch // 2,
            out_ch,
            kernel_size=1
        )

    def forward(self, x):
        x = self.conv_1(x)
        x = self.norm_relu(x)
        x = self.conv_2(x)

        return x


cc = 32


class asym_model_2D(nn.Module):
    def __init__(self, n_channels=1, n_classes=2):
        super().__init__()

        self.inconv = inconv(
            n_channels,
            cc
        )

        self.down1 = down(
            cc,
            2 * cc
        )

        self.down2 = down(
            2 * cc,
            4 * cc
        )

        self.down3 = down(
            4 * cc,
            8 * cc
        )

        self.down4 = down(
            8 * cc,
            16 * cc
        )

        self.up1 = up(
            24 * cc,
            8 * cc
        )

        self.up2 = up(
            12 * cc,
            4 * cc
        )

        self.up3 = up(
            6 * cc,
            2 * cc
        )

        self.outconv = outconv(
            2 * cc,
            n_classes
        )

    def forward(self, x):

        x0 = self.inconv(x)
        x1 = self.down1(x0)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        fine_feature = x

        coarse = self.outconv(x)

        return {
            "coarse": coarse,
            "p2_1": fine_feature,
            "p2_2": x0
        }