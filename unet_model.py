import torch
import torch.nn as nn
import torch.nn.functional as F


class BiFPN(nn.Module):
    """BiFPN for feature fusion."""
    def __init__(self, channels):
        super(BiFPN, self).__init__()
        # Top-down convolutions
        self.conv1_td = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2_td = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3_td = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4_td = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        # Bottom-up convolutions
        self.conv1_bu = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv2_bu = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv3_bu = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.conv4_bu = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

        self.relu = nn.ReLU(inplace=True)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, features):
        # Assume input features as [P2, P3, P4, P5, P6]
        P2, P3, P4, P5, P6 = features

        # Top-down pathway
        P5_td = self.relu(self.conv1_td(P5 + self.up(P6)))
        P4_td = self.relu(self.conv2_td(P4 + self.up(P5_td)))
        P3_td = self.relu(self.conv3_td(P3 + self.up(P4_td)))
        P2_td = self.relu(self.conv4_td(P2 + self.up(P3_td)))

        # Bottom-up pathway
        P3_bu = self.relu(self.conv1_bu(P3_td + self.down(P2_td)))
        P4_bu = self.relu(self.conv2_bu(P4_td + self.down(P3_bu)))
        P5_bu = self.relu(self.conv3_bu(P5_td + self.down(P4_bu)))
        P6_bu = self.relu(self.conv4_bu(P6 + self.down(P5_bu)))

        return P2_td


class UNetWithBiFPN(nn.Module):
    """UNet with BiFPN for semantic segmentation."""
    def __init__(self, in_channels=3, out_channels=1):
        super(UNetWithBiFPN, self).__init__()
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        self.bottleneck = self.conv_block(512, 1024)

        # Reduce feature map channels to 64 for BiFPN
        self.reduce_enc1 = nn.Conv2d(64, 64, kernel_size=1)
        self.reduce_enc2 = nn.Conv2d(128, 64, kernel_size=1)
        self.reduce_enc3 = nn.Conv2d(256, 64, kernel_size=1)
        self.reduce_enc4 = nn.Conv2d(512, 64, kernel_size=1)
        self.reduce_bottleneck = nn.Conv2d(1024, 64, kernel_size=1)

        # BiFPN decoder
        self.bifpn = BiFPN(channels=64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        e1 = self.enc1(x)  # Encoder block 1
        e2 = self.enc2(F.max_pool2d(e1, 2))  # Encoder block 2
        e3 = self.enc3(F.max_pool2d(e2, 2))  # Encoder block 3
        e4 = self.enc4(F.max_pool2d(e3, 2))  # Encoder block 4
        b = self.bottleneck(F.max_pool2d(e4, 2))  # Bottleneck

        # Reduce channel dimensions for BiFPN
        e1 = self.reduce_enc1(e1)  # P2
        e2 = self.reduce_enc2(e2)  # P3
        e3 = self.reduce_enc3(e3)  # P4
        e4 = self.reduce_enc4(e4)  # P5
        b = self.reduce_bottleneck(b)  # P6

        # BiFPN Fusion
        fused_features = self.bifpn([e1, e2, e3, e4, b])  # Use P2_td from BiFPN

        # Final segmentation output (only P2 used)
        return self.final(fused_features)
