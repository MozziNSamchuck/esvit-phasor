"""
BubbleCNN 모듈 모음.

공통 입출력:
  입력: (B, T*2, H, W) — normalized magnitude+phase, 채널 순서 [mag0, phase0, mag1, ...]
  출력: (B, 1, H, W)   — bubble logit (DINO SegHead의 bubble 채널에 합산)

BubbleCNN      : (2+1)D Conv — spatial(1×3×3) → temporal(3×1×1) 순차 적용
BubbleCNN3DDS  : 3D Depthwise Separable Conv — depthwise(3×3×3, groups=C) → pointwise(1×1×1)
"""

import torch
import torch.nn as nn


class BubbleCNN(nn.Module):
    """(2+1)D CNN: spatial conv → temporal conv 순차 적용."""

    def __init__(self, in_channels=2, channels=(16, 32, 64)):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                # spatial: 각 frame 독립적으로 3×3 conv
                nn.Conv3d(c_in, c_out, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
                # temporal: 인접 3 frame에 걸친 1D conv
                nn.Conv3d(c_out, c_out, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.conv3d = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        T = C // 2
        x = x.reshape(B, T, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv3d(x)
        x = x.mean(dim=2)
        return self.head(x)


class BubbleCNN3DDS(nn.Module):
    """3D Depthwise Separable Conv: depthwise(3×3×3) → pointwise(1×1×1)."""

    def __init__(self, in_channels=2, channels=(16, 32, 64)):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in channels:
            layers += [
                # depthwise: 각 채널을 독립적으로 3×3×3 시공간 필터로 처리
                nn.Conv3d(c_in, c_in, kernel_size=3, padding=1, groups=c_in, bias=False),
                nn.BatchNorm3d(c_in),
                nn.ReLU(inplace=True),
                # pointwise: 채널 간 정보 교환
                nn.Conv3d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm3d(c_out),
                nn.ReLU(inplace=True),
            ]
            c_in = c_out
        self.conv3d = nn.Sequential(*layers)
        self.head = nn.Conv2d(channels[-1], 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        T = C // 2
        x = x.reshape(B, T, 2, H, W).permute(0, 2, 1, 3, 4).contiguous()
        x = self.conv3d(x)
        x = x.mean(dim=2)
        return self.head(x)


class _ConvBnRelu(nn.Sequential):
    def __init__(self, in_ch, out_ch, kernel_size=3, padding=1):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )


class _EncBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x):
        return self.conv(x)


class _DecBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up   = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            _ConvBnRelu(in_ch + skip_ch, out_ch),
            _ConvBnRelu(out_ch, out_ch),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class BubbleUNet2D(nn.Module):
    """
    2D UNet: 60ch 입력(20frame × mag+phase+phase_diff) → bubble binary logit map.

    Encoder:    60→16 (100×108)  →  16→32 (50×54)  →  32→64 (25×27)
    Bottleneck: 64→64 (25×27, no downsampling)
    Decoder:    64→32 (50×54, skip enc2)  →  32→16 (100×108, skip enc1)
    Output:     16→1 (100×108)
    """

    def __init__(self, in_channels=60):
        super().__init__()
        self.enc1  = _EncBlock(in_channels, 16)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2  = _EncBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3  = _EncBlock(32, 64)

        self.bottleneck = _EncBlock(64, 64)

        self.dec1 = _DecBlock(64, 32, 32)   # up enc3→enc2 해상도, skip=enc2(32ch)
        self.dec2 = _DecBlock(32, 16, 16)   # up enc2→enc1 해상도, skip=enc1(16ch)

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)               # (B, 16, 100, 108)
        e2 = self.enc2(self.pool1(e1))  # (B, 32,  50,  54)
        e3 = self.enc3(self.pool2(e2))  # (B, 64,  25,  27)

        b  = self.bottleneck(e3)        # (B, 64,  25,  27)

        d1 = self.dec1(b,  e2)          # (B, 32,  50,  54)
        d2 = self.dec2(d1, e1)          # (B, 16, 100, 108)

        return self.head(d2)            # (B,  1, 100, 108)


class _EncBlock3D(nn.Sequential):
    def __init__(self, in_ch, out_ch):
        super().__init__(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
        )


class BubbleUNet3D(nn.Module):
    """
    3D UNet: 시간 축을 명시적으로 모델링.

    입력 reshape: (B, 60, H, W) → (B, 3, 20, H, W)
                  채널=(mag, phase, phase_diff), 시간=20frame

    Encoder (3D Conv):
      enc1: (B,  3, 20, 100, 108) → (B, 16, 20, 100, 108), pool(1,2,2)→(B,16,20,50,54)
      enc2: (B, 16, 20,  50,  54) → (B, 32, 20,  50,  54), pool(2,2,2)→(B,32,10,25,27)
      enc3: (B, 32, 10,  25,  27) → (B, 64, 10,  25,  27)

    Bottleneck: (B, 64, 10, 25, 27) → temporal mean → (B, 64, 25, 27)

    Decoder (2D):
      dec1: (B, 64+32, 25, 27) → (B, 32, 50,  54)   skip=enc2 temporal mean
      dec2: (B, 32+16, 50, 54) → (B, 16, 100, 108)  skip=enc1 temporal mean

    Output: Conv2d(16→1) → (B, 1, 100, 108)
    """

    def __init__(self, n_frames=20, in_ch_per_frame=3):
        super().__init__()
        self.n_frames       = n_frames
        self.in_ch_per_frame = in_ch_per_frame

        self.enc1  = _EncBlock3D(in_ch_per_frame, 16)
        self.pool1 = nn.MaxPool3d((1, 2, 2))
        self.enc2  = _EncBlock3D(16, 32)
        self.pool2 = nn.MaxPool3d((2, 2, 2))
        self.enc3  = _EncBlock3D(32, 64)

        self.bottleneck = _EncBlock3D(64, 64)

        self.dec1 = _DecBlock(64, 32, 32)
        self.dec2 = _DecBlock(32, 16, 16)

        self.head = nn.Conv2d(16, 1, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        # (B, 60, H, W) → (B, 3, 20, H, W)
        x = x.reshape(B, self.n_frames, self.in_ch_per_frame, H, W) \
             .permute(0, 2, 1, 3, 4).contiguous()

        e1 = self.enc1(x)                   # (B, 16, 20, 100, 108)
        e2 = self.enc2(self.pool1(e1))      # (B, 32, 20,  50,  54)
        e3 = self.enc3(self.pool2(e2))      # (B, 64, 10,  25,  27)

        b  = self.bottleneck(e3).mean(dim=2)  # (B, 64, 25, 27) — temporal collapse

        # skip: enc2, enc1 temporal mean → 2D
        s2 = e2.mean(dim=2)                 # (B, 32, 50, 54)
        s1 = e1.mean(dim=2)                 # (B, 16, 100, 108)

        d1 = self.dec1(b,  s2)              # (B, 32, 50,  54)
        d2 = self.dec2(d1, s1)              # (B, 16, 100, 108)

        return self.head(d2)                # (B,  1, 100, 108)


def build_bubble_cnn(cnn_type='2plus1d', in_channels=2, channels=(16, 32, 64)):
    if cnn_type == 'unet3d':
        return BubbleUNet3D(n_frames=20, in_ch_per_frame=3)
    if cnn_type == 'unet2d':
        return BubbleUNet2D(in_channels=in_channels)
    if cnn_type == '3d_dws':
        return BubbleCNN3DDS(in_channels=in_channels, channels=channels)
    return BubbleCNN(in_channels=in_channels, channels=channels)
