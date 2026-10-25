"""俯瞰画像の物体セグメンテーション用の小型 U-Net（約 0.5M パラメータ、CPU で学習可能）。"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SegNet(nn.Module):
    def __init__(self, num_classes: int, width: int = 16):
        super().__init__()
        w = [width, width * 2, width * 4, width * 8]
        self.enc = nn.ModuleList([_block(3, w[0]), _block(w[0], w[1]), _block(w[1], w[2]),
                                  _block(w[2], w[3])])
        self.dec = nn.ModuleList([_block(w[3] + w[2], w[2]), _block(w[2] + w[1], w[1]),
                                  _block(w[1] + w[0], w[0])])
        self.head = nn.Conv2d(w[0], num_classes, 1)

    def forward(self, x):
        skips = []
        for i, enc in enumerate(self.enc):
            x = enc(x)
            if i < len(self.enc) - 1:
                skips.append(x)
                x = F.max_pool2d(x, 2)
        for dec in self.dec:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
            x = dec(torch.cat([x, skips.pop()], 1))
        return self.head(x)


MEAN = torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1)
STD = torch.tensor([0.25, 0.25, 0.25]).view(1, 3, 1, 1)


def preprocess(rgb_uint8: torch.Tensor) -> torch.Tensor:
    """(B,H,W,3) uint8 -> (B,3,H,W) float 正規化済み。"""
    x = rgb_uint8.permute(0, 3, 1, 2).float() / 255.0
    return (x - MEAN) / STD
