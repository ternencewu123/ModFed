import torch
import torch.nn as nn


class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels, ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.body(x)
        return out


class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels, ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

    def forward(self, x):
        out = self.body(x)
        return out


class LaplacianAttention(nn.Module):
    def __init__(self, channel, reduction=16):
        super(LaplacianAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        # 3, 5, 7->1, 3, 5
        self.c1 = BasicBlock(channel, channel // reduction, 3, 1, 3, 3)  # 3, 1, 3, 3
        self.c2 = BasicBlock(channel, channel // reduction, 3, 1, 5, 5)  # 3, 1, 5, 5
        self.c3 = BasicBlock(channel, channel // reduction, 3, 1, 7, 7)  # 3, 1, 7, 7
        self.c4 = BasicBlockSig((channel // reduction)*3, channel, 3, 1, 1)  # 3, 1, 1

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_pool = self.avg_pool(x)
        avg_c1 = self.c1(avg_pool)
        avg_c2 = self.c2(avg_pool)
        avg_c3 = self.c3(avg_pool)
        avg_pool = torch.cat([avg_c1, avg_c2, avg_c3], dim=1)
        avg_pool = self.c4(avg_pool)

        max_pool = self.max_pool(x)
        max_c1 = self.c1(max_pool)
        max_c2 = self.c2(max_pool)
        max_c3 = self.c3(max_pool)
        max_pool = torch.cat([max_c1, max_c2, max_c3], dim=1)
        max_pool = self.c4(max_pool)

        out = avg_pool + max_pool
        laplacian_feature = self.sigmoid(out)

        return x*laplacian_feature


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)
        spatial_feature = self.sigmoid(out)

        return x*spatial_feature


class SLAM(nn.Module):
    def __init__(self, channel, reduction, kernel_size):
        super(SLAM, self).__init__()

        self.channel = channel
        self.reducation = reduction
        self.kernel_size = kernel_size

        self.laplacian = LaplacianAttention(self.channel, self.reducation)
        self.spatial = SpatialAttention(self.kernel_size)

    def forward(self, x):
        l_feature = self.laplacian(x)
        res = self.spatial(l_feature)
        return res
