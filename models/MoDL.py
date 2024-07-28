import torch
import torch.nn as nn
from util.maths import fft2c, ifft2c
from models.RSLAM import SLAM
import torch.nn.functional as F


def AtA(data, mask):  # x -> x
    # input.shape: [b, h, w], input.dtype: complex
    # output.shape: [b, h, w], output.dtype: complex
    data = fft2c(data)
    data = data * mask
    data = ifft2c(data)

    return data


class Dw(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Dw, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.attention = SLAM(64, 8, 7)

        self.conv1 = nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv4 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv5 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv6 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv7 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.conv8 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.conv9 = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = F.relu(out, inplace=True)
        out = self.conv3(out)
        out = self.conv4(out)
        out = F.relu(out, inplace=True)
        out = self.conv5(out)
        out = self.conv6(out)
        out = F.relu(out, inplace=True)
        out = self.conv7(out)
        out = self.conv8(out)
        out = F.relu(out, inplace=True)
        out = self.attention(out)
        out = self.conv9(out)

        return x + out


class ConjugatedGrad(nn.Module):
    def __init__(self):
        super(ConjugatedGrad, self).__init__()

    def forward(self, rhs, mask, lam):
        rhs = torch.view_as_complex(rhs.permute(0, 2, 3, 1).contiguous())  # rhs.shape: [b, h, w]

        x = torch.zeros_like(rhs)  # [b,h,w]
        i, r, p = 0, rhs, rhs
        rTr = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real  # rTr.shape: [b]
        num_iter, epsilon = 10, 1e-10
        for i in range(num_iter):

            Ap = AtA(p, mask) + lam * p  # [b,h,w]  # data consistency
            alpha = rTr / torch.sum(torch.conj(p) * Ap, dim=(-2, -1)).real  # [b]
            x = x + alpha[:, None, None] * p  # [b,h,w]
            r = r - alpha[:, None, None] * Ap  # [b,h,w]
            rTrNew = torch.sum(torch.conj(r) * r, dim=(-2, -1)).real  # [b]
            if rTrNew.max() < epsilon:
                break
            beta = rTrNew / rTr   # [b]
            rTr = rTrNew   # [b]
            p = r + beta[:, None, None] * p  # [b,h,w]
        return x


class MoDL(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers, device):
        super(MoDL, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers

        self.layers = Dw(self.in_channels, self.out_channels)  #
        self.lam = nn.Parameter(torch.FloatTensor([0.05]).to(device), requires_grad=True)  #

        self.CG = ConjugatedGrad()

    def forward(self, under_img, under_mask):
        x = under_img
        for i in range(self.num_layers):
            x = self.layers(x)
            x = under_img + self.lam * x
            x = self.CG(x, under_mask, self.lam)
            x = torch.view_as_real(x).permute(0, 3, 1, 2).contiguous()  # [2, 2, 320, 320]
        x_final = x
        return x_final
