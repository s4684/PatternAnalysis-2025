import torch
import torch.nn as nn
import torch.nn.functional as F

class InConv(nn.Module):
    def __init__(self, in_chls, out_chls):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_chls, out_chls, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chls),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_chls, out_chls, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_chls),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class DownLayer(nn.Module):
    def __init__(self, in_chls, out_chls):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv = InConv(in_chls, out_chls)

    def forward(self, x):
        return self.conv(self.pool(x))


class UpLayer(nn.Module):
    def __init__(self, in_chls, out_chls):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_chls, in_chls // 2, kernel_size=2, stride=2)
        self.conv = InConv(in_chls, out_chls)

    def forward(self, x, skip):
        x = self.up(x)

        dx = skip.size(3) - x.size(3)
        dy = skip.size(2) - x.size(2)

        if (dx or dy):
            x = F.pad(x, [dx//2, dx - dx//2, dy//2, dy-dy//2])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_chls, out_chls):
        super().__init__()
        self.conv = nn.Conv2d(in_chls, out_chls, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):

    def __init__(self, in_chls=1, num_classes=4, base_chls=64):
        super().__init__()

        c = [base_chls * 2**i for i in range(5)]

        self.input = InConv(in_chls, base_chls)
        self.down1 = DownLayer(c[0], c[1])
        self.down2 = DownLayer(c[1], c[2])
        self.down3 = DownLayer(c[2], c[3])
        self.down4 = DownLayer(c[3], c[4])

        self.up1 = UpLayer(c[4], c[3])
        self.up2 = UpLayer(c[3], c[2])
        self.up3 = UpLayer(c[2], c[1])
        self.up4 = UpLayer(c[1], c[0])
        self.output = OutConv(c[0], num_classes)

    def forward(self, x):
        x1 = self.input(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        return self.output(
            self.up4(
                self.up3(
                    self.up2(
                        self.up1(x5, x4), 
                        x3), 
                    x2), 
                x1)
            )


class MCDiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, logits, target):
        # Convert raw output to probabilities
        probs = logits.softmax(dim=1)
        B, C, H, W = probs.shape

        target_oh = F.one_hot(target, num_classes=C).permute(0, 3, 1, 2).float()

        probs_flat = probs.reshape(B, C, -1)
        target_flat = target_oh.reshape(B, C, -1)

        intersect = (probs_flat * target_flat).sum(dim=-1)
        total = probs_flat.sum(dim=-1) + target_flat.sum(dim=-1)

        dice = (2 * intersect + self.eps) / (total + self.eps)

        return 1 - dice.mean()

