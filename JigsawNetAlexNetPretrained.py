import torch
import torch.nn as nn
from torchvision import models
from thop import profile
from thop import clever_format


class JigsawNet(nn.Module):
    def __init__(self, n_classes):
        super(JigsawNet, self).__init__()
        alexnet = models.alexnet(pretrained=True)
        alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2)
        self.conv = nn.Sequential(
            alexnet.features,
            nn.AdaptiveAvgPool2d((6, 6)),
        )

        # get the number of features just before the classifier
        num_ftrs = 256 * 6 * 6
        self.fc6 = nn.Linear(num_ftrs, 4096)

        self.fc7 = nn.Linear(4096 * 9, 4096)  # We have 9 patches
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            p = self.fc6(p)
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        p = self.classifier(p)

        return p

    def encode(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            p = self.fc6(p)
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        return p


if __name__ == '__main__':
    x = torch.rand(16, 9, 1, 227, 227)  # Adjust input size to 227x227
    model = JigsawNet(n_classes=100)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops, params)

    print(model(x).shape)
