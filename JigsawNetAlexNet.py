import torch.nn as nn
import torch
from thop import profile
from thop import clever_format

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class AlexNet(nn.Module):
    def __init__(self, in_channels):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return x

class JigsawNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(JigsawNet, self).__init__()
        self.conv = AlexNet(in_channels)
        self.fc6 = nn.Linear(256 * 6 * 6, 4096)
        self.fc7 = nn.Linear(36864, 4096)
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
    x = torch.rand(32, 9, 1, 227, 227)  # Adjust input size to 227x227
    model = JigsawNet(in_channels=1, n_classes=1000)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops, params)

    print(model(x).shape)
