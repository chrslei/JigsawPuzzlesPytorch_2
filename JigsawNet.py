import torch.nn as nn
import torch
from torchvision.models import resnet101
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

class JigsawNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(JigsawNet, self).__init__()

        self.conv = resnet101(pretrained=False, num_classes=1000)
        self.conv.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Modify fc6 to match the output features of the ResNet101 model
        self.fc6 = nn.Linear(9000, 512)  # Adjust the size to 512
        self.fc7 = nn.Linear(512, 4096)
        self.classifier = nn.Linear(4096, n_classes)

    # Rest of the code...

    def forward(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            res.append(p)

        p = torch.cat(res, dim=1)
        p = p.view(B, -1)  # Reshape the tensor to flatten the last two dimensions

        p = self.fc6(p)
        p = self.fc7(p)

        return p

    def encode(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc6(p)
        return p

if __name__ == '__main__':
    x = torch.rand(16, 9, 1, 64, 64)  # Input with 1 channel
    model = JigsawNet(in_channels=1, n_classes=1000)  # Use 1 channel

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print("ResNet Features Shape:", model.fc6.in_features)
    print("Total Parameters:", params)

    print(model(x).shape)
