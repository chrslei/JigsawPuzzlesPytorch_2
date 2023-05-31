import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile
from thop import clever_format
from torchvision.models import vgg16, VGG16_Weights

class JigsawNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(JigsawNet, self).__init__()

        # Load the pretrained VGG16 model
        self.conv = models.vgg16(weights = VGG16_Weights.IMAGENET1K_V1)

        # Change the first convolution layer to accept grayscale images
        self.conv.features[0] = nn.Conv2d(in_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.fc6 = nn.Linear(1000, 512)  # Adjusted based on the output of the modified VGG16 model
        self.fc7 = nn.Linear(512 * 9, 4096)  # Assuming you're still working with 9 patches
        self.classifier = nn.Linear(4096, n_classes)

    def forward(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])
            p = p.view(B, -1)
            #print(p.shape)
            p = self.fc6(p)
            res.append(p)

        p = torch.cat(res, 1)
        p = self.fc7(p)
        p = self.classifier(p)

        return p

if __name__ == '__main__':
    x = torch.rand(16, 9, 1, 64, 64)
    model = JigsawNet(in_channels=1, n_classes=100)

    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")

    print(flops, params)
    print(model(x).shape)
