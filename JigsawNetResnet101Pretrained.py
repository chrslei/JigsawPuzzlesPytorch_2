# Import notwendiger Bibliotheken
import torch.nn as nn
import torch
from torchvision.models import resnet101, ResNet101_Weights
from thop import profile
from thop import clever_format

# Definition des JigsawNet Modells
class JigsawNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(JigsawNet, self).__init__()

        # Initialisiere das vortrainierte ResNet101 Modell
        self.conv = resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # Ersetze den ersten Convolutional Layer des ResNet101 Modells, um die Anzahl der Eingangskanäle anzupassen
        self.conv.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        # Erstellen zusätzlicher Fully-Connected Layer mit angepassten Dimensionen
        self.fc6 = nn.Linear(1000, 512)  # Die Größe auf 512 anpassen
        self.fc7 = nn.Linear(4608, 4096)  # Die Größe auf 4096 anpassen
        # Klassifizierungs-Layer am Ende des Netzes
        self.classifier = nn.Linear(4096, n_classes)

    # Definiere, wie Daten durch das Netz fließen (Forward-Pass)
    def forward(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])  # Convolutional Schritt mit ResNet101
            p = self.fc6(p)  # Fully Connected Schritt
            res.append(p)

        p = torch.cat(res, dim=1)  # Sammeln aller Resultate in einer Liste
        p = self.fc7(p)  # Weiterer Fully Connected Schritt
        p = self.classifier(p)  # Klassifizierungsschritt

        return p

    # Methode zum Kodieren der Eingangsdaten
    def encode(self, x):
        B, _, _, _, _ = x.size()
        res = []
        for i in range(9):
            p = self.conv(x[:, i, ...])  # Convolutional Schritt mit ResNet101
            p = p.view(B, -1)
            p = self.fc6(p)  # Fully Connected Schritt
            res.append(p)

        p = torch.cat(res, 1)  # Sammeln aller Resultate in einer Liste
        p = self.fc7(p)  # Weiterer Fully Connected Schritt
        return p

# Prüfen des Modells mit Beispieldaten
if __name__ == '__main__':
    x = torch.rand(16, 9, 1, 224, 224)  # Eingabe mit 1 Kanal und der Größe 224x224
    model = JigsawNet(in_channels=1, n_classes=100)  # Verwenden von 1 Kanal

    # Profiling des Modells zur Ermittlung von FLOPs und Parametern
    flops, params = profile(model, inputs=(x,))
    flops, params = clever_format([flops, params], "%.3f")
