import torch
import torch.nn as nn
from torchvision import models


class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.fc = nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet18(x)
        return x


if __name__ == '__main__':
    model = BinaryClassifier()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)

    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
