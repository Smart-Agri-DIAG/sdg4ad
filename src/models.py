import torch
from torchvision import models


class BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.resnet18 = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.resnet18.fc = torch.nn.Linear(512, 1)

    def forward(self, x):
        x = self.resnet18(x)
        return x


if __name__ == '__main__':
    model = BinaryClassifier()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
