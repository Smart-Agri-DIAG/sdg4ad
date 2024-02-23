import torch
from torchvision import models


class BinaryClassifier(torch.nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.efficientnet = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.efficientnet.classifier = torch.nn.Linear(1280, 1)

    def forward(self, x):
        x = self.efficientnet(x)
        return x


if __name__ == '__main__':
    model = BinaryClassifier()
    print(model)

    total_params = sum(p.numel() for p in model.parameters())
    print("Total parameters:", total_params)
