import torch
from torch import nn

CLASSES = ["invoice", "letter", "form", "report"]

class DocumentCNN(nn.Module):
    def __init__(self, num_classes=len(CLASSES)):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(), nn.Dropout(0.2), nn.Linear(64, num_classes))
    def forward(self, x):
        return self.network(x)
