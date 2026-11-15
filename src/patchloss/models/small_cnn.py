"""Low-cost image-space learning baseline, not Bayes oracle."""

from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 12, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(12, 24, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(24 * 4 * 4, 48), nn.ReLU(), nn.Linear(48, classes))

    def forward(self, x):
        return self.head(self.features(x))
