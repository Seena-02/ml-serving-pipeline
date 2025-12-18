# Imports
import torch.nn as nn
import torch.nn.functional as F


# Create a simply CNN with pytorch.
class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = F.relu(x)

        # Conv Block 2
        x = self.conv2(x)
        x = F.relu(x)

        # Downsample
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # (batch_size, 64 * 14 * 14)

        x = self.fc1(x)
        x = F.relu(x)

        x = self.fc2(x)

        return x
