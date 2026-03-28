# EfficientNet-B0 backbone with Attention
import torch
import torch.nn as nn
import torchvision.models as models
from attention import AttentionLayer

class AdvancedModel(nn.Module):
    def __init__(self, num_classes=4):
        super(AdvancedModel, self).__init__()

        self.base = models.efficientnet_b0(pretrained=True)

        # Remove classifier
        self.features = self.base.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # EfficientNetB0 output features = 1280
        self.attention = AttentionLayer(1280)

        self.fc1 = nn.Linear(1280, 128)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # (batch, 1280)

        # Add sequence dimension for attention
        x = x.unsqueeze(1)

        x = self.attention(x)

        x = torch.relu(self.fc1(x))

        # apply dropout (mc dropout uses this during test)
        x = self.dropout(x)

        x = self.fc2(x)

        return x