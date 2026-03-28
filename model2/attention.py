import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionLayer(nn.Module):
    def __init__(self, in_features):
        super(AttentionLayer, self).__init__()
        self.W = nn.Linear(in_features, in_features)
        self.b = nn.Parameter(torch.zeros(in_features))

    def forward(self, x):
        # x shape: (batch, seq_len, features)

        score = torch.tanh(self.W(x) + self.b)
        weights = F.softmax(score, dim=1)

        context = weights * x
        context = torch.sum(context, dim=1)

        return context