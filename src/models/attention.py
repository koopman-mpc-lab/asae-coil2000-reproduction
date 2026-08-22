from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation style gate over a fully connected hidden vector."""

    def __init__(self, m: int = 128, reduction: int = 8):
        super().__init__()
        hidden = max(1, m // reduction)
        self.fc1 = nn.Linear(m, hidden)
        self.fc2 = nn.Linear(hidden, m)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        a = torch.sigmoid(self.fc2(F.relu(self.fc1(h))))
        return h * a, a
