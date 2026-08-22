from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import ChannelAttention


class ASAE(nn.Module):
    """Attention-based Symmetric AutoEncoder (85 → 128 → 64 → 32)."""

    def __init__(
        self,
        d_in: int = 85,
        d_h1: int = 128,
        d_h2: int = 64,
        d_z: int = 32,
        d_cls: int = 16,
        reduction: int = 8,
        dropout: float = 0.3,
        use_attention: bool = True,
        use_decoder: bool = True,
        use_residual: bool = True,
        use_dropout: bool = True,
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_decoder = use_decoder
        self.use_residual = use_residual
        p = dropout if use_dropout else 0.0

        self.enc1 = nn.Linear(d_in, d_h1)
        self.bn1 = nn.BatchNorm1d(d_h1)
        self.attn = ChannelAttention(d_h1, reduction)
        self.enc2 = nn.Linear(d_h1, d_h2)
        self.bn2 = nn.BatchNorm1d(d_h2)
        self.to_z = nn.Linear(d_h2, d_z)
        self.drop = nn.Dropout(p)

        self.dec1 = nn.Linear(d_z, d_h2)
        self.dbn1 = nn.BatchNorm1d(d_h2)
        self.dec2 = nn.Linear(d_h2, d_h1)
        self.dbn2 = nn.BatchNorm1d(d_h1)
        self.to_x = nn.Linear(d_h1, d_in)

        self.cls_h = nn.Linear(d_z, d_cls)
        self.cls = nn.Linear(d_cls, 2)

        if not use_decoder:
            for mod in (self.dec1, self.dbn1, self.dec2, self.dbn2, self.to_x):
                for p_ in mod.parameters():
                    p_.requires_grad = False

    def encode(self, x: torch.Tensor):
        h1 = self.drop(F.relu(self.bn1(self.enc1(x))))
        if self.use_attention:
            h1, attn = self.attn(h1)
        else:
            attn = torch.ones_like(h1)
        h2 = self.drop(F.relu(self.bn2(self.enc2(h1))))
        z = self.to_z(h2)
        return z, h1, h2, attn

    def decode(self, z: torch.Tensor, h1: torch.Tensor, h2: torch.Tensor):
        if not self.use_decoder:
            return None
        d1 = self.dec1(z)
        if self.use_residual:
            d1 = d1 + h2
        d1 = F.relu(self.dbn1(d1))
        d2 = self.dec2(d1)
        if self.use_residual:
            d2 = d2 + h1
        d2 = F.relu(self.dbn2(d2))
        return self.to_x(d2)

    def classify(self, z: torch.Tensor) -> torch.Tensor:
        return self.cls(F.relu(self.cls_h(z)))

    def forward(self, x: torch.Tensor, return_aux: bool = False):
        z, h1, h2, attn = self.encode(x)
        xhat = self.decode(z, h1, h2)
        logits = self.classify(z)
        if return_aux:
            return {"logits": logits, "z": z, "xhat": xhat, "attn": attn}
        return logits


def asae_from_flags(flags: dict | None = None, **kwargs) -> ASAE:
    flags = dict(flags or {})
    flags.update(kwargs)
    return ASAE(**flags)
