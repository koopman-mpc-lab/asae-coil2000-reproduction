from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPClassifier(nn.Module):
    def __init__(self, d_in: int = 85, dims=(128, 64, 32), dropout: float = 0.3):
        super().__init__()
        layers = []
        prev = d_in
        for d in dims:
            layers.extend([nn.Linear(prev, d), nn.BatchNorm1d(d), nn.ReLU(), nn.Dropout(dropout)])
            prev = d
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2)

    def forward(self, x, return_aux: bool = False):
        h = self.backbone(x)
        logits = self.head(h)
        if return_aux:
            return {"logits": logits, "z": h, "xhat": None, "attn": None}
        return logits


class SymmetricAE(nn.Module):
    """Shared encoder/decoder used by AE, DAE, SAE, and RAE variants."""

    def __init__(
        self,
        d_in: int = 85,
        d_h1: int = 128,
        d_h2: int = 64,
        d_z: int = 32,
        dropout: float = 0.3,
        noise_std: float = 0.0,
        latent_l2: float = 0.0,
        variational: bool = False,
    ):
        super().__init__()
        self.noise_std = noise_std
        self.latent_l2 = latent_l2
        self.variational = variational
        self.enc1 = nn.Linear(d_in, d_h1)
        self.bn1 = nn.BatchNorm1d(d_h1)
        self.enc2 = nn.Linear(d_h1, d_h2)
        self.bn2 = nn.BatchNorm1d(d_h2)
        self.to_z = nn.Linear(d_h2, d_z)
        self.to_logvar = nn.Linear(d_h2, d_z) if variational else None
        self.drop = nn.Dropout(dropout)
        self.dec1 = nn.Linear(d_z, d_h2)
        self.dbn1 = nn.BatchNorm1d(d_h2)
        self.dec2 = nn.Linear(d_h2, d_h1)
        self.dbn2 = nn.BatchNorm1d(d_h1)
        self.to_x = nn.Linear(d_h1, d_in)
        self.cls_h = nn.Linear(d_z, 16)
        self.cls = nn.Linear(16, 2)

    def encode(self, x):
        if self.noise_std > 0 and self.training:
            x = x + self.noise_std * torch.randn_like(x)
        h1 = self.drop(F.relu(self.bn1(self.enc1(x))))
        h2 = self.drop(F.relu(self.bn2(self.enc2(h1))))
        mu = self.to_z(h2)
        if self.variational:
            logvar = self.to_logvar(h2)
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)
            return z, mu, logvar
        return mu, mu, None

    def forward(self, x, return_aux: bool = False):
        z, mu, logvar = self.encode(x)
        d1 = F.relu(self.dbn1(self.dec1(z)))
        d2 = F.relu(self.dbn2(self.dec2(d1)))
        xhat = self.to_x(d2)
        logits = self.cls(F.relu(self.cls_h(mu)))
        if return_aux:
            return {"logits": logits, "z": mu, "xhat": xhat, "logvar": logvar, "attn": None}
        return logits


class TabNetLite(nn.Module):
    """Small sequential-attention tabular net used for the TabNet baseline."""

    def __init__(self, d_in: int = 85, n_d: int = 24, n_steps: int = 5, gamma: float = 1.3):
        super().__init__()
        self.n_steps = n_steps
        self.gamma = gamma
        self.bn0 = nn.BatchNorm1d(d_in)
        self.feat = nn.Linear(d_in, n_d)
        self.attns = nn.ModuleList([nn.Linear(n_d, d_in) for _ in range(n_steps)])
        self.steps = nn.ModuleList([nn.Sequential(nn.Linear(d_in, n_d), nn.ReLU()) for _ in range(n_steps)])
        self.head = nn.Linear(n_d, 2)

    def forward(self, x, return_aux: bool = False):
        x0 = self.bn0(x)
        prior = torch.ones_like(x0)
        agg = 0.0
        masks = []
        h = torch.relu(self.feat(x0))
        for attn, step in zip(self.attns, self.steps):
            mask = torch.softmax(attn(h) * prior, dim=-1)
            prior = prior * (self.gamma - mask)
            out = step(x0 * mask)
            agg = agg + out
            h = out
            masks.append(mask)
        logits = self.head(agg)
        if return_aux:
            return {"logits": logits, "z": agg, "xhat": None, "attn": torch.stack(masks, dim=1)}
        return logits
