from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, w0: float, w1: float) -> torch.Tensor:
    weight = torch.tensor([w0, w1], device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, y.long(), weight=weight)


def recon_mse(x: torch.Tensor, xhat: torch.Tensor | None) -> torch.Tensor:
    if xhat is None:
        return x.new_zeros(())
    return F.mse_loss(xhat, x)


def recon_mixed(
    x: torch.Tensor,
    xhat: torch.Tensor | None,
    numeric_dim: int,
    block_sizes: list[int],
) -> torch.Tensor:
    """MSE on the numeric slice, softmax cross-entropy per one-hot block."""
    if xhat is None:
        return x.new_zeros(())
    loss = F.mse_loss(xhat[:, :numeric_dim], x[:, :numeric_dim])
    ofs = numeric_dim
    for size in block_sizes:
        target = x[:, ofs : ofs + size].argmax(dim=1)
        loss = loss + F.cross_entropy(xhat[:, ofs : ofs + size], target) / len(block_sizes)
        ofs += size
    return loss


def entropy_penalty(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    var = z.var(dim=0, unbiased=False)
    return -(var + eps).log().mean()


def total_loss(logits, y, x, xhat, z, w0, w1, alpha=0.5, beta=1e-2, use_ent=True, mixed_layout=None):
    l_cls = weighted_ce(logits, y, w0, w1)
    if mixed_layout is not None:
        l_rec = recon_mixed(x, xhat, mixed_layout[0], mixed_layout[1])
    else:
        l_rec = recon_mse(x, xhat)
    l_ent = entropy_penalty(z) if use_ent else logits.new_zeros(())
    return l_cls + alpha * l_rec + beta * l_ent, {
        "cls": float(l_cls.detach()),
        "recon": float(l_rec.detach()),
        "ent": float(l_ent.detach()),
    }
