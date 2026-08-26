from __future__ import annotations

import torch
import torch.nn.functional as F


def weighted_ce(logits: torch.Tensor, y: torch.Tensor, w0: float, w1: float) -> torch.Tensor:
    """Class-weighted cross entropy, normalised by the original batch size.

    PyTorch's weighted ``mean`` divides by the sum of selected weights.  The
    objective used here is instead sum_i w[y_i] CE_i / N.
    """
    weight = torch.tensor([w0, w1], device=logits.device, dtype=logits.dtype)
    return F.cross_entropy(logits, y.long(), weight=weight, reduction="sum") / logits.shape[0]


def recon_mse(x: torch.Tensor, xhat: torch.Tensor | None) -> torch.Tensor:
    """Per-record squared-error sum over features, followed by batch mean."""
    if xhat is None:
        return x.new_zeros(())
    return F.mse_loss(xhat, x, reduction="none").sum(dim=1).mean()


def recon_mixed(
    x: torch.Tensor,
    xhat: torch.Tensor | None,
    numeric_dim: int,
    block_sizes: list[int],
    numeric_scale: float = 1.0,
    categorical_scale: float = 1.0,
) -> torch.Tensor:
    """Mixed reconstruction with an explicit, per-record common scale.

    Numeric features contribute their summed squared errors.  Every nominal
    one-hot block contributes one categorical cross entropy.  Both components
    are first formed per record, scaled explicitly, summed, and then averaged
    over the batch.  Thus ``alpha`` in :func:`total_loss` has the same
    interpretation for numeric and mixed representations.
    """
    if xhat is None:
        return x.new_zeros(())
    if numeric_dim < 0 or numeric_dim + sum(block_sizes) != x.shape[1]:
        raise ValueError("mixed reconstruction layout does not match input width")
    per_record = numeric_scale * F.mse_loss(
        xhat[:, :numeric_dim], x[:, :numeric_dim], reduction="none"
    ).sum(dim=1)
    ofs = numeric_dim
    for size in block_sizes:
        target = x[:, ofs : ofs + size].argmax(dim=1)
        per_record = per_record + categorical_scale * F.cross_entropy(
            xhat[:, ofs : ofs + size], target, reduction="none"
        )
        ofs += size
    return per_record.mean()


def entropy_penalty(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    var = z.var(dim=0, unbiased=False)
    return -(var + eps).log().mean()


def total_loss(
    logits,
    y,
    x,
    xhat,
    z,
    w0,
    w1,
    alpha=0.5,
    beta=1e-2,
    use_ent=True,
    mixed_layout=None,
    mixed_numeric_scale=1.0,
    mixed_categorical_scale=1.0,
):
    l_cls = weighted_ce(logits, y, w0, w1)
    if mixed_layout is not None:
        l_rec = recon_mixed(
            x,
            xhat,
            mixed_layout[0],
            mixed_layout[1],
            numeric_scale=mixed_numeric_scale,
            categorical_scale=mixed_categorical_scale,
        )
    else:
        l_rec = recon_mse(x, xhat)
    l_ent = entropy_penalty(z) if use_ent else logits.new_zeros(())
    return l_cls + alpha * l_rec + beta * l_ent, {
        "cls": float(l_cls.detach()),
        "recon": float(l_rec.detach()),
        "ent": float(l_ent.detach()),
    }
