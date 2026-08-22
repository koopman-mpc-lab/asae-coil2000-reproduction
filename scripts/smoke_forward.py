from __future__ import annotations

import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.models import ASAE, MLPClassifier, SymmetricAE, TabNetLite
from src.utils import param_line


def _run(name: str, model: torch.nn.Module, x: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        out = model(x, return_aux=True)
    z = out["z"]
    xhat = out["xhat"]
    print(f"{name:12s}  {param_line(model)}")
    print(f"{'':12s}  logits {tuple(out['logits'].shape)}  z {tuple(z.shape)}"
          + (f"  xhat {tuple(xhat.shape)}" if xhat is not None else "  xhat None"))


def main():
    x = torch.randn(8, 85)
    _run("ASAE", ASAE(), x)
    _run("MLP", MLPClassifier(), x)
    _run("SAE", SymmetricAE(), x)
    _run("TabNetLite", TabNetLite(), x)
    # type-consistent representation: 83 numeric + 40/10 one-hot blocks
    _run("ASAE-133d", ASAE(d_in=133), torch.randn(8, 133))
    print("smoke_forward ok")


if __name__ == "__main__":
    main()
