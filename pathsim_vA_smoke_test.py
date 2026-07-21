#!/usr/bin/env python3
import math
import tempfile
from pathlib import Path

import torch

from sgmse.pathsim_vA import (
    BandTimeVariancePathEstimator,
    load_pathsim_va_checkpoint,
    variance_nll_loss,
)


def main():
    torch.manual_seed(0)
    B, C, F, T = 2, 1, 64, 40
    y_pre = torch.randn(B, C, F, T, dtype=torch.complex64)
    y_noisy = y_pre + 0.3 * torch.randn_like(y_pre)
    t = torch.tensor([0.3, 0.7])
    model = BandTimeVariancePathEstimator(
        hidden_ch=16, num_bands=8, num_time_blocks=5, min_rho=0.5, max_rho=2.0
    )
    out = model(y_pre, y_noisy, t, return_full=True)
    assert out["log_var_tile"].shape == (B, 1, 8, 5)
    assert out["log_var_full"].shape == (B, 1, F, T)
    assert torch.allclose(out["rho_tile"], torch.ones_like(out["rho_tile"]), atol=1e-6)

    # The NLL derivative vanishes when exp(log_var) equals target |z|^2.
    target_energy = 1.7
    z = torch.full((B, C, F, T), math.sqrt(target_energy), dtype=torch.complex64)
    log_var = torch.full((B, 1, 8, 5), math.log(target_energy), requires_grad=True)
    loss, _ = variance_nll_loss(log_var, z)
    loss.backward()
    assert log_var.grad.abs().max().item() < 1e-5

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "pathsim.pt"
        torch.save({"state_dict": model.state_dict(), "config": model.get_config()}, path)
        loaded, _ = load_pathsim_va_checkpoint(str(path))
        out_loaded = loaded(y_pre, y_noisy, t)["log_var_tile"]
        assert torch.allclose(out_loaded, out["log_var_tile"], atol=1e-6)

    print("PathSim V-A smoke test passed")


if __name__ == "__main__":
    main()
