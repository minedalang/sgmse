#!/usr/bin/env python3
"""Targeted finite-difference check for Future-grad/SSL conflicts.

This is a read-only diagnostic.  It reuses the geometry and frozen WavLM
helpers from rrsd_pg_agreement_probe, and monkey-patches the Future-grad
method only long enough to inspect a fixed number of batches.
"""
import os
import runpy
import sys

import torch

from rrsd_pg_agreement_probe import _encoder, _feat, _future_at, _wave


def _run_probe(model, *, target, x0_front, step_out, pred, batch_num):
    score, x_t, y_main, t, sigma, wx, _, x0_base = model._score_step_geometry(step_out)
    y_cond = step_out.get("y_cond", None)
    if y_cond is None:
        y_cond = y_main
    t = t.detach().float().flatten()
    tmin = float(os.environ.get("PG_T_MIN", "0.05"))
    tmax = float(os.environ.get("PG_T_MAX", "0.50"))
    chosen = torch.nonzero((t >= tmin) & (t < tmax), as_tuple=False).flatten()
    if chosen.numel() == 0:
        print(f"[RRSD-FD-SKIP] batch={batch_num} t=" + ",".join(f"{v:.5f}" for v in t.tolist()), flush=True)
        return

    def sl(z):
        return z.detach().index_select(0, chosen)

    x_t, y_main, y_cond, sigma, wx, score = map(sl, (x_t, y_main, y_cond, sigma, wx, score))
    x0_base, x0_front, target = map(sl, (x0_base, x0_front, target))
    t = t.index_select(0, chosen)
    router_alpha = sl(pred["alpha_tile"]).float()

    if os.environ.get("PG_ANCHOR", "router") != "router":
        raise ValueError("targeted conflict FD requires PG_ANCHOR=router")
    alpha0 = router_alpha.detach().clone()
    alpha0_req = alpha0.detach().clone().requires_grad_(True)
    base_dt = float(model.rrsd_lookahead_dt)
    if base_dt <= 0:
        base_dt = (float(getattr(model.sde, "T", 1.0)) - float(model.t_eps)) / max(int(getattr(model.sde, "N", 30)) - 1, 1)
    dt = torch.minimum(torch.full_like(t, max(base_dt, 1e-6)), (t - float(model.t_eps)).clamp_min(0.0))
    if (dt <= 0).any():
        raise RuntimeError("t too near t_eps")

    wavlm = _encoder(x_t.device)
    layer = int(os.environ.get("PG_LAYER", "3"))
    maxlen = int(os.environ.get("PG_MAX_SAMPLES", "32000"))
    with torch.no_grad():
        h_clean = _feat(wavlm, _wave(model, target, maxlen), layer).detach()
    with torch.enable_grad():
        future_x0, err0 = _future_at(
            model, alpha0_req, x_t, y_main, y_cond, t, score, sigma, wx,
            x0_front - x0_base, dt, target,
        )
        future_vec = err0 / err0.detach().clamp_min(float(model.rrsd_oracle_eps))
        h_future = _feat(wavlm, _wave(model, future_x0, maxlen), layer)
        perc_vec = (h_future - h_clean).square().mean(dim=(1, 2))
        gf = torch.autograd.grad(future_vec.sum(), alpha0_req, retain_graph=True)[0].detach().float()
        gp = torch.autograd.grad(perc_vec.sum(), alpha0_req)[0].detach().float()

    rel_tol = float(os.environ.get("PG_REL_TOL", "0.10"))
    eps_f = float(os.environ.get("PG_F_MIN", "1e-10"))
    eps_p = float(os.environ.get("PG_P_MIN", "1e-10"))
    af = router_alpha.flatten(1)
    ff = gf.flatten(1)
    pp = gp.flatten(1)
    tf = torch.maximum(ff.abs().mean(1) * rel_tol, torch.full_like(t, eps_f)).unsqueeze(1)
    tp = torch.maximum(pp.abs().mean(1) * rel_tol, torch.full_like(t, eps_p)).unsqueeze(1)
    active = af >= float(os.environ.get("PG_MIN_ALPHA", "0.006"))
    conflict = (ff < -tf) & (pp > tp) & active
    # Select strongest conflict tiles only; this keeps the perturbation local.
    topk = min(int(os.environ.get("PG_FD_TOPK", "16")), ff.shape[1])
    priority = torch.where(conflict, ff.abs() / (ff.abs().mean(1, keepdim=True) + 1e-12)
                           + pp.abs() / (pp.abs().mean(1, keepdim=True) + 1e-12),
                           torch.full_like(ff, -1.0))
    mask = torch.zeros_like(ff)
    k = min(topk, int(conflict.sum(1).max().item()) if conflict.numel() else 0)
    if k > 0:
        inds = priority.topk(k, dim=1).indices
        valid = priority.gather(1, inds) >= 0
        mask.scatter_(1, inds, valid.float())
    mask = mask.view_as(alpha0)
    nsel = mask.flatten(1).sum(1)
    fd_eps = float(os.environ.get("PG_FD_EPS", "0.005"))

    with torch.no_grad():
        alpha_plus = (alpha0 + fd_eps * mask).clamp(0.0, 1.0)
        alpha_minus = (alpha0 - fd_eps * mask).clamp(0.0, 1.0)
        xp, ep = _future_at(model, alpha_plus, x_t, y_main, y_cond, t, score, sigma, wx,
                            x0_front - x0_base, dt, target)
        xm, em = _future_at(model, alpha_minus, x_t, y_main, y_cond, t, score, sigma, wx,
                            x0_front - x0_base, dt, target)
        hp = _feat(wavlm, _wave(model, xp, maxlen), layer)
        hm = _feat(wavlm, _wave(model, xm, maxlen), layer)
        lp = (hp - h_clean).square().mean(dim=(1, 2))
        lm = (hm - h_clean).square().mean(dim=(1, 2))
        base = err0.detach().clamp_min(float(model.rrsd_oracle_eps))
        dfp = ep / base - future_vec.detach()
        dfm = em / base - future_vec.detach()
        dpp = lp - perc_vec.detach()
        dpm = lm - perc_vec.detach()

    for i in range(chosen.numel()):
        plus_ok = (dfp[i].item() < 0.0) and (dpp[i].item() > 0.0)
        minus_ok = (dfm[i].item() > 0.0) and (dpm[i].item() < 0.0)
        print(
            f"[RRSD-FD] batch={batch_num} sample={int(chosen[i])} t={t[i].item():.5f} "
            f"conflict_tiles={int(nsel[i])} alpha_mean={af[i].mean().item():.8f} "
            f"fd_eps={fd_eps:.6f} plus_df={dfp[i].item():+.6e} plus_dp={dpp[i].item():+.6e} "
            f"plus_ok={int(plus_ok)} minus_df={dfm[i].item():+.6e} minus_dp={dpm[i].item():+.6e} "
            f"minus_ok={int(minus_ok)}", flush=True,
        )


def main():
    from sgmse.joint_model import JointFrontendSgmseModel

    original = JointFrontendSgmseModel._rrsd_future_grad_loss

    def wrapper(self, *, target, x0_front, step_out, pred):
        n = int(getattr(self, "_rrsd_conflict_fd_n", 0)) + 1
        limit = int(os.environ.get("PG_BATCHES", "4"))
        if n <= limit:
            if not bool(getattr(self, "rrsd_freeze_base", False)):
                raise RuntimeError("targeted FD requires --rrsd_freeze_base")
            self._rrsd_conflict_fd_n = n
            _run_probe(self, target=target, x0_front=x0_front, step_out=step_out, pred=pred, batch_num=n)
            if n >= limit:
                print("[RRSD-FD-DONE] finished without optimizer update/checkpoint save", flush=True)
                raise SystemExit(0)
        return original(self, target=target, x0_front=x0_front, step_out=step_out, pred=pred)

    JointFrontendSgmseModel._rrsd_future_grad_loss = wrapper
    sys.argv = ["joint_train.py"] + sys.argv[1:]
    runpy.run_path("joint_train.py", run_name="__main__")


if __name__ == "__main__":
    main()
