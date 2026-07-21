#!/usr/bin/env python3
"""Train the independent Path Simulation Version-A variance estimator.

The baseline frontend and score model are frozen.  For every batch, the script
runs the baseline reverse sampler to one randomly selected intermediate step,
constructs

    z_rollout = (x_t_rollout - marginal_mean(clean, y_main, t)) / std(t),

and fits a frontend-guided band-time zero-mean Gaussian variance model using
heteroscedastic NLL.  Only the small path estimator is optimized and saved.
"""

from __future__ import annotations

import argparse
import math
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from sgmse.backbones.shared import BackboneRegistry
from sgmse.frontends.storm import StormFrontend
from sgmse.joint_model import JointFrontendSgmseModel
from sgmse.pathsim_vA import (
    BandTimeVariancePathEstimator,
    standard_normal_kl,
    standardized_rollout_residual,
    total_variation,
    variance_nll_loss,
)
from sgmse.sampling import CorrectorRegistry, PredictorRegistry
from sgmse.sdes import SDERegistry
from sgmse.util.other import set_torch_cuda_arch_list

set_torch_cuda_arch_list()
torch.set_float32_matmul_precision("high")


def torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def argparse_groups(parser: argparse.ArgumentParser, args: argparse.Namespace):
    result = {}
    for group in parser._action_groups:
        result[group.title] = argparse.Namespace(
            **{a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        )
    return result


def load_baseline_weights(model: JointFrontendSgmseModel, ckpt_path: str, use_ema: bool) -> None:
    ckpt = torch_load(ckpt_path)
    state = ckpt.get("state_dict", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[baseline] state loaded: missing={len(missing)} unexpected={len(unexpected)}")
    model._ensure_ema_is_compatible(force_rebuild=True)
    ema_state = ckpt.get("ema") if isinstance(ckpt, dict) else None
    if use_ema and ema_state is not None:
        model.ema.load_state_dict(ema_state)
        model.ema.copy_to(list(model._get_ema_parameters()))
        print("[baseline] EMA weights applied")
    elif use_ema:
        print("[baseline] warning: checkpoint has no EMA; using raw weights")


def frontend_specs(model: JointFrontendSgmseModel, clean_wav: torch.Tensor, noisy_wav: torch.Tensor):
    X = model._sgmse_stft(clean_wav)
    Y_noisy = model._sgmse_stft(noisy_wav)
    if model.frontend_name == "storm" and hasattr(model.frontend, "forward_spec"):
        Y_pre = model.frontend.forward_spec(noisy_wav)
    else:
        y_pre = model._frontend_forward(noisy_wav)
        if y_pre.shape[-1] != noisy_wav.shape[-1]:
            if y_pre.shape[-1] > noisy_wav.shape[-1]:
                y_pre = y_pre[..., : noisy_wav.shape[-1]]
            else:
                y_pre = F.pad(y_pre, (0, noisy_wav.shape[-1] - y_pre.shape[-1]))
        Y_pre = model._sgmse_stft(y_pre)
    return X, Y_noisy, Y_pre


@torch.no_grad()
def reverse_state_at_random_step(
    model: JointFrontendSgmseModel,
    y_main: torch.Tensor,
    y_cond: torch.Tensor,
    N: int,
    progress_min: float,
    progress_max: float,
    predictor_name: str,
    corrector_name: str,
    corrector_steps: int,
    snr: float,
):
    """Return the actual state seen by the score model at a random reverse step.

    progress=0 corresponds to the first/high-noise reverse step and progress=1
    to the last/low-noise step.
    """
    if N < 2:
        raise ValueError("N must be >=2")
    lo = max(0, min(N - 1, int(round(progress_min * (N - 1)))))
    hi = max(lo, min(N - 1, int(round(progress_max * (N - 1)))))
    target_idx = random.randint(lo, hi)

    score_fn = partial(model, y_cond=y_cond)
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)
    predictor = predictor_cls(model.sde, score_fn, probability_flow=False)
    corrector = corrector_cls(model.sde, score_fn, snr=snr, n_steps=corrector_steps)

    x_t = model.sde.prior_sampling(y_main.shape, y_main).to(y_main.device)
    timesteps = torch.linspace(float(model.sde.T), float(model.t_eps), N, device=y_main.device)
    for i in range(target_idx + 1):
        t_scalar = timesteps[i]
        vec_t = torch.full((y_main.shape[0],), float(t_scalar.item()), device=y_main.device)
        x_t, _ = corrector.update_fn(x_t, y_main, vec_t)
        if i == target_idx:
            progress = float(i) / float(max(N - 1, 1))
            return x_t.detach(), vec_t.detach(), target_idx + 1, progress
        stepsize = timesteps[i] - timesteps[i + 1]
        x_t, _ = predictor.update_fn(x_t, y_main, vec_t, stepsize)
    raise RuntimeError("Failed to collect reverse state")


def save_path_model(path: str, model, optimizer, step: int, args, stats: Dict[str, float]):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "pathsim_vA",
            "version": 1,
            "state_dict": model.state_dict(),
            "config": model.get_config(),
            "optimizer": optimizer.state_dict(),
            "step": int(step),
            "train_args": vars(args),
            "last_stats": stats,
        },
        out,
    )
    print(f"[save] {out}")


def main():
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument("--backbone", default="ncsnpp_v2", choices=BackboneRegistry.get_all_names())
    bootstrap.add_argument("--sde", default="ouve", choices=SDERegistry.get_all_names())
    bootstrap.add_argument("--dataset", default="joint_scp", choices=("joint_specs", "joint_scp"))
    bootstrap.add_argument("--train_type", default="mode2", choices=("mode2", "mode3"))
    known, _ = bootstrap.parse_known_args()

    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default=known.backbone, choices=BackboneRegistry.get_all_names())
    parser.add_argument("--sde", default=known.sde, choices=SDERegistry.get_all_names())
    parser.add_argument("--dataset", default=known.dataset, choices=("joint_specs", "joint_scp"))
    parser.add_argument("--train_type", default=known.train_type, choices=("mode2", "mode3"))
    parser.add_argument("--baseline_ckpt", required=True)
    parser.add_argument("--output_ckpt", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--use_ema", action="store_true")

    train_group = parser.add_argument_group("PathSimTrain")
    train_group.add_argument("--pathsim_steps", type=int, default=30000)
    train_group.add_argument("--pathsim_lr", type=float, default=2e-4)
    train_group.add_argument("--pathsim_weight_decay", type=float, default=1e-5)
    train_group.add_argument("--pathsim_grad_clip", type=float, default=1.0)
    train_group.add_argument("--pathsim_kl_weight", type=float, default=0.005)
    train_group.add_argument("--pathsim_smooth_weight", type=float, default=0.003)
    train_group.add_argument("--pathsim_log_interval", type=int, default=50)
    train_group.add_argument("--pathsim_save_interval", type=int, default=5000)
    train_group.add_argument("--pathsim_N", type=int, default=30)
    train_group.add_argument("--pathsim_progress_min", type=float, default=0.10)
    train_group.add_argument("--pathsim_progress_max", type=float, default=0.90)
    train_group.add_argument("--pathsim_predictor", default="reverse_diffusion")
    train_group.add_argument("--pathsim_corrector", default="none")
    train_group.add_argument("--pathsim_corrector_steps", type=int, default=1)
    train_group.add_argument("--pathsim_snr", type=float, default=0.5)
    train_group.add_argument("--pathsim_hidden_ch", type=int, default=48)
    train_group.add_argument("--pathsim_num_bands", type=int, default=32)
    train_group.add_argument("--pathsim_num_time_blocks", type=int, default=16)
    train_group.add_argument("--pathsim_min_rho", type=float, default=0.5)
    train_group.add_argument("--pathsim_max_rho", type=float, default=2.0)

    JointFrontendSgmseModel.add_argparse_args(parser.add_argument_group("JointModel"))
    SDERegistry.get_by_name(known.sde).add_argparse_args(parser.add_argument_group("SDE"))
    BackboneRegistry.get_by_name(known.backbone).add_argparse_args(parser.add_argument_group("Backbone"))
    StormFrontend.add_argparse_args(parser)
    if known.dataset == "joint_scp":
        from sgmse.joint_data_module import JointWaveScpDataModule as data_cls
    else:
        from sgmse.joint_data_module import JointWaveDataModule as data_cls
    data_cls.add_argparse_args(parser.add_argument_group("DataModule"))

    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device)

    groups = argparse_groups(parser, args)
    kwargs = {
        **vars(groups["JointModel"]),
        **vars(groups["SDE"]),
        **vars(groups["Backbone"]),
        **vars(groups["DataModule"]),
    }
    # PathSim must be disabled while collecting the frozen baseline trajectory.
    # Keep the checkpoint's Anchor V1/V2 architecture arguments unchanged so
    # that state_dict and full-model EMA can be restored exactly.  The reverse
    # sampler below does not enable either anchor, so the controllers are never
    # used to modify the collected trajectory.
    kwargs["pathsim_va_enable"] = False
    model = JointFrontendSgmseModel(
        backbone=args.backbone,
        sde=args.sde,
        train_type=args.train_type,
        data_module_cls=data_cls,
        **kwargs,
    )
    load_baseline_weights(model, args.baseline_ckpt, args.use_ema)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    model.data_module.setup("fit")
    loader = model.data_module.train_dataloader()
    iterator = iter(loader)

    path_model = BandTimeVariancePathEstimator(
        hidden_ch=args.pathsim_hidden_ch,
        num_bands=args.pathsim_num_bands,
        num_time_blocks=args.pathsim_num_time_blocks,
        min_rho=args.pathsim_min_rho,
        max_rho=args.pathsim_max_rho,
    ).to(device)
    path_model.train()
    optimizer = torch.optim.AdamW(
        path_model.parameters(), lr=args.pathsim_lr, weight_decay=args.pathsim_weight_decay
    )

    running = {}
    for step in range(1, args.pathsim_steps + 1):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        clean_wav, noisy_wav = batch[0].to(device), batch[1].to(device)

        with torch.no_grad():
            X, Y_noisy, Y_pre = frontend_specs(model, clean_wav, noisy_wav)
            if args.train_type == "mode2":
                y_main, y_cond = Y_pre, Y_noisy
            else:
                y_main, y_cond = Y_noisy, Y_pre
            x_roll, t, step_index, progress = reverse_state_at_random_step(
                model=model,
                y_main=y_main,
                y_cond=y_cond,
                N=args.pathsim_N,
                progress_min=args.pathsim_progress_min,
                progress_max=args.pathsim_progress_max,
                predictor_name=args.pathsim_predictor,
                corrector_name=args.pathsim_corrector,
                corrector_steps=args.pathsim_corrector_steps,
                snr=args.pathsim_snr,
            )
            mean, std = model.sde.marginal_prob(X, y_main, t)
            z_target = standardized_rollout_residual(x_roll, mean, std)

        pred = path_model(Y_pre.detach(), Y_noisy.detach(), t.detach(), t_max=float(model.sde.T))
        nll, nll_stats = variance_nll_loss(pred["log_var_tile"], z_target.detach())
        kl = standard_normal_kl(pred["log_var_tile"])
        smooth = total_variation(pred["log_var_tile"])
        loss = nll + args.pathsim_kl_weight * kl + args.pathsim_smooth_weight * smooth

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(path_model.parameters(), args.pathsim_grad_clip)
        optimizer.step()

        stats = {
            "loss": float(loss.detach().cpu()),
            "nll": float(nll.detach().cpu()),
            "kl": float(kl.detach().cpu()),
            "smooth": float(smooth.detach().cpu()),
            "rho": float(nll_stats["pred_rho_mean"].cpu()),
            "target_energy": float(nll_stats["target_energy_mean"].cpu()),
            "target_step": float(step_index),
            "progress": float(progress),
        }
        for key, value in stats.items():
            running[key] = 0.98 * running.get(key, value) + 0.02 * value

        if step == 1 or step % args.pathsim_log_interval == 0:
            print(
                f"step={step:07d} loss={running['loss']:.5f} nll={running['nll']:.5f} "
                f"rho={running['rho']:.4f} targetE={running['target_energy']:.4f} "
                f"progress={running['progress']:.3f}",
                flush=True,
            )
        if step % args.pathsim_save_interval == 0:
            stem, ext = os.path.splitext(args.output_ckpt)
            save_path_model(f"{stem}-step{step}{ext or '.pt'}", path_model, optimizer, step, args, running)

    save_path_model(args.output_ckpt, path_model, optimizer, args.pathsim_steps, args, running)


if __name__ == "__main__":
    main()
