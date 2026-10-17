import os, time, resource
def mark(msg):
    rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(
            f"[MARK] {time.strftime('%H:%M:%S')} "
            f"pid={os.getpid()} "
            f"ppid={os.getppid()} "
            f"RANK={os.environ.get('RANK')} "
            f"LOCAL_RANK={os.environ.get('LOCAL_RANK')} "
            f"WORLD_SIZE={os.environ.get('WORLD_SIZE')} "
            f"rss={rss_mb:.1f}MB :: {msg}",
            flush=True,
        )
mark("after imports")
print("\n========== Checkpoint Configuration ==========", flush=True)
print(
    "STORM_CHECKPOINT:",
    os.environ.get("SGMSE_STORM_CHECKPOINT", "0"),
    flush=True,
)
print(
    "STORM_CHECKPOINT_MODE:",
    os.environ.get("SGMSE_STORM_CHECKPOINT_MODE", "full"),
    flush=True,
)
print(
    "SCORE_CHECKPOINT:",
    os.environ.get("SGMSE_SCORE_CHECKPOINT", "0"),
    flush=True,
)
print(
    "PLM_CHECKPOINT:",
    os.environ.get("SGMSE_PLM_CHECKPOINT", "0"),
    flush=True,
)
print("==============================================\n", flush=True)

import torch
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from os.path import join
from sgmse.frontends.storm import StormFrontend
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.sdes import SDERegistry

from sgmse.joint_model import JointFrontendSgmseModel
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python /home/xyf/cwh_project/sgmse-main/joint_train.py \
  --sde sbve \
  --loss_type denoiser --loss_weighting edm \
  --c_in edm --c_out edm --c_skip edm \
  --cross_attn --film_modulate \
  --train_type mode3 \
  --dataset joint_scp \
  --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tr_synthetic/wav.scp \
  --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/cv_synthetic/wav.scp \
  --test_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tt_real_recordings/wav.scp \
  --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/clean \
  --batch_size 8 \
  --wandb_name sgmse-sbve-edm-crossattn-film-mode3-batchsize4*8-joint \
  --frontend semamba \
  --frontend_root /home/xyf/cwh_project/SEMamba-main/SEMamba-main \
  --frontend_config /home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/config.yaml \
  --frontend_ckpt /home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/g_00134000.pth \
  --frontend_trainable \
  --frontend_lr_mult 0.1 \
  --frontend_aux_loss time_l1 --frontend_aux_weight 0.1


CUDA_VISIBLE_DEVICES=4,5,6,7 python joint_train.py --nf 128 --ch_mult 1 2 2 2 --num_res_blocks 1 --attn_resolutions 0 --num_eval_files 5 --loss_type denoiser --loss_weighting edm --c_in edm --c_out edm --c_skip edm --cross_attn --train_type mode2 --selector_type tfgrid_a --selector_hidden_ch 32 --selector_num_blocks 2 --selector_attn_levels 6,5 --dataset joint_scp --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tr_synthetic/wav.scp --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/cv_synthetic/wav.scp --test_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tt_real_recordings/wav.scp --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/clean --batch_size 8 --wandb_id i7k0rhor --frontend semamba --frontend_trainable --frontend_root /home/xyf/cwh_project/SEMamba-main/SEMamba-main --frontend_config /home/xyf/cwh_project/SEMamba-main/SEMamba-main/exp/SEMamba_trainedby_dnslowsnr-20_-15/config.yaml --ckpt joint_train_logs/i7k0rhor-None/last.ckpt
CUDA_VISIBLE_DEVICES=4,5,6,7 python joint_train.py --backbone ncsnpp_v2 --sde sbve --loss_type denoiser --num_eval_files 5 --loss_weighting edm --c_in edm --c_out edm --c_skip edm --cross_attn --train_type mode2 --nf 128 --ch_mult 1 2 2 2 --num_res_blocks 1 --attn_resolutions 0 --selector_type tfgrid_a --selector_hidden_ch 32 --selector_num_blocks 2 --selector_attn_levels 3,2 --dataset joint_scp --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tr_synthetic/wav.scp --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/cv_synthetic/wav.scp --test_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tt_real_recordings/wav.scp --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/clean --batch_size 8 --accelerator gpu --devices 4 --precision 32-true --frontend storm --frontend_ckpt <STAGE1_FRONTEND_LAST_CKPT> --frontend_trainable --storm_nf 128 --storm_ch_mult 1 2 2 2 --storm_num_res_blocks 1 --storm_attn_resolutions 0 --init_ckpt <STAGE2_BACKEND_LAST_CKPT> --wandb_name storm_frontend_plus_selector_joint_stage3
'''

def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups

def torch_load_ckpt(path: str):
    # PyTorch 2.6: torch.load 默认 weights_only=True，会拒绝反序列化 Lightning ckpt 里的自定义类
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        # 兼容旧 torch（没有 weights_only 参数）
        return torch.load(path, map_location="cpu")
    
if __name__ == '__main__':
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()

    for parser_ in (base_parser, parser):
        parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp_v2")
        parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
        parser_.add_argument("--nolog", action='store_true')
        parser_.add_argument("--wandb_name", type=str, default=None)
        parser_.add_argument("--ckpt", type=str, default=None,help="用于恢复检查点ckpt,与init_ckpt冲突")
        parser_.add_argument("--init_ckpt", type=str, default=None,help="用于加载预训练ckpt")
        parser_.add_argument(
            "--init_ckpt_use_ema",
            action="store_true",
            help=(
                "Materialize the EMA weights stored in --init_ckpt into the initialized model "
                "before training. Recommended when --rrsd_freeze_base is used, so the frozen "
                "base exactly matches the original EMA inference baseline."
            ),
        )
        parser_.add_argument("--wandb_id", type=str, default=None)
        parser_.add_argument("--log_dir", type=str, default="joint_train_logs")
        parser_.add_argument("--seed", type=int, default=2026,
                             help="Seed Python, NumPy, PyTorch and dataloader workers for reproducible A/B runs.")
        parser_.add_argument("--save_ckpt_interval", type=int, default=50000)
        parser_.add_argument("--save_init_snapshot", action="store_true",
                             help="save a full checkpoint immediately after --init_ckpt loading")
        parser_.add_argument("--init_snapshot_name", type=str, default="init_loaded_snapshot.ckpt")
        parser_.add_argument(
            "--dataset",
            type=str,
            choices=("joint_specs", "joint_scp"),
            default="joint_specs",
            help="joint_specs: base_dir train/valid/test; joint_scp: wav.scp"
        )
        parser_.add_argument(
            "--train_type",
            type=str,
            choices=("mode2", "mode3"),
            default="mode3",
            help="mode3: y=noisy, y_cond=SEMamba(noisy). mode2: y=SEMamba(noisy), y_cond=noisy"
        )
        # 前端相关参数（semamba/storm/tfgridnet）在 JointFrontendSgmseModel.add_argparse_args 中
    temp_args, _ = base_parser.parse_known_args()

    backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
    sde_class = SDERegistry.get_by_name(temp_args.sde)

    trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="gpu")
    trainer_parser.add_argument("--devices", default="auto")
    trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    trainer_parser.add_argument("--max_epochs", type=int, default=-1)
    trainer_parser.add_argument("--max_steps", type=int, default=1_000_000,
                                help="Exact step limit for controlled RRSD fine-tuning runs.")
    trainer_parser.add_argument("--precision", type=str, default="32-true")

    # model / sde / backbone args
    JointFrontendSgmseModel.add_argparse_args(parser.add_argument_group("JointModel", description=JointFrontendSgmseModel.__name__))
    sde_class.add_argparse_args(parser.add_argument_group("SDE", description=sde_class.__name__))
    backbone_cls.add_argparse_args(parser.add_argument_group("Backbone", description=backbone_cls.__name__))
    StormFrontend.add_argparse_args(parser)
    # data module args
    if temp_args.dataset == "joint_scp":
        from sgmse.joint_data_module import JointWaveScpDataModule as data_module_cls
    else:
        from sgmse.joint_data_module import JointWaveDataModule as data_module_cls
    # 允许 PyTorch 安全加载 checkpoint 中的自定义 DataModule 类
    torch.serialization.add_safe_globals([data_module_cls])
    
    data_module_cls.add_argparse_args(parser.add_argument_group("DataModule", description=data_module_cls.__name__))

    args = parser.parse_args()
    if args.init_ckpt is not None and args.ckpt is not None: #参数互斥
        raise SystemExit("--ckpt(resume) 和 --init_ckpt(init weights) 不能同时使用")
    if args.init_ckpt_use_ema and args.init_ckpt is None:
        raise SystemExit("--init_ckpt_use_ema requires --init_ckpt")
    arg_groups = get_argparse_groups(parser, args)
    pl.seed_everything(int(args.seed), workers=True)
    mark("after argument parsing")
    mark("before model init")

    model = JointFrontendSgmseModel(
        backbone=args.backbone,
        sde=args.sde,
        train_type=args.train_type,
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups['JointModel']),
            **vars(arg_groups['SDE']),
            **vars(arg_groups['Backbone']),
            **vars(arg_groups['DataModule']),
        },
    )
    mark("after model init")

    mark("before logger init")
    if args.nolog:
        logger = None
    else:
        logger = WandbLogger(project="sgmse_joint", log_model=False, save_dir=args.log_dir, name=args.wandb_name, id=args.wandb_id)
        # Code archiving can copy the whole repository into W&B's artifact
        # staging directory (often on the small root filesystem).  Keep the
        # historical behavior by default, but allow disk-constrained runs to
        # disable this optional upload while retaining metric/checkpoint logs.
        if os.environ.get("SGMSE_WANDB_LOG_CODE", "1").strip().lower() not in {"0", "false", "no"}:
            logger.experiment.log_code(".")
        else:
            print("[W&B] code archive disabled by SGMSE_WANDB_LOG_CODE", flush=True)
    mark("after logger init")
    mark("before callbacks init")
    callbacks = None
    if logger is not None:
        run_dir = join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}")
        callbacks = [
            ModelCheckpoint(dirpath=run_dir, save_last=True, filename='{epoch}-{step}-last'),
            ModelCheckpoint(dirpath=run_dir, filename='{epoch}-{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval),
            ModelCheckpoint(dirpath=run_dir, filename='{epoch}-{step}-{valid_loss:.3f}', monitor='valid_loss', mode='min', save_top_k=1, save_on_train_epoch_end=False),
        ]
    mark("after callbacks init")
    mark("before trainer init")
    strategy = "auto"
    if int(args.devices) > 1:
        strategy = "ddp_find_unused_parameters_true"
    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy=strategy,
        # SpeakerBatchSampler performs its own rank sharding when contrastive
        # training is enabled. Prevent Lightning from replacing/wrapping it with
        # another distributed sampler.
        use_distributed_sampler=not (float(getattr(args, "contrastive_weight", 0.0)) > 0.0),
        # Lightning treats ``logger=None`` as "use the default logger" in
        # recent releases.  ``--nolog`` must therefore pass an explicit
        # ``False`` or it will still create ``lightning_logs`` and checkpoint
        # callbacks may write to the repository filesystem.
        logger=(logger if logger is not None else False),
        enable_checkpointing=(logger is not None),
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    mark("after trainer init")
    mark("before trainer.fit")

    # 针对两种ckpt，分情况加载
    # ---- optional: initialize weights from an existing checkpoint (no optimizer/loop restore) ----
    if args.init_ckpt is not None:
        ckpt = torch_load_ckpt(args.init_ckpt)

        if args.init_ckpt_use_ema:
            # The RRSD target model may already have --rrsd_freeze_base applied,
            # in which case its EMA parameter set contains only rrsd_router.*.
            # Therefore we cannot load the old full-model EMA directly into the
            # target model.  Instead, reconstruct the source checkpoint model,
            # let its existing eval(no_ema=False) path materialize its own EMA
            # shadows into live parameters, then copy those live parameters into
            # the new RRSD model.
            print(f"[Init ckpt] loading EMA-materialized base from: {args.init_ckpt}")
            source_model = JointFrontendSgmseModel.load_from_checkpoint(
                args.init_ckpt,
                map_location="cpu",
                strict=False,
                data_module_cls=data_module_cls,
            )
            source_model.eval(no_ema=False)
            state = source_model.state_dict()
            missing, unexpected = model.load_state_dict(state, strict=False)
            del source_model
            print(
                f"[Init ckpt] loaded EMA-materialized state_dict strict=False: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )
        else:
            state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
            missing, unexpected = model.load_state_dict(state, strict=False)
            print(
                f"[Init ckpt] loaded RAW state_dict strict=False: "
                f"missing={len(missing)} unexpected={len(unexpected)}"
            )

        def _show_keys(title, keys, prefix=None, limit=20):
            sel = [k for k in keys if prefix is None or k.startswith(prefix)]
            if not sel:
                print(f"[Init ckpt] {title}: none")
                return sel
            print(f"[Init ckpt] {title}: showing {min(len(sel), limit)} / {len(sel)}")
            for k in sel[:limit]:
                print(f"    {k}")
            return sel

        missing_frontend = _show_keys("missing frontend keys", missing, prefix="frontend.")
        unexpected_frontend = _show_keys("unexpected frontend keys", unexpected, prefix="frontend.")
        _show_keys("missing dnn keys", missing, prefix="dnn.")
        _show_keys("unexpected dnn keys", unexpected, prefix="dnn.")

        if missing_frontend or unexpected_frontend:
            raise SystemExit(
                "[Init ckpt] frontend weights did not load cleanly. "
                f"missing_frontend={len(missing_frontend)} unexpected_frontend={len(unexpected_frontend)}"
            )

        # Build the *new run* EMA from the target model's trainable parameters.
        # With --rrsd_freeze_base this intentionally tracks only rrsd_router.*,
        # while the frozen base remains fixed at the EMA-materialized 300k values.
        model._ensure_ema_is_compatible(force_rebuild=True)
        print(
            f"[Init ckpt] initialized new-run EMA over "
            f"{len(list(model._get_ema_parameters()))} trainable parameters."
        )

        if args.save_init_snapshot:
            import os
            snapshot_dir = run_dir if logger is not None else join(args.log_dir, f"manual-{args.wandb_name or 'run'}")
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = join(snapshot_dir, args.init_snapshot_name)
            snapshot = {
                "state_dict": model.state_dict(),
                "hyper_parameters": dict(model.hparams),
                "pytorch-lightning_version": pl.__version__,
            }
            try:
                model.on_save_checkpoint(snapshot)
            except Exception as e:
                print(f"[Init ckpt] warning: on_save_checkpoint failed while building init snapshot: {e}")
            torch.save(snapshot, snapshot_path)
            print(f"[Init ckpt] saved immediate snapshot to: {snapshot_path}")

    # ---- resume full training (weights + optimizer/scheduler/loop). ONLY use with a compatible joint checkpoint ----
    if args.ckpt is not None:
        # Quick guard: old SGMSE-only ckpt typically has 1 param group; joint training has 2.
        ckpt = torch_load_ckpt(args.ckpt)
        mark("after load ckpt")

        opt_states = ckpt.get("optimizer_states", None) if isinstance(ckpt, dict) else None
        if opt_states and len(opt_states) > 0 and isinstance(opt_states[0], dict) and "param_groups" in opt_states[0]:
            n_groups_ckpt = len(opt_states[0]["param_groups"])
            expected = len(model.configure_optimizers().param_groups)
            if n_groups_ckpt != expected:
                raise SystemExit(
                    f"ckpt optimizer param_groups mismatch: ckpt has {n_groups_ckpt}, expected {expected}. "
                    f"If you are starting joint training from an SGMSE-only checkpoint, use --init_ckpt instead of --ckpt."
                )

        trainer.fit(model, ckpt_path=args.ckpt)
        mark("after model init")
    else:
        trainer.fit(model, ckpt_path=None)
        mark("after model init")

