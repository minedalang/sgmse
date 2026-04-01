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
        parser_.add_argument("--wandb_id", type=str, default=None)
        parser_.add_argument("--log_dir", type=str, default="joint_train_logs")
        parser_.add_argument("--save_ckpt_interval", type=int, default=50000)
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

    data_module_cls.add_argparse_args(parser.add_argument_group("DataModule", description=data_module_cls.__name__))

    args = parser.parse_args()
    if args.init_ckpt is not None and args.ckpt is not None: #参数互斥
        raise SystemExit("--ckpt(resume) 和 --init_ckpt(init weights) 不能同时使用")
    arg_groups = get_argparse_groups(parser, args)

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

    if args.nolog:
        logger = None
    else:
        logger = WandbLogger(project="sgmse_joint", log_model=False, save_dir=args.log_dir, name=args.wandb_name, id=args.wandb_id)
        logger.experiment.log_code(".")

    callbacks = None
    if logger is not None:
        run_dir = join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}")
        callbacks = [
            ModelCheckpoint(dirpath=run_dir, save_last=True, filename='{epoch}-{step}-last'),
            ModelCheckpoint(dirpath=run_dir, filename='{epoch}-{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval),
            ModelCheckpoint(dirpath=run_dir, filename='{epoch}-{step}-{valid_loss:.3f}', monitor='valid_loss', mode='min', save_top_k=1, save_on_train_epoch_end=False),
        ]

    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy="ddp_find_unused_parameters_true",
        logger=logger,
        max_steps=1_000_000,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
    )
    # 针对两种ckpt，分情况加载
    # ---- optional: initialize weights from an existing checkpoint (no optimizer/loop restore) ----
    if args.init_ckpt is not None:
        ckpt = torch_load_ckpt(args.init_ckpt)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init ckpt] loaded state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")

        # 如果 ckpt 里有 EMA（你的 ScoreModel 会保存），顺手也恢复掉
        if isinstance(ckpt, dict) and "ema" in ckpt:
            try:
                model.on_load_checkpoint(ckpt)
                print("[Init ckpt] restored EMA state.")
            except Exception as e:
                print(f"[Init ckpt] failed to restore EMA state: {e}")

    # ---- resume full training (weights + optimizer/scheduler/loop). ONLY use with a compatible joint checkpoint ----
    if args.ckpt is not None:
        # Quick guard: old SGMSE-only ckpt typically has 1 param group; joint training has 2.
        ckpt = torch_load_ckpt(args.ckpt)
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
    else:
        trainer.fit(model, ckpt_path=None)
