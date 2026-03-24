import argparse
from argparse import ArgumentParser
from os.path import join

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from sgmse.util.other import set_torch_cuda_arch_list
from sgmse.frontend_pretrain_model import FrontendPretrainModel

set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

'''
python pretrain_frontend.py \
  --frontend tfgridnet \
  --frontend_root /home/xyf/cwh_project/espnet \
  --frontend_module espnet2.enh.separator.tfgridnet_separator \
  --frontend_class TFGridNet \
  --frontend_config TF-GridNet-config/tfgridnet.yaml \
  --dataset joint_scp \
  --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/tr_synthetic/wav.scp \
  --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/scp/cv_synthetic/wav.scp \
  --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-20_-15/clean \
  --clean_template clean_fileid_{fileid}.wav \
  --batch_size 8 \
  --num_workers 4 \
  --expected_sr 16000 \
  --lr 1e-4 \
  --pretrain_loss l1 \
  --accelerator cuda \
  --devices 2 \
  --max_epochs 100 \
  --wandb_name tfgridnet_pretrain_3-20


'''
def get_argparse_groups(parser, args):
    groups = {}
    for group in parser._action_groups:
        group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
        groups[group.title] = argparse.Namespace(**group_dict)
    return groups


def torch_load_ckpt(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


if __name__ == '__main__':
    base_parser = ArgumentParser(add_help=False)
    parser = ArgumentParser()

    for parser_ in (base_parser, parser):
        parser_.add_argument("--nolog", action='store_true')
        parser_.add_argument("--wandb_name", type=str, default=None)
        parser_.add_argument("--wandb_id", type=str, default=None)
        parser_.add_argument("--log_dir", type=str, default="frontend_pretrain_logs")
        parser_.add_argument("--ckpt", type=str, default=None, help="resume checkpoint")
        parser_.add_argument("--init_ckpt", type=str, default=None, help="only initialize model weights")
        parser_.add_argument("--save_ckpt_interval", type=int, default=50000)
        parser_.add_argument(
            "--dataset",
            type=str,
            choices=("joint_specs", "joint_scp"),
            default="joint_scp",
        )

    trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
    trainer_parser.add_argument("--accelerator", type=str, default="cuda")
    trainer_parser.add_argument("--devices", default="auto")
    trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1)
    trainer_parser.add_argument("--max_epochs", type=int, default=-1)

    FrontendPretrainModel.add_argparse_args(parser.add_argument_group("FrontendPretrain", description=FrontendPretrainModel.__name__))

    temp_args, _ = base_parser.parse_known_args()
    if temp_args.dataset == "joint_scp":
        from sgmse.joint_data_module import JointWaveScpDataModule as data_module_cls
    else:
        from sgmse.joint_data_module import JointWaveDataModule as data_module_cls
    data_module_cls.add_argparse_args(parser.add_argument_group("DataModule", description=data_module_cls.__name__))

    args = parser.parse_args()
    if args.init_ckpt is not None and args.ckpt is not None:
        raise SystemExit("--ckpt(resume) 和 --init_ckpt(init weights) 不能同时使用")

    arg_groups = get_argparse_groups(parser, args)
    model = FrontendPretrainModel(
        data_module_cls=data_module_cls,
        **{
            **vars(arg_groups['FrontendPretrain']),
            **vars(arg_groups['DataModule']),
        },
    )

    if args.nolog:
        logger = None
    else:
        logger = WandbLogger(project="frontend_pretrain", log_model=False, save_dir=args.log_dir, name=args.wandb_name, id=args.wandb_id)
        logger.experiment.log_code(".")

    callbacks = None
    if logger is not None:
        run_dir = join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}")
        callbacks = [
            ModelCheckpoint(dirpath=run_dir, save_last=True, filename='{epoch}-{step}-last'),
            ModelCheckpoint(dirpath=run_dir, filename='{epoch}-{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval),
            ModelCheckpoint(dirpath=run_dir, filename='{epoch}-{step}-{valid_loss:.4f}', monitor='valid_loss', mode='min', save_top_k=1, save_on_train_epoch_end=False),
        ]

    trainer = pl.Trainer(
        **vars(arg_groups['Trainer']),
        strategy="ddp",
        logger=logger,
        max_steps=1_000_000,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        callbacks=callbacks,
        precision="bf16-mixed"
    )

    if args.init_ckpt is not None:
        ckpt = torch_load_ckpt(args.init_ckpt)
        state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[Init ckpt] loaded state_dict strict=False: missing={len(missing)} unexpected={len(unexpected)}")

    if args.ckpt is not None:
        trainer.fit(model, datamodule=model.data_module, ckpt_path=args.ckpt)
    else:
        trainer.fit(model, datamodule=model.data_module, ckpt_path=None)
