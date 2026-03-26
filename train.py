import torch
import wandb
import argparse
import pytorch_lightning as pl

from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from os.path import join

# Set CUDA architecture list and float32 matmul precision high
from sgmse.util.other import set_torch_cuda_arch_list
set_torch_cuda_arch_list()
torch.set_float32_matmul_precision('high')

from sgmse.backbones.shared import BackboneRegistry
from sgmse.sdes import SDERegistry
from sgmse.model import ScoreModel
'''
CUDA_VISIBLE_DEVICES=0,1,2,3 python /home/xyf/cwh_project/sgmse-main/train.py --sde sbve --loss_type denoiser --loss_weighting edm --c_in edm --c_out edm --c_skip edm --cross_attn --film_modulate --train_type mode3 --dataset dns_pre_enhance --train_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-15_-5/scp/tr_synthetic/wav.scp --valid_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-15_-5/scp/cv_synthetic/wav.scp --test_wav_scp /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-15_-5/scp/tt_real_recordings/wav.scp --clean_dir /home/xyf/cwh_project/DNS-Challenge-1/dns1-repo/synthetic_snr-15_-5/clean --enhanced_dir /home/xyf/cwh_project/SEMamba-main/SEMamba-main/dns1_synthetic_snr-15_-5_enhanced_by_vdpth --batch_size 8 --wandb_name sgmse-sbve-edm-crossattn-film-mode3-batchsize4*8-1-18 --ckpt logs/sinx0875-sgmse-sbve-edm-crossattn-film-mode3-batchsize4\*8-1-18/epoch\=1483-step\=500000.ckpt --wandb_id sinx0875
'''

def get_argparse_groups(parser):
     groups = {}
     for group in parser._action_groups:
          group_dict = { a.dest: getattr(args, a.dest, None) for a in group._group_actions }
          groups[group.title] = argparse.Namespace(**group_dict)
     return groups
# 定义自定义回调在wandb上展示epoch
class EpochLogger(pl.Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        # 记录当前epoch
        trainer.logger.log_metrics({"epoch": trainer.current_epoch})

if __name__ == '__main__':
     # throwaway parser for dynamic args - see https://stackoverflow.com/a/25320537/3090225
     base_parser = ArgumentParser(add_help=False)
     parser = ArgumentParser()
     for parser_ in (base_parser, parser):
          # parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp")
          parser_.add_argument("--backbone", type=str, choices=BackboneRegistry.get_all_names(), default="ncsnpp_v2")
          parser_.add_argument("--sde", type=str, choices=SDERegistry.get_all_names(), default="ouve")
          parser_.add_argument("--nolog", action='store_true', help="Turn off logging.")
          parser_.add_argument("--wandb_name", type=str, default=None, help="Name for wandb logger. If not set, a random name is generated.")
          parser_.add_argument("--ckpt", type=str, default=None, help="Resume training from checkpoint.")
          parser_.add_argument("--wandb_id", type=str, default=None, help="如果要在之前的wandb记录上接着训练，必须指定wandb_id")
          parser_.add_argument("--log_dir", type=str, default="logs", help="Directory to save logs.")
          parser_.add_argument("--save_ckpt_interval", type=int, default=50000, help="Save checkpoint interval.")
          parser_.add_argument(
               "--dataset",
               type=str,
               choices=("specs", "dns", "dns_pre_enhance"),
               default="specs",
               help="specs: 原始SpecsDataModule; dns: DNS(不读enhanced_wav); dns_pre_enhance: DNS(读enhanced_wav)"
          )
          parser_.add_argument(
               "--train_type",
               type=str,
               choices=("default", "mode2","mode3"),
               default="default",
               help="default: 原始训练(观测=Noisy). mode2: 观测=Enhanced, 条件注入=Noisy." \
               "mode3: 观测=Noisy, 条件注入=Enhanced"
               )
     temp_args, _ = base_parser.parse_known_args()

     # Add specific args for ScoreModel, pl.Trainer, the SDE class and backbone DNN class
     backbone_cls = BackboneRegistry.get_by_name(temp_args.backbone)
     sde_class = SDERegistry.get_by_name(temp_args.sde)
     trainer_parser = parser.add_argument_group("Trainer", description="Lightning Trainer")
     trainer_parser.add_argument("--accelerator", type=str, default="gpu", help="Supports passing different accelerator types.")
     trainer_parser.add_argument("--devices", default="auto", help="How many gpus to use.")
     trainer_parser.add_argument("--accumulate_grad_batches", type=int, default=1, help="Accumulate gradients.")
     trainer_parser.add_argument("--max_epochs", type=int, default=-1, help="Number of epochs to train.")

     ScoreModel.add_argparse_args(
          parser.add_argument_group("ScoreModel", description=ScoreModel.__name__))
     sde_class.add_argparse_args(
          parser.add_argument_group("SDE", description=sde_class.__name__))
     backbone_cls.add_argparse_args(
          parser.add_argument_group("Backbone", description=backbone_cls.__name__))
     

     # ====== 加载数据集 ======
     if temp_args.dataset == "dns_pre_enhance":
          from sgmse.data_module import DNS1EnhancedCondScpDataModule as data_module_cls
     elif temp_args.dataset == "dns":
          from sgmse.data_module import DNS1ScpDataModule as data_module_cls   # 不加载 enhanced_wav
     else:
          from sgmse.data_module import SpecsDataModule as data_module_cls
          
     data_module_cls.add_argparse_args(
          parser.add_argument_group("DataModule", description=data_module_cls.__name__))
    
      # Parse args and separate into groups
     args = parser.parse_args()
     if args.dataset == "dns" and (args.train_type == "mode2" or args.train_type == "mode3"):
          raise ValueError("dataset=dns 与 train_type=mode2/mode3冲突")

     arg_groups = get_argparse_groups(parser)
     # Initialize logger, trainer, model, datamodule
     model = ScoreModel(
          backbone=args.backbone, sde=args.sde, data_module_cls=data_module_cls,
          train_type=args.train_type,
          **{
               **vars(arg_groups['ScoreModel']),
               **vars(arg_groups['SDE']),
               **vars(arg_groups['Backbone']),
               **vars(arg_groups['DataModule'])
          }
     )

     # Set up logger configuration
     if args.nolog:
          logger = None
     else:
          logger = WandbLogger(project="sgmse", log_model=False, save_dir="logs", name=args.wandb_name, id=args.wandb_id)
          logger.experiment.log_code(".")
     
     callbacks = []
     # 早停：连续若干次验证没有提升就停
     early_stop = EarlyStopping(
          monitor="train_loss_epoch",   # 指标名要和你 self.log 的完全一致
          mode="min",           
          patience=8,           # “耐心值”：连续 10 次验证无提升就停
          min_delta=20,       # 认为“有提升”的最小幅度
          verbose=True
     )
     callbacks = [early_stop, EpochLogger()]

     # Set up callbacks for logger
     if logger != None:
          # 保存最新权重
          callbacks = [ModelCheckpoint(dirpath=join(args.log_dir,  f'{str(logger.version)}-{args.wandb_name}'), save_last=True, 
               filename='{epoch}-{step}-last',monitor=None)]
          # 每50000步保存一次权重
          callbacks += [ModelCheckpoint(dirpath=join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}'),
               filename='{epoch}-{step}', save_top_k=-1, every_n_train_steps=args.save_ckpt_interval,monitor=None)]
          # 每100个epoch保存一次权重
          # callbacks += [ModelCheckpoint(dirpath=join(args.log_dir, f'{str(logger.version)}-{args.wandb_name}'),
          #      filename='{epoch}-{step}', save_top_k=-1, every_n_epochs=100, monitor=None)]
          # 保存验证集最好权重
          callbacks += [ModelCheckpoint(
               dirpath=join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}"),
               filename="{epoch}-{step}-{valid_loss:.3f}",
               monitor="valid_loss", mode="min",
               save_top_k=1,
               save_on_train_epoch_end=False
          )]
          # 保存训练集最好权重
          callbacks += [ModelCheckpoint(
               dirpath=join(args.log_dir, f"{str(logger.version)}-{args.wandb_name}"),
               filename="{epoch}-{step}-{train_loss:.3f}",
               monitor="train_loss", mode="min",
               save_top_k=1,
               save_on_train_epoch_end=True
          )]
     else:
          callbacks = None
     
   
     # Initialize the Trainer and the DataModule
     trainer = pl.Trainer(
          **vars(arg_groups['Trainer']),
          strategy="ddp", logger=logger,
          max_steps=1_000_000,
          log_every_n_steps=10, num_sanity_val_steps=0,
          check_val_every_n_epoch=1,
          callbacks=callbacks,
          precision=args.precision
     )
     # Train model
     trainer.fit(model, ckpt_path=args.ckpt)