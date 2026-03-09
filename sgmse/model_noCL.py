import time
from math import ceil
import warnings

import torch
import pytorch_lightning as pl
import torch.distributed as dist
from torchaudio import load
from torch_ema import ExponentialMovingAverage
from librosa import resample

from sgmse import sampling
from sgmse.sdes import SDERegistry,SBVESDE
from sgmse.backbones import BackboneRegistry
from sgmse.util.inference import evaluate_model
from sgmse.util.other import pad_spec, si_sdr
from pesq import pesq
from pystoi import stoi
from torch_pesq import PesqLoss
import math

class ScoreModel(pl.LightningModule):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--lr", type=float, default=1e-4, help="The learning rate (1e-4 by default)")
        parser.add_argument("--ema_decay", type=float, default=0.999, help="The parameter EMA decay constant (0.999 by default)")
        parser.add_argument("--t_eps", type=float, default=0.03, help="The minimum process time (0.03 by default)")
        parser.add_argument("--num_eval_files", type=int, default=10, help="Number of files for speech enhancement performance evaluation during training. Pass 0 to turn off (no checkpoints based on evaluation metrics will be generated).")
        parser.add_argument("--loss_type", type=str, default="score_matching", help="The type of loss function to use.")
        parser.add_argument("--loss_weighting", type=str, default="sigma^2", help="The weighting of the loss function.")
        parser.add_argument("--network_scaling", type=str, default=None, help="The type of loss scaling to use.")
        parser.add_argument("--c_in", type=str, default="1", help="The input scaling for x.")
        parser.add_argument("--c_out", type=str, default="1", help="The output scaling.")
        parser.add_argument("--c_skip", type=str, default="0", help="The skip connection scaling.")
        parser.add_argument("--sigma_data", type=float, default=0.1, help="The data standard deviation.")
        parser.add_argument("--l1_weight", type=float, default=0.001, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--pesq_weight", type=float, default=0.0, help="The balance between the time-frequency and time-domain losses.")
        parser.add_argument("--sr", type=int, default=16000, help="The sample rate of the audio files.")
        return parser

    def __init__(
        self, backbone, sde, lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, data_module_cls=None, **kwargs
    ):
        """
        Create a new ScoreModel.

        Args:
            backbone: Backbone DNN that serves as a score-based model.
            sde: The SDE that defines the diffusion process.
            lr: The learning rate of the optimizer. (1e-4 by default).
            ema_decay: The decay constant of the parameter EMA (0.999 by default).
            t_eps: The minimum time to practically run for to avoid issues very close to zero (1e-5 by default).
            loss_type: The type of loss to use (wrt. noise z/std). Options are 'mse' (default), 'mae'
        """
        super().__init__()
        # Initialize Backbone DNN
        self.backbone = backbone
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        self.dnn = dnn_cls(**kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = ExponentialMovingAverage(self.parameters(), decay=self.ema_decay)
        self._error_loading_ema = False
        self.t_eps = t_eps
        self.loss_type = loss_type
        self.loss_weighting = loss_weighting
        self.l1_weight = l1_weight
        self.pesq_weight = pesq_weight
        self.network_scaling = network_scaling
        self.c_in = c_in
        self.c_out = c_out
        self.c_skip = c_skip
        self.sigma_data = sigma_data
        self.num_eval_files = num_eval_files
        self.sr = sr
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self.ema.update(self.dnn.parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema = checkpoint.get('ema', None)
        if ema is not None:
            self.ema.load_state_dict(checkpoint['ema'])
        else:
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")

    def on_save_checkpoint(self, checkpoint):
        checkpoint['ema'] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        res = super().train(mode)  # call the standard `train` method with the given mode
        if not self._error_loading_ema:
            if mode == False and not no_ema:
                # eval
                self.ema.store(self.dnn.parameters())        # store current params in EMA
                self.ema.copy_to(self.dnn.parameters())      # copy EMA parameters over current params for evaluation
            else:
                # train
                if self.ema.collected_params is not None:
                    self.ema.restore(self.dnn.parameters())  # restore the EMA weights (if stored)
        return res

    def eval(self, no_ema=False):
        return self.train(False, no_ema=no_ema)
    
    def _sigma_for_precond(self, t):
        # EDM+SBVE/ouve 需要用标准化后的σ_eff
        if self.loss_weighting == "edm":
            return self.sde._sigma_eff_for_edm(t)
        else:#不选择EDM的情况，依然使用原来的 std
            return self.sde._std(t)
        
    # loss计算
    def _loss(self, forward_out, x_t, z, t, mean, x, y):
        """
        Different loss functions can be used to train the score model, see the paper: 
        
        Julius Richter, Danilo de Oliveira, and Timo Gerkmann
        "Investigating Training Objectives for Generative Speech Enhancement"
        https://arxiv.org/abs/2409.10753

        """
        # 获取时间步噪声等级
        sigma_xt = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "score_matching":
            score = forward_out
            if self.loss_weighting == "sigma^2":
                losses = torch.square(torch.abs(score * sigma_xt + z)) # Eq. (7)
            else:
                raise ValueError("Invalid loss weighting for loss_type=score_matching: {}".format(self.loss_weighting))
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))
        elif self.loss_type == "denoiser": 
            if self.loss_weighting =="edm" :
                # SBVE + EDM：forward_out 就是 x0_hat
                if self.sde.__class__.__name__ == "SBVESDE":
                    x0_hat = forward_out
                    losses = torch.square(torch.abs(x0_hat - x))
                # 其他（比如 OUVESDE + EDM）：forward_out 是 score
                else:
                    score = forward_out
                    # 先从 score 还原 μ̂_t(x0_hat, y)
                    D = score * sigma_xt.pow(2) + x_t
                    # 再用 SB/OUVE 的 reweighting 解回 x0_hat
                    w_xt, w_yt = self.sde.weights(t)  # 保证 weights 返回 [B,1,1,1]
                    x0_hat = (D - w_yt * y) / (w_xt + 1e-12)
                    losses = torch.square(torch.abs(x0_hat - x))

                # 赋EDM 权重，要用 σ_eff计算权重
                sigma_eff = self._sigma_for_precond(t)[:, None, None, None]
                w = (sigma_eff**2 + self.sigma_data**2) / (
                    (sigma_eff * self.sigma_data) ** 2 + 1e-12
                )
                losses = w * losses

            else:
                # 非 EDM：保持原论文，拟合 μ_t
                 # 这里 forward_out 被当成 score 用
                score = forward_out
                # D(x_t, y, t) = x_t + Σ_xt^2 * score
                D = score * sigma_xt.pow(2) + x_t
                target = mean                      # μ_t(x0,y)
                losses = torch.square(torch.abs(D - target))

                if self.loss_weighting == "1":
                    pass
                elif self.loss_weighting == "sigma^2":
                    losses = losses * sigma_xt**2
                else: 
                    raise ValueError("Invalid loss weighting for loss_type=denoiser: {}".format(self.loss_weighting))
            
            # Sum over spatial dimensions and channels and mean over batch
            loss = torch.mean(0.5*torch.sum(losses.reshape(losses.shape[0], -1), dim=-1))   

        elif self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = torch.mean(0.5*torch.sum(losses_tf.reshape(losses_tf.shape[0], -1), dim=-1))

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = torch.mean(0.5*torch.sum(losses_l1.reshape(losses_l1.shape[0], -1), dim=-1))

            # losses using PESQ
            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.mean(losses_pesq)
                # combine the losses
                loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq 
            else:
                loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss
    
    # 训练步（采样，前向传播，计算loss）
    def _step(self, batch, batch_idx):
        x, y = batch

        # 采样一个x_t
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        mean, std = self.sde.marginal_prob(x, y, t) 
        z = torch.randn_like(x)
        x_t = mean + std[:, None, None, None] * z

        forward_out = self(x_t, y, t)
        loss = self._loss(forward_out, x_t, z, t, mean, x, y)
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch, batch_idx)
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        # Evaluate speech enhancement performance
        # 在每个验证轮的第一个batch做一次全文件评测
        if batch_idx == 0 and (self.num_eval_files or 0) > 0:
             # DDP 信息
            if dist.is_available() and dist.is_initialized():
                world_size = dist.get_world_size()
                rank = dist.get_rank()
            else:
                world_size, rank = 1, 0

             # 取前 num_eval_files 个样本，用“向上取整”的均匀切分，保证前几卡至少有样本
            num_eval = int(self.num_eval_files)
            per_gpu = int(math.ceil(num_eval / world_size))
            start = rank * per_gpu
            end   = min((rank + 1) * per_gpu, num_eval)

            # 将评估files分配到各个GPU
            # eval_files_per_gpu = self.num_eval_files // world_size
            clean_all  = self.data_module.valid_set.clean_files[:num_eval]
            noisy_all = self.data_module.valid_set.noisy_files[:num_eval]
            clean_files = clean_all[start:end]
            noisy_files = noisy_all[start:end]
           
            # 本卡累加（sum & count；用 16k 对齐后计算三项指标，避免采样率不一致）
            local_pesq_sum  = 0.0
            local_si_sdr_sum = 0.0
            local_estoi_sum = 0.0
            local_count     = 0.0

            with torch.no_grad():
                for clean_file, noisy_file in zip(clean_files, noisy_files):
                    # 读入与检查
                    x, sr_x = load(clean_file); x = x.squeeze().numpy()
                    y, sr_y = load(noisy_file)
                    assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"

                    # 统一到 16k 评测
                    x_16k = resample(x, orig_sr=sr_x, target_sr=16000).squeeze() if sr_x != 16000 else x
                    x_hat = self.enhance(y, N=self.sde.N)
                    x_hat_16k = resample(x_hat, orig_sr=self.sr, target_sr=16000).squeeze() if self.sr != 16000 else x_hat

                    # 计算指标（均在16k上，保证长度/采样率一致）
                    local_pesq_sum  += pesq(16000, x_16k, x_hat_16k, 'wb')
                    local_si_sdr_sum += si_sdr(x_16k, x_hat_16k)
                    local_estoi_sum += stoi(x_16k, x_hat_16k, 16000, extended=True)
                    local_count     += 1.0

            # 跨卡汇总成“全局 sum & count”
            t = torch.tensor([local_pesq_sum, local_si_sdr_sum, local_estoi_sum, local_count],
                            device=self.device, dtype=torch.float64)
            if dist.is_available() and dist.is_initialized() and world_size > 1:
                dist.all_reduce(t, op=dist.ReduceOp.SUM)

            g_count = max(float(t[3].item()), 1.0)   # 防止除零
            # t中三行分别是pesq/si/estoi总和
            g_pesq  = (t[0] / g_count).item()
            g_si    = (t[1] / g_count).item()
            g_estoi = (t[2] / g_count).item()
            # 每张卡都 log 相同的“全局均值”，键名与回调 monitor 对齐；我们已手动同步，sync_dist=False 即可
            self.log('val_pesq',  g_pesq,  on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
            self.log('val_si_sdr', g_si,   on_step=False, on_epoch=True, prog_bar=True,  sync_dist=False)
            self.log('val_estoi',  g_estoi,on_step=False, on_epoch=True, prog_bar=False, sync_dist=False)
           
        loss = self._step(batch, batch_idx)
        self.log('valid_loss', loss, on_step=False, on_epoch=True, sync_dist=True)

        return loss

    def forward(self, x_t, y, t):
        """
        The model forward pass. In [1] and [2], the model estimates the score function. In [3], the model estimates 
        either the score function or the target data for the Schrödinger bridge (loss_type='data_prediction').
        
        [1] Julius Richter, Simon Welker, Jean-Marie Lemercier, Bunlong Lay, and  Timo Gerkmann 
            "Speech Enhancement and Dereverberation with Diffusion-Based Generative Models"
            IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 31, pp. 2351-2364, 2023. 

        [2] Julius Richter, Yi-Chiao Wu, Steven Krenn, Simon Welker, Bunlong Lay, Shinji Watanabe, Alexander Richard, and Timo Gerkmann
            "EARS: An Anechoic Fullband Speech Dataset Benchmarked for Speech Enhancement and Dereverberation"
            ISCA Interspecch, Kos, Greece, Sept. 2024. 

        [3] Julius Richter, Danilo de Oliveira, and Timo Gerkmann
            "Investigating Training Objectives for Generative Speech Enhancement"
            https://arxiv.org/abs/2409.10753

        """
        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            # —— 在前向里转换到 EDM 坐标 —— 得到过程的权重，通过这个权重将x_t从原始坐标对应到edm坐标中
            if self.loss_weighting =="edm":
                w_xt, w_yt = self.sde.weights(t)                 # [B]
                x_in = (x_t - w_yt * y) / (w_xt + 1e-12)         # x̄_t
            else:
                x_in = x_t
            F = self.dnn(self._c_in(t) * x_in,  self._c_in(t) * y, t)
            
            # Scaling the network output, see below Eq. (7) in the paper
            if self.network_scaling == "1/sigma":
                std = self._sigma_for_precond(t)
                F = F / std[:, None, None, None]
            elif self.network_scaling == "1/t":
                F = F / t[:, None, None, None]
            # 根据不同的losstype返回不同的东西
            # The loss type determines the output of the model
            if self.loss_type == "score_matching":
                score = self._c_skip(t) * x_in + self._c_out(t) * F
                return score
            elif self.loss_type == "denoiser":
                sigmas = self._sigma_for_precond(t)[:, None, None, None]
                if self.loss_weighting =="edm":
                    if self.sde.__class__.__name__ in ('OUVESDE','VESDE',"VPSDE"): #OUVE的采样器需要score
                        x_hat_bar = self._c_skip(t) * x_in  + self._c_out(t) * F  #转为x0
                        # 还原到原坐标: 映射回时间步 t 的均值 μ_t(x0_hat, y)
                        x_hat = w_xt * x_hat_bar + w_yt * y
                        # 转为分数返回  
                        sigma_xt = self.sde._std(t)[:,None,None,None]
                        score = (x_hat - x_t) / (sigma_xt**2) # 对应 score = (x_hat - x_t)/σ_xt^2
                        return score
                    elif self.sde.__class__.__name__ == 'SBVESDE':#SBVE的采样器需要x0
                        x_hat = self._c_skip(t) * x_in  + self._c_out(t) * F   # ≈ x0_hat
                        return x_hat
                else:  # 非 EDM：跟原来一样，返回 score
                    score = (F - x_t) / sigmas.pow(2)
                    return score
            elif self.loss_type == 'data_prediction':
                # sb使用这种预测目标，直接预测x0
                x_hat = self._c_skip(t) * x_in + self._c_out(t) * F
                return x_hat
                
        # In [1] and [2], we use the old code:
        else:
            dnn_input = torch.cat([x_t, y], dim=1)            
            score = -self.dnn(dnn_input, t)
            return score

    def _c_in(self, t):
        if self.c_in == "1":
            return 1.0
        elif self.c_in == "edm":
            sigma = self._sigma_for_precond(t)
            return (1.0 / torch.sqrt(sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_in type: {}".format(self.c_in))
    
    def _c_out(self, t):
        if self.c_out == "1":
            return 1.0
        elif self.c_out == "sigma":
            return self._sigma_for_precond(t)[:, None, None, None]
        elif self.c_out == "1/sigma":
            return 1.0 / self._sigma_for_precond(t)[:, None, None, None] 
        elif self.c_out == "edm":
            sigma = self._sigma_for_precond(t)
            return ((sigma * self.sigma_data) / torch.sqrt(self.sigma_data**2 + sigma**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_out type: {}".format(self.c_out))
    
    def _c_skip(self, t):
        if self.c_skip == "0":
            return 0.0
        elif self.c_skip == "edm":
            sigma = self._sigma_for_precond(t)
            return (self.sigma_data**2 / (sigma**2 + self.sigma_data**2))[:, None, None, None]
        else:
            raise ValueError("Invalid c_skip type: {}".format(self.c_skip))

    def to(self, *args, **kwargs):
        """Override PyTorch .to() to also transfer the EMA of the model weights"""
        self.ema.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        if minibatch is None:
            return sampling.get_ode_sampler(sde, self, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, self, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None, **kwargs):
        N = sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N if N is not None else sde.N

        return sampling.get_sb_sampler(sde, self, y=y, sampler_type=sampler_type, **kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader()

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)

    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, sampler_type="pc", predictor="reverse_diffusion",
        corrector="ald", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()
        T_orig = y.size(1) 
        norm_factor = y.abs().max().item()
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)

        # OUVESDE和VESDE选择的采样器，都接受分数
        if self.sde.__class__.__name__ in  ('OUVESDE', 'VESDE','VPSDE'):
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(), N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False,
                    **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(), N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
        # Schrödinger bridge sampling with VE SDE
        # sbve有专属采样器，接受预测的x0
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y.cuda(), sampler_type=self.sde.sampler_type)
        else:
            raise ValueError("Invalid SDE type for speech enhancement: {}".format(self.sde.__class__.__name__))

        sample, nfe = sampler()
        x_hat = self.to_audio(sample.squeeze(), T_orig)
        x_hat = x_hat * norm_factor
        x_hat = x_hat.squeeze().cpu().numpy()
        end = time.time()
        if timeit:
            rtf = (end-start)/(len(x_hat)/self.sr)
            return x_hat, nfe, rtf
        else:
            return x_hat
