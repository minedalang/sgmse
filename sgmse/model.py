import time
from math import ceil
import warnings
import torch.nn.functional as F
from transformers import UniSpeechSatModel
from functools import partial

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
        parser.add_argument(
            "--detach",
            action="store_true",
            help="Detach y when constructing SDE marginal_prob, e.g. mean,std = sde.marginal_prob(x, y.detach(), t).",
        )
        # 对比学习参数
        parser.add_argument("--contrastive_weight", type=float, default=0.0,
                            help="Weight of PLM-based contrastive loss (0.0 disables it).")
        parser.add_argument("--plm_name_or_path", type=str,
                            default="microsoft/unispeech-sat-base-plus-sv",
                            help="Hugging Face model name/path for UniSpeech-SAT.")
        parser.add_argument("--contrastive_temperature", type=float, default=0.1,
                            help="Temperature for InfoNCE contrastive loss.")
        parser.add_argument("--max_plm_length", type=int, default=32000,
                            help="Max waveform length (in samples) fed into PLM.")
        return parser

    def __init__(
        self, backbone, sde, train_type: str = "default",lr=1e-4, ema_decay=0.999, t_eps=0.03, num_eval_files=20, loss_type='score_matching', 
        loss_weighting='sigma^2', network_scaling=None, c_in='1', c_out='1', c_skip='0', sigma_data=0.1, 
        l1_weight=0.001, pesq_weight=0.0, sr=16000, detach=False,data_module_cls=None, 
        contrastive_weight=0.0,
        plm_name_or_path="microsoft/unispeech-sat-base-plus-sv",
        contrastive_temperature=0.1,
        max_plm_length=32000,
        mode2_include_noisy_main=False,
        **kwargs
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
        self.train_type = train_type # 保存给 validation_step / enhance 用
        self.backbone = backbone
        self.mode2_include_noisy_main = bool(mode2_include_noisy_main)
        dnn_cls = BackboneRegistry.get_by_name(backbone)
        dnn_kwargs = dict(kwargs)
        dnn_kwargs["mode2_include_noisy_main"] = self.mode2_include_noisy_main
        self.dnn = dnn_cls(**dnn_kwargs)
        # Initialize SDE
        sde_cls = SDERegistry.get_by_name(sde)
        self.sde = sde_cls(**kwargs)
        # Store hyperparams and save them
        self.lr = lr
        self.ema_decay = ema_decay
        self.ema = None
        self._loaded_ema_state = None
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
        self.detach_sde_input = bool(detach)
        # Initialize PESQ loss if pesq_weight > 0.0
        if pesq_weight > 0.0:
            self.pesq_loss = PesqLoss(1.0, sample_rate=sr).eval()
            for param in self.pesq_loss.parameters():
                param.requires_grad = False
        self.save_hyperparameters(ignore=['no_wandb'])
        self.data_module = data_module_cls(**kwargs, gpu=kwargs.get('gpus', 0) > 0)
        # 初始化对比学习
        self.contrastive_weight = contrastive_weight
        self.plm_name_or_path = plm_name_or_path
        self.contrastive_temperature = contrastive_temperature
        self.max_plm_length = max_plm_length

        # 是否启用对比学习
        self.use_contrastive = self.contrastive_weight > 0.0
        self.plm_model = None
        if self.use_contrastive:
            try:
                # 预训练 UniSpeech-SAT
                self.plm_model = UniSpeechSatModel.from_pretrained(self.plm_name_or_path)
                self.plm_model.eval()
                for p in self.plm_model.parameters():
                    p.requires_grad = False  # 冻结参数，但仍允许对输入求梯度
                if self.sr != 16000:
                    warnings.warn(
                        f"UniSpeech-SAT expects 16 kHz audio, but sr={self.sr}. "
                        "You may want to resample before feeding into PLM."
                    )
            except Exception as e:
                warnings.warn(
                    f"Failed to load UniSpeech-SAT model '{self.plm_name_or_path}': {e}. "
                    "Contrastive loss will be disabled."
                )
                self.use_contrastive = False
                self.plm_model = None
    
    
    def _get_ema_named_parameters(self):
        """
        Full-model EMA, but exclude teacher/frozen helper branches if present.
        """
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            # 不把 teacher 分支放进 student EMA
            if name.startswith("teacher_") or ".teacher_" in name:
                continue
            yield name, p


    def _get_ema_parameters(self):
        for _, p in self._get_ema_named_parameters():
            yield p


    def _rebuild_ema(self):
        params = list(self._get_ema_parameters())
        self.ema = ExponentialMovingAverage(params, decay=self.ema_decay)
        self._error_loading_ema = False


    def _ensure_ema_is_compatible(self, force_rebuild: bool = False):
        params = list(self._get_ema_parameters())

        if self.ema is None or force_rebuild:
            self.ema = ExponentialMovingAverage(params, decay=self.ema_decay)
            self._error_loading_ema = False
            return

        shadow_params = getattr(self.ema, "shadow_params", None)
        shadow_len = len(shadow_params) if shadow_params is not None else -1

        if shadow_len != len(params):
            warnings.warn(
                f"EMA shadow param count ({shadow_len}) != current param count ({len(params)}), "
                "rebuilding EMA from current model parameters."
            )
            self.ema = ExponentialMovingAverage(params, decay=self.ema_decay)
            self._error_loading_ema = False
   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def optimizer_step(self, *args, **kwargs):
        # Method overridden so that the EMA params are updated after each optimizer step
        super().optimizer_step(*args, **kwargs)
        self._ensure_ema_is_compatible(force_rebuild=False)
        self.ema.update(self._get_ema_parameters())

    # on_load_checkpoint / on_save_checkpoint needed for EMA storing/loading
    def on_load_checkpoint(self, checkpoint):
        ema_state = checkpoint.get('ema', None)
        if ema_state is None:
            self._loaded_ema_state = None
            self._error_loading_ema = True
            warnings.warn("EMA state_dict not found in checkpoint!")
        else:
            # 先缓存，等模型完整构建后再尝试加载
            self._loaded_ema_state = ema_state
            self._error_loading_ema = False

    def on_fit_start(self):
        super().on_fit_start()

        # 先按“当前完整模型参数”确保 EMA 结构正确
        self._ensure_ema_is_compatible(force_rebuild=False)

        # 再尝试加载旧 checkpoint 里的 EMA
        if self._loaded_ema_state is not None:
            try:
                self.ema.load_state_dict(self._loaded_ema_state)

                current_len = len(list(self._get_ema_parameters()))
                shadow_len = len(self.ema.shadow_params)

                if shadow_len != current_len:
                    raise ValueError(
                        f"EMA shadow param count mismatch after load: "
                        f"{shadow_len} vs current {current_len}"
                    )

                print(f"[EMA] loaded successfully, shadow params = {shadow_len}")

            except Exception as e:
                warnings.warn(
                    f"EMA state incompatible with current model, reinitializing fresh EMA. "
                    f"Reason: {e}"
                )
                self._rebuild_ema()

            finally:
                self._loaded_ema_state = None
        else:
            # 没有旧 EMA，就保证当前 fresh EMA 正常
            self._ensure_ema_is_compatible(force_rebuild=False)

        print(f"[EMA] current tracked params = {len(list(self._get_ema_parameters()))}")
        if self.ema is not None:
            print(f"[EMA] current shadow params  = {len(self.ema.shadow_params)}")

    def on_save_checkpoint(self, checkpoint):
        if self.ema is None:
            checkpoint["ema"] = None
        else:
            checkpoint["ema"] = self.ema.state_dict()

    def train(self, mode, no_ema=False):
        if mode and self.ema is not None:
            try:
                params = list(self._get_ema_parameters())
                self.ema.restore(params)
            except Exception:
                pass

        return super().train(mode)

    def eval(self, no_ema=False):
        super().eval()

        # 如果请求 no_ema，先恢复原始训练参数
        if no_ema and self.ema is not None:
            try:
                self.ema.restore(list(self._get_ema_parameters()))
            except Exception:
                pass
            return self

        if self.ema is not None:
            self._ensure_ema_is_compatible(force_rebuild=False)
            params = list(self._get_ema_parameters())
            self.ema.store(params)
            self.ema.copy_to(params)

        return self
    
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
            loss = 0.5 * torch.sum(losses)
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
            loss = 0.5 * torch.sum(losses)
        elif self.loss_type == "data_prediction":
            x_hat = forward_out
            B, C, F, T = x.shape

            # losses in the time-frequency domain (tf)
            losses_tf = (1/(F*T))*torch.square(torch.abs(x_hat - x))
            losses_tf = 0.5 * torch.sum(losses_tf)

            # losses in the time domain (td)
            target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
            x_hat_td = self.to_audio(x_hat.squeeze(), target_len)
            x_td = self.to_audio(x.squeeze(), target_len)
            losses_l1 = (1 / target_len) * torch.abs(x_hat_td - x_td)
            losses_l1 = 0.5 * torch.sum(losses_l1)

            # losses using PESQ
            if self.pesq_weight > 0.0:
                losses_pesq = self.pesq_loss(x_td, x_hat_td)
                losses_pesq = torch.sum(losses_pesq)
                # combine the losses
                loss = losses_tf + self.l1_weight * losses_l1 + self.pesq_weight * losses_pesq 
            else:
                loss = losses_tf + self.l1_weight * losses_l1
        else:
            raise ValueError("Invalid loss type: {}".format(self.loss_type))

        return loss
    
    # 训练步（采样，前向传播，计算loss）
    def _step(self, batch, batch_idx, return_details: bool = False):
        """
        主训练 step。

        默认行为：
            返回一个标量 loss
        return_details=True 时：
            除了总 loss，还会返回若干中间量（如 x_hat_spec / t / x / y / y_cond）。
            这样 joint_model 就可以在不重写主训练逻辑的前提下，
            额外叠加 final-content loss / selector-content loss。
        支持：
        - (x, y)
        - (x, y, spk_ids)
        - (x, y_in, y_cond)
        - (x, y_in, y_cond, spk_ids)
        """
        y_cond = None
        spk_ids = None
        # 支持 (x, y) 或 (x, y, spk_ids)  后者要有说话人信息，为对比学习准备的
        if isinstance(batch, (tuple, list)):
            if len(batch) == 2:
                x, y = batch
                spk_ids = None
            elif len(batch) == 3:
                x, y, third = batch
                # 兼容：spk_ids 通常是 int 且 1D；否则当作 y_cond
                if torch.is_tensor(third) and third.dtype in (torch.int32, torch.int64) and third.dim() == 1:
                    spk_ids = third
                else:
                    y_cond = third
            elif len(batch) == 4:
                x, y, y_cond, spk_ids = batch
            else:
                x, y = batch[0], batch[1]
        else:
            x, y = batch

        # 如果是mode3，需要调换y和y_cond的顺序
        if self.train_type == "mode2" and y_cond is not None:
            # 让 主输入=enhanced, 条件=noisy
            y, y_cond = y_cond, y
        
        if self.train_type in ("mode2", "mode3"):
            assert y_cond is not None, "mode2/mode3 requires y_cond, but got None"

        # 采样一个x_t
        t = torch.rand(x.shape[0], device=x.device) * (self.sde.T - self.t_eps) + self.t_eps
        y_for_sde = y.detach() if self.detach_sde_input else y
        mean, std = self.sde.marginal_prob(x, y_for_sde, t)  #y=y_pre的情况需要加detach。在传播路径如y -> mean -> x_t -> net -> loss下，当y=y_pre时，y会收到不合适的梯度，semamba的训练目标发生偏移
        z = torch.randn_like(x)
        x_t = mean + std[:, None, None, None] * z

        forward_out = self(x_t, y, t, y_cond=y_cond)
        # 主恢复损失
        base_loss = self._loss(forward_out, x_t, z, t, mean, x, y)

        # 如果需要额外 loss，或者contrastive loss，就恢复增强频谱
        x_hat_spec = None
        if return_details or (self.use_contrastive and self.plm_model is not None and self.contrastive_weight > 0.0):
            x_hat_spec = self._get_enhanced_spec(forward_out, x_t, t, mean, x, y)

        # 对比学习损失
        if self.use_contrastive and self.plm_model is not None and self.contrastive_weight > 0.0:
            if x_hat_spec is not None:
                contrastive_loss = self._compute_contrastive_loss(x_hat_spec, x, spk_ids, weight_clean=1.0, weight_speaker=0.5)
                total_loss = base_loss + self.contrastive_weight * contrastive_loss
                self.log('contrastive_loss', contrastive_loss,
                         on_step=True, on_epoch=True, sync_dist=True, prog_bar=False)
            else:
                total_loss = base_loss
        else:
            total_loss = base_loss
        
        if return_details:
            return {
                "loss": total_loss,
                "base_loss": base_loss,
                "x_hat_spec": x_hat_spec,
                "forward_out": forward_out,
                "x_t": x_t,
                "t": t,
                "mean": mean,
                "x": x,
                "y": y,
                "y_cond": y_cond,
            }
        return total_loss
    
    def _compute_contrastive_loss(self, x_hat, x_clean, spk_ids=None, weight_clean=1.0, weight_speaker=0.5):
        """
        基于 UniSpeech-SAT 的 InfoNCE：
        - query: 增强语音嵌入
        - positive: 同一语音的干净嵌入（以及同说话人的其他样本可作为额外正例）
        - negative: 同 batch 中不同说话人的样本
        """
        B = x_hat.shape[0]
        if B < 2 or self.plm_model is None:
            # 没有负样本 / 没有 PLM 就不加约束
            return torch.tensor(0.0, device=x_hat.device)

        # 频谱转波形
        enhanced_wav = self._spec_to_wave_batch(x_hat)
        clean_wav = self._spec_to_wave_batch(x_clean)

        # 控制输入长度，避免 PLM OOM
        if self.max_plm_length is not None and enhanced_wav.shape[1] > self.max_plm_length:
            L = self.max_plm_length
            enhanced_wav = enhanced_wav[:, :L]
            clean_wav = clean_wav[:, :L]

        # 去均值 + 能量归一化到 [-1, 1] 左右
        enhanced_wav = enhanced_wav - enhanced_wav.mean(dim=1, keepdim=True)
        clean_wav = clean_wav - clean_wav.mean(dim=1, keepdim=True)
        enhanced_wav = enhanced_wav / (enhanced_wav.abs().max(dim=1, keepdim=True)[0] + 1e-6)
        clean_wav = clean_wav / (clean_wav.abs().max(dim=1, keepdim=True)[0] + 1e-6)

        # 通过UniSpeech-SAT提取高层表示
        plm_enh = self.plm_model(input_values=enhanced_wav, return_dict=True)
        plm_clean = self.plm_model(input_values=clean_wav, return_dict=True)

        # 使用 last_hidden_state 做时序特征，再做时间平均得到句级 embedding
        emb_enh = plm_enh.last_hidden_state.mean(dim=1)   # [B, D]
        emb_clean = plm_clean.last_hidden_state.mean(dim=1)

        # 获取同批次其他说话人的嵌入作为负样本
        # negative_embs = self._get_negative_samples(spk_ids, emb_enh, emb_clean)
        
        # 双向 InfoNCE（增强->干净，干净->增强）
        loss_q_to_k = self._info_nce_multi(emb_enh, emb_clean, spk_ids, weight_clean, weight_speaker)
        loss_k_to_q = self._info_nce_multi(emb_clean, emb_enh, spk_ids, weight_clean, weight_speaker)

        return 0.5 * (loss_q_to_k + loss_k_to_q)

    def _info_nce_multi(self, queries, keys, spk_ids=None,weight_clean=1.0, weight_speaker=0.5):
        """
        多正例 InfoNCE：
        - queries: [B, D]
        - keys   : [B, D]
        - negative_embs是负样本
        - spk_ids: 可选 [B]，相同说话人视作多正例；
                   若 None，则默认只有对角线是正例（一一对应）。
        """
        queries = F.normalize(queries, dim=-1)
        keys = F.normalize(keys, dim=-1)
        # key value相似度
        logits = torch.matmul(queries, keys.T) / self.contrastive_temperature 
        B = logits.size(0)

        # 没有说话人 ID：退化为标准一正样本 InfoNCE
        if spk_ids is None:
            labels = torch.arange(B, device=logits.device)
            return F.cross_entropy(logits, labels)

        # 有说话人 ID：同说话人多正例，不同说话人负样本
        if not torch.is_tensor(spk_ids):  
            spk_ids = torch.tensor(spk_ids, device=logits.device)
        else:
            spk_ids = spk_ids.to(logits.device)

        spk_ids = spk_ids.view(-1)
        if spk_ids.numel() != B:
            raise ValueError(
                "spk_ids长度必须等于batch size"
            )
        
        # 正样本的掩码：同说话人的样本为正例
        pos_mask = spk_ids.unsqueeze(0) == spk_ids.unsqueeze(1)  # [B, B] bool
        log_prob = F.log_softmax(logits, dim=1) #分母

        # 针对每个正样本进行计算
        losses = []
        for i in range(B):
            pos_idx = pos_mask[i].nonzero(as_tuple=False).squeeze(1)
            if pos_idx.numel() == 0:
                losses.append(-log_prob[i, i])
                continue
            log_pos = log_prob[i, pos_idx]  # 该 query 对所有正例的对数概率
            
            # 对正例的加权计算
            # 干净语音正样本的权重为 weight_clean，同一说话人其他语音正样本的加权为 weight_speaker
            weights = torch.tensor([weight_clean if spk_ids[idx] == spk_ids[i] else weight_speaker for idx in pos_idx], device=logits.device)
            log_mean_pos = torch.logsumexp(log_pos * weights, dim=0) - torch.log(
                torch.tensor(float(pos_idx.numel()), device=logits.device)
                )  
            # 对正例加权计算损失
            losses.append(-log_mean_pos)
        

        return torch.stack(losses).mean()
    def _get_negative_samples(self, spk_ids, emb_enh, emb_clean):
        """
        获取同批次的负样本：选择同一批次但不同说话人的样本作为负样本。
        """
        # 检查 spk_ids 是否有效
        if spk_ids is None:
            return None

        # 找到不同说话人的样本，排除相同说话人（正样本）
        unique_spk_ids = torch.unique(spk_ids)
        negative_embs = []

        # 选择与当前批次样本不同说话人的嵌入作为负样本
        for spk_id in unique_spk_ids:
            # 选择不同说话人的样本
            negative_samples = torch.cat(
                [emb_enh[spk_ids != spk_id], emb_clean[spk_ids != spk_id]], dim=0
            )  # 合并来自不同说话人的嵌入
            negative_embs.append(negative_samples)

        return torch.cat(negative_embs, dim=0)
    def _get_enhanced_spec(self, forward_out, x_t, t, mean, x, y):
        """
        负责从 forward_out 恢复增强频谱x_hat
        被用于计算InfoNCE、final enhanced和selector 的 content loss
        
        - data_prediction: forward_out 本身就是 x_hat
        - denoiser: 根据是否使用edm分情况处理
        - score_matching: 使用 Tweedie 公式 x_t + sigma^2 * score 做近似
        """

        if self.loss_type == "data_prediction":
            # 直接预测 x0 的情况
            return forward_out

        sigma_xt = self.sde._std(t)[:, None, None, None]

        if self.loss_type == "denoiser":
            if self.loss_weighting == "edm":
                if self.sde.__class__.__name__ == "SBVESDE":
                    # SBVE + EDM：forward_out 已经是 x0_hat
                    return forward_out
                else:
                    # 其他 SDE：forward_out 是 score，需要按 _loss 中的方式解回 x0_hat
                    score = forward_out
                    D = score * sigma_xt.pow(2) + x_t
                    w_xt, w_yt = self.sde.weights(t)
                    w_xt = w_xt[:, None, None, None]
                    w_yt = w_yt[:, None, None, None]
                    x0_hat = (D - w_yt * y) / (w_xt + 1e-12)
                    return x0_hat
            else:
                # 非 EDM：forward_out 也是 score，可以先算 μ_t 再按 weights 解回 x0_hat
                score = forward_out
                D = score * sigma_xt.pow(2) + x_t
                if hasattr(self.sde, "weights"):
                    w_xt, w_yt = self.sde.weights(t)
                    w_xt = w_xt[:, None, None, None]
                    w_yt = w_yt[:, None, None, None]
                    x0_hat = (D - w_yt * y) / (w_xt + 1e-12)
                    return x0_hat
                else:
                    # 没有 weights 的话，就把 μ_t 当成“去噪结果”用
                    return D

        elif self.loss_type == "score_matching":
            # 经典 Tweedie 近似：x0_hat ≈ x_t + σ_t^2 * score
            score = forward_out
            x0_hat = x_t + sigma_xt.pow(2) * score
            return x0_hat

        # 其它情况暂不支持
        return None
    
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
            # eval_files_per_gpu = self.num_eval_files // world_size
            num_eval = int(self.num_eval_files)
            per_gpu = int(math.ceil(num_eval / world_size))
            start = rank * per_gpu
            end   = min((rank + 1) * per_gpu, num_eval)

            clean_all  = self.data_module.valid_set.clean_files[:num_eval]
            noisy_all = self.data_module.valid_set.noisy_files[:num_eval]
            if self.train_type == "mode2": #将预增强wav作为输入
                enh_all   = self.data_module.valid_set.enhanced_files[:num_eval]
                clean_files = clean_all[start:end]
                y_cond_files = noisy_all[start:end]
                y_in_files = enh_all[start:end]
            elif self.train_type=="mode3":#将预增强wav作为条件
                enh_all   = self.data_module.valid_set.enhanced_files[:num_eval]
                clean_files = clean_all[start:end]
                y_in_files = noisy_all[start:end] 
                y_cond_files = enh_all[start:end] #互换预增强wav和原始noisy wav
            else:
                clean_files = clean_all[start:end]
                y_in_files = noisy_all[start:end]
                y_cond_files = None  # mode1 不用条件

           
            # 本卡累加（sum & count；用 16k 对齐后计算三项指标，避免采样率不一致）
            local_pesq_sum  = 0.0
            local_si_sdr_sum = 0.0
            local_estoi_sum = 0.0
            local_count     = 0.0

            with torch.no_grad():
                if self.train_type == "mode2" or self.train_type == "mode3" :
                    file_iter = zip(clean_files, y_in_files, y_cond_files) #mode2 & mode3
                else:
                    file_iter = zip(clean_files, y_in_files) 
                for item in file_iter:
                    if self.train_type == "mode2" or self.train_type == "mode3":
                        clean_file, y_in_file, y_cond_file = item #mode2 & mode3
                    else:
                        clean_file, y_in_file = item 
                    # 读入与检查
                    x, sr_x = load(clean_file)   # [C, T]
                    y_in, sr_y = load(y_in_file)
                     # mode2 读取 pre enhanced wav 作为输入 y_in
                    if self.train_type == "mode2" or self.train_type == "mode3":
                        y_cond, sr_e = load(y_cond_file)
                        assert sr_e == sr_y, f"Sample rates of enhanced({sr_e}) and noisy({sr_y}) do not match!"
                    else: #普通模式 输入的y就是y_cond
                        y_cond = y_in
                    
                    assert sr_x == sr_y, "Sample rates of clean and noisy files do not match!"
                    # 保证 datamodule/model 的 self.sr 与数据一致
                    # 后面 resample(orig_sr=self.sr) 可能用错
                    assert sr_y == self.sr, f"Data sr={sr_y} != model sr={self.sr}. Please align."

                    # numpy for metric/resample
                    x_np = x.squeeze().cpu().numpy()

                    # 统一到 16k 评测（clean）
                    x_16k = resample(x_np, orig_sr=sr_x, target_sr=16000).squeeze() if sr_x != 16000 else x_np

                    # 推理：mode2 用 (y_in=enhanced, y_cond=noisy)；mode1 用 (noisy,noisy)
                    x_hat = self.enhance(y_in, y_cond=y_cond, N=self.sde.N)  # x_hat: numpy (T,)

                    # 统一到 16k 评测（enhanced output）
                    x_hat_16k = resample(x_hat, orig_sr=sr_y, target_sr=16000).squeeze() if sr_y != 16000 else x_hat

                    # 计算指标
                    local_pesq_sum += pesq(16000, x_16k, x_hat_16k, 'wb')
                    local_si_sdr_sum += si_sdr(x_16k, x_hat_16k)
                    local_estoi_sum += stoi(x_16k, x_hat_16k, 16000, extended=True)
                    local_count += 1.0

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

    def forward(self, x_t, y, t, y_cond=None):
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
     
        y_cond = y if y_cond is None else y_cond
        y_extra = None
        if self.train_type == "mode2" and self.mode2_include_noisy_main and y_cond is not None:
            y_extra = y_cond
        # In [3], we use new code with backbone='ncsnpp_v2':
        if self.backbone == "ncsnpp_v2":
            # —— 在前向里转换到 EDM 坐标 —— 得到过程的权重，通过这个权重将x_t从原始坐标对应到edm坐标中
            if self.loss_weighting =="edm":
                w_xt, w_yt = self.sde.weights(t)                 # [B]
                x_in = (x_t - w_yt * y) / (w_xt + 1e-12)         # x̄_t
            else:
                x_in = x_t
            try:
                y_extra_in = None if y_extra is None else self._c_in(t) * y_extra
                F = self.dnn(self._c_in(t) * x_in,  self._c_in(t) * y, t, y_cond=self._c_in(t) * y_cond, y_extra=y_extra_in)
            except TypeError:
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
        if getattr(self, "plm_model", None) is not None:
            self.plm_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def get_pc_sampler(self, predictor_name, corrector_name, y, N=None, minibatch=None,score_fn=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        # 原来score_fn默认是self。但我通过修改score_fn，使采样器能调用score_fn(x,t,y,y_cond)，以传递y_cond
        score_fn = self if score_fn is None else score_fn

        if minibatch is None:
            return sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=score_fn, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_pc_sampler(predictor_name, corrector_name, sde=sde, score_fn=score_fn, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return samples, ns
            return batched_sampling_fn

    def get_ode_sampler(self, y, N=None, minibatch=None,score_fn=None, **kwargs):
        N = self.sde.N if N is None else N
        sde = self.sde.copy()
        sde.N = N

        kwargs = {"eps": self.t_eps, **kwargs}
        score_fn = self if score_fn is None else score_fn

        if minibatch is None:
            return sampling.get_ode_sampler(sde, score_fn, y=y, **kwargs)
        else:
            M = y.shape[0]
            def batched_sampling_fn():
                samples, ns = [], []
                for i in range(int(ceil(M / minibatch))):
                    y_mini = y[i*minibatch:(i+1)*minibatch]
                    sampler = sampling.get_ode_sampler(sde, score_fn, y=y_mini, **kwargs)
                    sample, n = sampler()
                    samples.append(sample)
                    ns.append(n)
                samples = torch.cat(samples, dim=0)
                return sample, ns
            return batched_sampling_fn

    def get_sb_sampler(self, sde, y, sampler_type="ode", N=None,score_fn=None, **kwargs):
        N = sde.N if N is None else N
        sde = sde.copy()
        sde.N = N
        score_fn = self if score_fn is None else score_fn
        return sampling.get_sb_sampler(sde, score_fn=score_fn, y=y, sampler_type=sampler_type,**kwargs)

    def train_dataloader(self):
        return self.data_module.train_dataloader(self.contrastive_weight)

    def val_dataloader(self):
        return self.data_module.val_dataloader()

    def test_dataloader(self):
        return self.data_module.test_dataloader()

    def setup(self, stage=None):
        return self.data_module.setup(stage=stage)

    def to_audio(self, spec, length=None):
        return self._istft(self._backward_transform(spec), length)
    
    def _spec_to_wave_batch(self, spec):
        """
        将一批频谱 [B, C, F, T] 转换成波形 [B, L]，
        L 按 data_module 的 num_frames / hop_length 计算。
        """
        B = spec.shape[0]
        target_len = (self.data_module.num_frames - 1) * self.data_module.hop_length
        waves = []
        for b in range(B):
            wav = self.to_audio(spec[b], length=target_len)
            if wav.dim() > 1:
                wav = wav.squeeze(0)
            waves.append(wav)
        return torch.stack(waves, dim=0)


    def _forward_transform(self, spec):
        return self.data_module.spec_fwd(spec)

    def _backward_transform(self, spec):
        return self.data_module.spec_back(spec)

    def _stft(self, sig):
        return self.data_module.stft(sig)

    def _istft(self, spec, length=None):
        return self.data_module.istft(spec, length)

    def enhance(self, y, y_cond=None, sampler_type="pc", predictor="reverse_diffusion",
        corrector="none", N=30, corrector_steps=1, snr=0.5, timeit=False,
        **kwargs
    ):
        """
        One-call speech enhancement of noisy speech `y`, for convenience.
        """
        start = time.time()

        # 保证 [1, T]
        if y.dim() == 1:
            y = y[None, :]
        y = y.to(self.device)
        T_orig = y.size(1)

        # y_cond 对齐
        if y_cond is not None:
            if y_cond.dim() == 1:
                y_cond = y_cond[None, :]
            y_cond = y_cond.to(self.device)

            if y_cond.size(1) != T_orig:
                if y_cond.size(1) > T_orig:
                    s = (y_cond.size(1) - T_orig) // 2
                    y_cond = y_cond[:, s:s + T_orig]
                else:
                    pad = T_orig - y_cond.size(1)
                    left = pad // 2
                    right = pad - left
                    y_cond = F.pad(y_cond, (left, right), mode="constant", value=0.0)
        
        # 归一：用 原始noisy作参考
        if y_cond is None:
            ref = y
        else:
            # mode2: y=preenh, y_cond=noisy -> ref应该是noisy
            # mode3: y=noisy, y_cond=preenh -> ref也应该是noisy
            ref = y_cond if self.train_type == "mode2" else y
        norm_factor = ref.abs().max().clamp_min(1e-12).item()
    
       
        y = y / norm_factor
        Y = torch.unsqueeze(self._forward_transform(self._stft(y.cuda())), 0)
        Y = pad_spec(Y)
        
        score_fn = None
        if y_cond is not None:
            y_cond_n = y_cond / norm_factor
            y_cond = torch.unsqueeze(self._forward_transform(self._stft(y_cond_n)), 0)
            y_cond = pad_spec(y_cond)
            # 采样器调用网络推理时 附加额外的y_cond参数
            score_fn = partial(self, y_cond=y_cond)
        
        # OUVESDE和VESDE选择的采样器，都接受分数
        if self.sde.__class__.__name__ in  ('OUVESDE', 'VESDE','VPSDE'):
            if self.sde.sampler_type == "pc":
                sampler = self.get_pc_sampler(predictor, corrector, Y.cuda(),score_fn=score_fn, N=N, 
                    corrector_steps=corrector_steps, snr=snr, intermediate=False,
                    **kwargs)
            elif self.sde.sampler_type == "ode":
                sampler = self.get_ode_sampler(Y.cuda(),score_fn=score_fn, N=N, **kwargs)
            else:
                raise ValueError("Invalid sampler type for SGMSE sampling: {}".format(sampler_type))
        # Schrödinger bridge sampling with VE SDE
        # sbve有专属采样器，接受预测的x0
        elif self.sde.__class__.__name__ == 'SBVESDE':
            sampler = self.get_sb_sampler(sde=self.sde, y=Y.cuda(),score_fn=score_fn, sampler_type=self.sde.sampler_type,N=N)
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
