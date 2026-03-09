"""
Abstract SDE classes, Reverse SDE, and VE/VP SDEs.

Taken and adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sde_lib.py
"""
import abc
import warnings

import numpy as np
from sgmse.util.tensors import batch_broadcast
import torch

from sgmse.util.registry import Registry


SDERegistry = Registry("SDE")


class SDE(abc.ABC):
    """SDE abstract class. Functions are designed for a mini-batch of inputs."""

    def __init__(self, N):
        """Construct an SDE.

        Args:
            N: number of discretization time steps.
        """
        super().__init__()
        self.N = N

    @property
    @abc.abstractmethod
    def T(self):
        """End time of the SDE."""
        pass

    @abc.abstractmethod
    def sde(self, x, y, t, *args):
        pass

    @abc.abstractmethod
    def marginal_prob(self, x, y, t, *args):
        """Parameters to determine the marginal distribution of the SDE, $p_t(x|args)$."""
        pass

    @abc.abstractmethod
    def prior_sampling(self, shape, *args):
        """Generate one sample from the prior distribution, $p_T(x|args)$ with shape `shape`."""
        pass

    @abc.abstractmethod
    def prior_logp(self, z):
        """Compute log-density of the prior distribution.

        Useful for computing the log-likelihood via probability flow ODE.

        Args:
            z: latent code
        Returns:
            log probability density
        """
        pass

    @staticmethod
    @abc.abstractmethod
    def add_argparse_args(parent_parser):
        """
        Add the necessary arguments for instantiation of this SDE class to an argparse ArgumentParser.
        """
        pass

    def discretize(self, x, y, t, stepsize):
        """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

        Useful for reverse diffusion sampling and probabiliy flow sampling.
        Defaults to Euler-Maruyama discretization.

        Args:
            x: a torch tensor
            t: a torch float representing the time step (from 0 to `self.T`)

        Returns:
            f, G
        """
        dt = stepsize
        drift, diffusion = self.sde(x, y, t)
        f = drift * dt
        G = diffusion * torch.sqrt(dt)
        return f, G

    def reverse(oself, score_model, probability_flow=False):
        """Create the reverse-time SDE/ODE.

        Args:
            score_model: A function that takes x, t and y and returns the score.
            probability_flow: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = oself.N
        T = oself.T
        sde_fn = oself.sde
        discretize_fn = oself.discretize

        # Build the class for reverse-time SDE.
        class RSDE(oself.__class__):
            def __init__(self):
                self.N = N
                self.probability_flow = probability_flow

            @property
            def T(self):
                return T

            def sde(self, x, y, t, *args):
                """Create the drift and diffusion functions for the reverse SDE/ODE."""
                rsde_parts = self.rsde_parts(x, y, t, *args)
                total_drift, diffusion = rsde_parts["total_drift"], rsde_parts["diffusion"]
                return total_drift, diffusion

            def rsde_parts(self, x, y, t, *args):
                sde_drift, sde_diffusion = sde_fn(x, y, t, *args)
                score = score_model(x, y, t, *args)
                score_drift = -sde_diffusion[:, None, None, None]**2 * score * (0.5 if self.probability_flow else 1.)
                diffusion = torch.zeros_like(sde_diffusion) if self.probability_flow else sde_diffusion
                total_drift = sde_drift + score_drift
                return {
                    'total_drift': total_drift, 'diffusion': diffusion, 'sde_drift': sde_drift,
                    'sde_diffusion': sde_diffusion, 'score_drift': score_drift, 'score': score,
                }

            def discretize(self, x, y, t, stepsize):
                """Create discretized iteration rules for the reverse diffusion sampler."""
                f, G = discretize_fn(x, y, t, stepsize)
                rev_f = f - G[:, None, None, None] ** 2 * score_model(x, y, t) * (0.5 if self.probability_flow else 1.)
                rev_G = torch.zeros_like(G) if self.probability_flow else G
                return rev_f, rev_G

        return RSDE()

    @abc.abstractmethod
    def copy(self):
        pass


@SDERegistry.register("ouve")
class OUVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--theta", type=float, default=1.5, help="The constant stiffness of the Ornstein-Uhlenbeck process. 1.5 by default.")
        parser.add_argument("--sigma-min", type=float, default=0.05, help="The minimum sigma to use. 0.05 by default.")
        parser.add_argument("--sigma-max", type=float, default=0.5, help="The maximum sigma to use. 0.5 by default.")
        parser.add_argument("--N", type=int, default=30, help="The number of timesteps in the SDE discretization. 30 by default")
        parser.add_argument("--sampler_type", type=str, default="pc", help="Type of sampler to use. 'pc' by default.")
        return parser

    def __init__(self, theta, sigma_min, sigma_max, N=30, sampler_type="pc", **ignored_kwargs):
        """Construct an Ornstein-Uhlenbeck Variance Exploding SDE.

        Note that the "steady-state mean" `y` is not provided at construction, but must rather be given as an argument
        to the methods which require it (e.g., `sde` or `marginal_prob`).

        dx = -theta (y-x) dt + sigma(t) dw

        with

        sigma(t) = sigma_min (sigma_max/sigma_min)^t * sqrt(2 log(sigma_max/sigma_min))

        Args:
            theta: stiffness parameter.
            sigma_min: smallest sigma.
            sigma_max: largest sigma.
            N: number of discretization steps
        """
        super().__init__(N)
        self.theta = theta
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        return OUVESDE(self.theta, self.sigma_min, self.sigma_max, N=self.N, sampler_type=self.sampler_type)

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        drift = self.theta * (y - x)
        # the sqrt(2*logsig) factor is required here so that logsig does not in the end affect the perturbation kernel
        # standard deviation. this can be understood from solving the integral of [exp(2s) * g(s)^2] from s=0 to t
        # with g(t) = sigma(t) as defined here, and seeing that `logsig` remains in the integral solution
        # unless this sqrt(2*logsig) factor is included.
        sigma = self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        diffusion = sigma * np.sqrt(2 * self.logsig)
        return drift, diffusion

    def _mean(self, x0, y, t):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp * x0 + (1 - exp_interp) * y
    
    # x_t重采样公式的 x和y的系数
    def weights(self,t):
        theta = self.theta
        exp_interp = torch.exp(-theta * t)[:, None, None, None]
        return exp_interp, (1 - exp_interp)
    # 适配edm的标准差
    def _sigma_eff_for_edm(self, t):
        sigma = self._std(t)        
        a = self.alpha(t) 
        a = a.clamp_min(1e-12)
        return sigma / a


    def alpha(self, t):
        return torch.exp(-self.theta * t)

    def _std(self, t):
        # This is a full solution to the ODE for P(t) in our derivations, after choosing g(s) as in self.sde()
        sigma_min, theta, logsig = self.sigma_min, self.theta, self.logsig
        # could maybe replace the two torch.exp(... * t) terms here by cached values **t
        return torch.sqrt(
            (
                sigma_min**2
                * torch.exp(-2 * theta * t)
                * (torch.exp(2 * (theta + logsig) * t) - 1)
                * logsig
            )
            /
            (theta + logsig)
        )
    
    
    def marginal_prob(self, x0, y, t):
        return self._mean(x0, y, t), self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std = self._std(torch.ones((y.shape[0],), device=y.device))
        x_T = y + torch.randn_like(y) * std[:, None, None, None]
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for OU SDE not yet implemented!")


@SDERegistry.register("sbve")
class SBVESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--N", type=int, default=50, help="The number of timesteps in the SDE discretization. 50 by default")
        parser.add_argument("--k", type=float, default=2.6, help="Parameter of the diffusion coefficient. 2.6 by default.")
        parser.add_argument("--c", type=float, default=0.4, help="Parameter of the diffusion coefficient. 0.4 by default.")
        parser.add_argument("--eps", type=float, default=1e-8, help="Small constant to avoid numerical instability. 1e-8 by default.")
        parser.add_argument("--sampler_type", type=str, default="ode")
        return parser

    def __init__(self, k, c, N=50, eps=1e-8, sampler_type="ode", **ignored_kwargs):
        """Construct a Schrodinger Bridge with Variance Exploding SDE.

        As described in Jukić et al., „Schrödinger Bridge for Generative Speech Enhancement“, 2024.

        Args:
            k: stiffness parameter.
            c: diffusion parameter.
            N: number of discretization steps
        """
        super().__init__(N)
        self.k = k
        self.c = c
        self.N = N
        self.eps = eps
        self.sampler_type = sampler_type

    def copy(self):
        return SBVESDE(self.k, self.c, N=self.N)

    @property
    def T(self):
        return 1

    def sde(self, x, y, t):
        """
        计算 SDE 中的漂移项（f）和扩散项（g）。
        
        公式：
        f = 0  # 无漂移项，模型设定为零
        g = sqrt(c) * k^t  # 扩散项，依赖于 t 和参数 c, k
        
        Args:
            x: 当前的状态变量。
            y: 目标信号（通常是干净的语音信号）。
            t: 当前的时间步。
        
        Returns:
            f, g: 漂移项和扩散项。
        """
        f = 0.0                                                                 # Table 1
        g = torch.sqrt(torch.tensor(self.c)) * self.k**(t)                      # Table 1
        return f, g

    def _sigmas_alphas(self, t):
        """
        计算模型中的 alpha_t 和 sigma_t 等参数，用于计算标准差和信号缩放。

        公式：    
        sigma_t^2 = (c * (k^(2t) - 1)) / (2 * log(k))
        alpha_t = 1  # alpha_t 定义为 1
        alpha_T = 1  # alpha_T 也是常数 1

        计算公式（公式9和10）：
        alpha_t / (alpha_T + eps) 为 alpha_bart
        sigma_bart = sqrt(sigma_T^2 - sigma_t^2 + eps)

        Args:
            t: 当前时间步。
        
        Returns:
            sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart: 各种参数，用于后续计算。
        """
        alpha_t = torch.ones_like(t)
        alpha_T = torch.ones_like(t)
        sigma_t = torch.sqrt((self.c*(self.k**(2*t)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1
        sigma_T = torch.sqrt((self.c*(self.k**(2*self.T)-1.0)) \
            / (2*torch.log(torch.tensor(self.k))))                              # Table 1   
        
        alpha_bart = alpha_t / (alpha_T + self.eps)                             # below Eq. (9)
        sigma_bart = torch.sqrt(sigma_T**2 - sigma_t**2 + self.eps)             # below Eq. (9)

        return sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart

    def _mean(self, x0, y, t):
        """
        计算从 `x0` 到 `y` 的条件期望mu(t)，作为 SDE 漂移项。

        公式：    
        mu = w_xt * x0 + w_yt * y
        其中:
        w_xt = (alpha_t * sigma_bart^2) / (sigma_T^2 + eps)
        w_yt = (alpha_bart * sigma_t^2) / (sigma_T^2 + eps)

        Args:
            x0: 原始信号（干净的语音信号）。
            y: 目标信号（带噪语音）。
            t: 当前时间步。
        
        Returns:
            mu: 计算得到的期望值，表示去噪后的信号。
        """
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        w_xt = alpha_t * sigma_bart**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        w_yt = alpha_bart * sigma_t**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        
        mu = w_xt[:, None, None, None] * x0 + w_yt[:, None, None, None] * y     # Eq. (11)
        return mu

    def _std(self, t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)

        sigma_xt = (alpha_t * sigma_bart * sigma_t) / (sigma_T + self.eps) 
        return sigma_xt
    # 
    def _sigma_eff_for_edm(self, t):
        # σ_eff = Σ_xt / w_xt = (σ_T * σ_t) / (σ̄_t)
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)
        return (sigma_T * sigma_t) / (sigma_bart + self.eps)
        
    def marginal_prob(self, x0, y, t):  
        return self._mean(x0, y, t), self._std(t)

    def weights(self,t):
        sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = self._sigmas_alphas(t)
        w_xt = alpha_t * sigma_bart**2 / (sigma_T**2 + self.eps)                # below Eq. (11)
        w_yt = alpha_bart * sigma_t**2 / (sigma_T**2 + self.eps)                # below Eq. (11)

        return w_xt[:, None, None, None] , w_yt[:, None, None, None] 



    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        x_T = y
        return x_T

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for SBVE SDE not yet implemented!")  



@SDERegistry.register("ve")
class VESDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--sigma-min", type=float, default=0.04, help="Minimum sigma.")
        parser.add_argument("--sigma-max", type=float, default=1.7, help="Maximum sigma.")
        parser.add_argument("--N", type=int, default=30, help="Number of time steps.")
        parser.add_argument("--sampler_type", type=str, default="pc", help="pc or ode.")
        return parser

    def __init__(self, sigma_min, sigma_max, N=30, sampler_type="pc", **ignored_kwargs):
        super().__init__(N)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.logsig = np.log(self.sigma_max / self.sigma_min)
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        return VESDE(self.sigma_min, self.sigma_max, N=self.N, sampler_type=self.sampler_type)

    @property
    def T(self):
        return 1

    # f=0, g(t)=σ(t)*sqrt(2*logσratio)
    def sde(self, x, y, t):
        drift = torch.zeros_like(x)
        sigma_t = self._std(t)
        diffusion = sigma_t * np.sqrt(2 * self.logsig)
        return drift, diffusion

    # 标准 VE 边际 std：σ(t)
    def _std(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t

    # EDM 里的 σ_eff；对于 VE，α(t)=1，因此就是 σ(t)
    def _sigma_eff_for_edm(self, t):
        return self._std(t)

    # 给 EDM 坐标变换用的权重：x_t = w_xt * x + w_yt * y；VE 与 y 无关 → w_xt=1, w_yt=0
    def weights(self, t):
        wxt = torch.ones_like(t)[:, None, None, None]
        wyt = torch.zeros_like(t)[:, None, None, None]
        return wxt, wyt

    # p_t(x|x0) = N(x0, σ(t)^2 I)
    def marginal_prob(self, x0, y, t):
        return x0, self._std(t)

    def prior_sampling(self, shape, y):
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Ignoring target shape.")
        std_T = self._std(torch.ones((shape[0],), device=y.device))
        # 关键：用 randn_like(y) 保持和 y 相同的 dtype（包括 complex）
        noise = torch.randn_like(y)
        x_T = noise * std_T[:, None, None, None]
        return x_T
        # return torch.randn(shape, device=y.device) * std_T[:, None, None, None]

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VE not implemented")

@SDERegistry.register("vp")
class VPSDE(SDE):
    @staticmethod
    def add_argparse_args(parser):
        parser.add_argument("--beta-min", type=float, default=0.01,
                            help="Minimum beta in the VP SDE. 0.01 by default.")
        parser.add_argument("--beta-max", type=float, default=1,
                            help="Maximum beta in the VP SDE. 1 by default.")
        parser.add_argument("--N", type=int, default=30,
                            help="The number of timesteps in the SDE discretization. 30 by default.")
        parser.add_argument("--sampler_type", type=str, default="pc",
                            help="Type of sampler to use. 'pc' or 'ode'.")
        return parser

    def __init__(self, beta_min, beta_max, N=30, sampler_type="pc", **ignored_kwargs):
        """
        VP SDE:
            d x = -1/2 * beta(t) * x dt + sqrt(beta(t)) dW_t
        其中 beta(t) 在线性插值:
            beta(t) = beta_min + t * (beta_max - beta_min),  t ∈ [0, 1].
        边际分布:
            x_t | x_0 ~ N(alpha(t) * x_0, (1 - alpha(t)^2) I)
        """
        super().__init__(N)
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.N = N
        self.sampler_type = sampler_type

    def copy(self):
        return VPSDE(self.beta_min, self.beta_max, N=self.N, sampler_type=self.sampler_type)

    @property
    def T(self):
        return 1.0

    # ------- 一些内部辅助函数 -------

    def _beta(self, t):
        """
        线性 beta(t) = beta_min + t * (beta_max - beta_min)
        t: [B]
        """
        return self.beta_min + t * (self.beta_max - self.beta_min)

    def _log_mean_coeff(self, t):
        """
        log(alpha(t))，参考 Song et al. 的 VPSDE 推导:

            alpha(t) = exp( -1/2 ∫_0^t beta(s) ds )
                      = exp( -0.25 * (beta_max - beta_min) * t^2
                             -0.5  * beta_min * t )
        """
        return -0.25 * (self.beta_max - self.beta_min) * t**2 - 0.5 * self.beta_min * t

    def alpha(self, t):
        """
        alpha(t) = exp(log_mean_coeff(t))
        """
        return torch.exp(self._log_mean_coeff(t))

    def _std(self, t):
        """
        标准 VP 边际 std:

            Var[x_t | x_0] = 1 - alpha(t)^2
        """
        log_alpha = self._log_mean_coeff(t)
        # exp(2 * log_alpha) = alpha^2
        var = 1.0 - torch.exp(2.0 * log_alpha)
        return torch.sqrt(var.clamp_min(1e-12))

    # 给 EDM 预条件化用的有效 sigma，这里就直接用 std(t)
    def _sigma_eff_for_edm(self, t):
        return self._std(t)

    # 时间步状态x_t的均值权重：x_t = w_xt * x + w_yt * y
    def weights(self, t):
        alpha_t = self.alpha(t)                        # [B]
        w_xt = alpha_t[:, None, None, None]
        w_yt = 0
        return w_xt, w_yt

    # ------- SDE 接口实现 -------

    def sde(self, x, y, t):
        """
        标准 VP SDE:
            d x = -1/2 * beta(t) * x dt + sqrt(beta(t)) dW_t

        不依赖 y，y 在此处只为接口兼容。
        """
        beta_t = self._beta(t)                         # [B]
        drift = -0.5 * beta_t[:, None, None, None] * x
        diffusion = torch.sqrt(beta_t)                 # [B]
        return drift, diffusion
 
    def marginal_prob(self, x0, y, t):
        """
        x_t | x0 ~ N(alpha(t) * x0, std(t)^2 I)
        """
        alpha_t = self.alpha(t)                        # [B]
        mean = alpha_t[:, None, None, None] * x0
        std = self._std(t)                             # [B]
        return mean, std

    def prior_sampling(self, shape, y):
        """
        prior p_T(x) ≈ N(0, I)（标准 VPSDE 近似），
        不依赖 y，这里只用 y.device / dtype 保持一致。
        """
        if shape != y.shape:
            warnings.warn(f"Target shape {shape} does not match shape of y {y.shape}! Using y.shape instead.")
            shape = y.shape

        # 生成与 y 同 dtype/device 的噪声（支持 complex）
        noise = torch.randn(shape, device=y.device, dtype=y.dtype)
        return noise

    def prior_logp(self, z):
        raise NotImplementedError("prior_logp for VP SDE not yet implemented!")
