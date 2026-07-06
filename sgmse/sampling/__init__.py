# Adapted from https://github.com/yang-song/score_sde_pytorch/blob/1618ddea340f3e4a2ed7852a0694a809775cf8d0/sampling.py
"""Various sampling methods."""
from scipy import integrate
import torch

from .predictors import Predictor, PredictorRegistry, ReverseDiffusionPredictor
from .correctors import Corrector, CorrectorRegistry


__all__ = [
    'PredictorRegistry', 'CorrectorRegistry', 'Predictor', 'Corrector',
    'get_sampler'
]


def to_flattened_numpy(x):
    """Flatten a torch tensor `x` and convert it to numpy."""
    return x.detach().cpu().numpy().reshape((-1,))


def from_flattened_numpy(x, shape):
    """Form a torch tensor with the given `shape` from a flattened numpy array `x`."""
    return torch.from_numpy(x.reshape(shape))




def _prepare_regen_strength(regeneration_strength, y):
    """Return a scalar regeneration strength in [0,1] for sampler control.

    The current samplers operate on a shared time grid for the whole mini-batch,
    so batched strengths are averaged. In normal enhancement scripts B=1, hence
    this remains per-utterance.
    """
    if regeneration_strength is None:
        return None
    if not torch.is_tensor(regeneration_strength):
        regeneration_strength = torch.tensor(float(regeneration_strength), device=y.device, dtype=torch.float32)
    regeneration_strength = regeneration_strength.to(device=y.device, dtype=torch.float32).detach().flatten()
    if regeneration_strength.numel() == 0:
        return None
    return regeneration_strength.clamp(0.0, 1.0).mean()


def _adaptive_prior_sampling(sde, y, t_start):
    """Sample an initial state at an adaptive diffusion time t_start.

    For OUVESDE this matches the usual conditional prior when t_start=1:
        x_t = y + std(t) * z.
    For VE/VP-style SDEs, the initial sample is centered at zero as in their
    standard prior. Unsupported SDEs fall back to the original prior.
    """
    t_vec = torch.ones((y.shape[0],), device=y.device, dtype=torch.float32) * float(t_start)
    name = sde.__class__.__name__
    if hasattr(sde, "_std"):
        std = sde._std(t_vec)
        noise = torch.randn_like(y)
        if name == "OUVESDE":
            return y + noise * std[:, None, None, None]
        if name in ("VESDE", "VPSDE"):
            return noise * std[:, None, None, None]
    return sde.prior_sampling(y.shape, y).to(y.device)


def get_pc_sampler(
    predictor_name, corrector_name, sde, score_fn, y,
    denoise=True, eps=3e-2, snr=0.1, corrector_steps=1, probability_flow: bool = False,
    intermediate=False, regeneration_strength=None, anchor_calibrator=None,
    score_calibrator=None, **kwargs
):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
        predictor_name: The name of a registered `sampling.Predictor`.
        corrector_name: The name of a registered `sampling.Corrector`.
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        denoise: If `True`, add one-step denoising to the final samples.
        eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
        snr: The SNR to use for the corrector. 0.1 by default, and ignored for `NoneCorrector`.
        N: The number of reverse sampling steps. If `None`, uses the SDE's `N` property by default.
        score_calibrator: Optional callable used to calibrate the predictor score
            as ``score_calibrator(x_t, y, t, score_base)``. The corrector is not
            modified. This is used by fine-grained Anchor V2.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor_cls = PredictorRegistry.get_by_name(predictor_name)
    corrector_cls = CorrectorRegistry.get_by_name(corrector_name)

    # Anchor V2 calibrates only the predictor score.  The corrector keeps the
    # original score function, preventing repeated V2 application when a
    # multi-step stochastic corrector is enabled.
    predictor_score_fn = score_fn
    if score_calibrator is not None:
        def predictor_score_fn(x, y_cond, t):
            score_base = score_fn(x, y_cond, t)
            return score_calibrator(x, y_cond, t, score_base)

    predictor = predictor_cls(sde, predictor_score_fn, probability_flow=probability_flow)
    corrector = corrector_cls(sde, score_fn, snr=snr, n_steps=corrector_steps)

    def pc_sampler():
        """The PC sampler function."""
        with torch.no_grad():
            regen_strength_value = _prepare_regen_strength(regeneration_strength, y)
            if regen_strength_value is None:
                xt = sde.prior_sampling(y.shape, y).to(y.device)
                t_start = float(sde.T)
            else:
                # r=1 => original full regeneration from T; r=0 => nearly identity/weak regeneration from eps.
                t_start = float(eps) + float(regen_strength_value.item()) * (float(sde.T) - float(eps))
                xt = _adaptive_prior_sampling(sde, y, t_start).to(y.device)
            timesteps = torch.linspace(t_start, eps, sde.N, device=y.device)
            for i in range(sde.N):
                t = timesteps[i]
                if i != len(timesteps) - 1:
                    stepsize = t - timesteps[i+1]
                else:
                    stepsize = timesteps[-1] # from eps to 0
                vec_t = torch.ones(y.shape[0], device=y.device) * t
                xt, xt_mean = corrector.update_fn(xt, y, vec_t)
                xt_before_predictor = xt
                xt, xt_mean = predictor.update_fn(xt, y, vec_t, stepsize)
                if anchor_calibrator is not None:
                    t_next = (vec_t - stepsize).clamp_min(0.0)
                    xt_mean_anchor = anchor_calibrator(xt_before_predictor, xt_mean, y, t_next)
                    # Keep the same stochastic part, but shift the transition mean.
                    mean_shift = xt_mean_anchor - xt_mean
                    xt = xt + mean_shift
                    xt_mean = xt_mean_anchor
            x_result = xt_mean if denoise else xt
            ns = sde.N * (corrector.n_steps + 1)
            return x_result, ns
    
    return pc_sampler


def get_ode_sampler(
    sde, score_fn, y, inverse_scaler=None,
    denoise=True, rtol=1e-5, atol=1e-5,
    method='RK45', eps=3e-2, device='cuda', regeneration_strength=None, **kwargs
):
    """Probability flow ODE sampler with the black-box ODE solver.

    Args:
        sde: An `sdes.SDE` object representing the forward SDE.
        score_fn: A function (typically learned model) that predicts the score.
        y: A `torch.Tensor`, representing the (non-white-)noisy starting point(s) to condition the prior on.
        inverse_scaler: The inverse data normalizer.
        denoise: If `True`, add one-step denoising to final samples.
        rtol: A `float` number. The relative tolerance level of the ODE solver.
        atol: A `float` number. The absolute tolerance level of the ODE solver.
        method: A `str`. The algorithm used for the black-box ODE solver.
            See the documentation of `scipy.integrate.solve_ivp`.
        eps: A `float` number. The reverse-time SDE/ODE will be integrated to `eps` for numerical stability.
        device: PyTorch device.

    Returns:
        A sampling function that returns samples and the number of function evaluations during sampling.
    """
    predictor = ReverseDiffusionPredictor(sde, score_fn, probability_flow=False)
    rsde = sde.reverse(score_fn, probability_flow=True)

    def denoise_update_fn(x):
        vec_eps = torch.ones(x.shape[0], device=x.device) * eps
        _, x = predictor.update_fn(x, y, vec_eps)
        return x

    def drift_fn(x, y, t):
        """Get the drift function of the reverse-time SDE."""
        return rsde.sde(x, y, t)[0]

    def ode_sampler(z=None, **kwargs):
        """The probability flow ODE sampler with black-box ODE solver.

        Args:
            model: A score model.
            z: If present, generate samples from latent code `z`.
        Returns:
            samples, number of function evaluations.
        """
        with torch.no_grad():
            # If regeneration_strength is provided, start from an adaptive diffusion time.
            regen_strength_value = _prepare_regen_strength(regeneration_strength, y)
            if regen_strength_value is None:
                x = sde.prior_sampling(y.shape, y).to(device)
                t_start = float(sde.T)
            else:
                t_start = float(eps) + float(regen_strength_value.item()) * (float(sde.T) - float(eps))
                x = _adaptive_prior_sampling(sde, y, t_start).to(device)

            def ode_func(t, x):
                x = from_flattened_numpy(x, y.shape).to(device).type(torch.complex64)
                vec_t = torch.ones(y.shape[0], device=x.device) * t
                drift = drift_fn(x, y, vec_t)
                return to_flattened_numpy(drift)

            # Black-box ODE solver for the probability flow ODE
            solution = integrate.solve_ivp(
                ode_func, (t_start, eps), to_flattened_numpy(x),
                rtol=rtol, atol=atol, method=method, **kwargs
            )
            nfe = solution.nfev
            x = torch.tensor(solution.y[:, -1]).reshape(y.shape).to(device).type(torch.complex64)

            # Denoising is equivalent to running one predictor step without adding noise
            if denoise:
                x = denoise_update_fn(x)

            if inverse_scaler is not None:
                x = inverse_scaler(x)
            return x, nfe

    return ode_sampler

def get_sb_sampler(sde, y, score_fn, eps=1e-4, sampler_type="ode", **kwargs):
    # print(f"score_fn:{score_fn.keywords}")
    # y_cond = score_fn.keywords.get("y_cond", "<not set>")
    # print("y_cond =", y_cond)
    # adapted from https://github.com/NVIDIA/NeMo/blob/78357ae99ff2cf9f179f53fbcb02c88a5a67defb/nemo/collections/audio/parts/submodules/schroedinger_bridge.py#L382
    def sde_sampler():
        """The SB-SDE sampler function.期望forward返回x0"""
        with torch.no_grad():
            xt = y[:, [0], :, :] # special case for storm_2ch
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN  重要改动 之前这里调用的model() ，导致y_cond无法传递！
                current_estimate = score_fn(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t**2 / (alpha_prev * sigma_prev**2 + sde.eps)
                tmp = 1 - sigma_t**2 / (sigma_prev**2 + sde.eps)
                weight_estimate = alpha_t * tmp
                weight_z = alpha_t * sigma_t * torch.sqrt(tmp)

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_z = weight_z[:, None, None, None]

                # Random sample
                z_norm = torch.randn_like(xt)
                
                if t == time_steps[-1]:
                    weight_z = 0.0

                # Update state: weighted sum of previous state, current estimate and noise
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_z * z_norm

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, sde.N + 1

    def ode_sampler():
        """The SB-ODE sampler function."""
        with torch.no_grad():
            xt = y
            time_steps = torch.linspace(sde.T, eps, sde.N + 1, device=y.device)

            # Initial values
            time_prev = time_steps[0] * torch.ones(xt.shape[0], device=xt.device)
            sigma_prev, sigma_T, sigma_bar_prev, alpha_prev, alpha_T, alpha_bar_prev = sde._sigmas_alphas(time_prev)

            for t in time_steps[1:]:
                # Prepare time steps for the whole batch
                time = t * torch.ones(xt.shape[0], device=xt.device)

                # Get noise schedule for current time
                sigma_t, sigma_T, sigma_bart, alpha_t, alpha_T, alpha_bart = sde._sigmas_alphas(time)

                # Run DNN
                current_estimate = score_fn(xt, y, time)

                # Calculate scaling for the first-order discretization from the paper
                weight_prev = alpha_t * sigma_t * sigma_bart / (alpha_prev * sigma_prev * sigma_bar_prev + sde.eps)
                weight_estimate = (
                    alpha_t
                    / (sigma_T**2 + sde.eps)
                    * (sigma_bart**2 - sigma_bar_prev * sigma_t * sigma_bart / (sigma_prev + sde.eps))
                )
                weight_prior_mean = (
                    alpha_t
                    / (alpha_T * sigma_T**2 + sde.eps)
                    * (sigma_t**2 - sigma_prev * sigma_t * sigma_bart / (sigma_bar_prev + sde.eps))
                )

                # View as [B, C, D, T]
                weight_prev = weight_prev[:, None, None, None]
                weight_estimate = weight_estimate[:, None, None, None]
                weight_prior_mean = weight_prior_mean[:, None, None, None]

                # Update state: weighted sum of previous state, current estimate and prior
                xt = weight_prev * xt + weight_estimate * current_estimate + weight_prior_mean * y

                # Save previous values
                time_prev = time
                alpha_prev = alpha_t
                sigma_prev = sigma_t
                sigma_bar_prev = sigma_bart

            return xt, sde.N + 1
    
    if sampler_type == "sde":
        return sde_sampler
    elif sampler_type == "ode":
        return ode_sampler
    else:
        raise ValueError("Invalid type. Choose 'ode' or 'sde'.")
