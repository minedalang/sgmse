"""Pure tensor prototype, not wired to StoRM or a training objective."""
import math
import torch


def paired_objective(d, delta, mask, *, lower_margin, upper_margin,
                     upper_weight, orthogonal_weight, outside_weight,
                     min_delta_energy):
    """Shape [batch, ...], identical CPU shapes/dtypes; real soft mask.

    Sum coordinates per sample, average over valid samples only. Invalid
    samples (including their outside term) are explicitly skipped. Delta/mask
    are caller-supplied constants; no automatic detach hides gradient paths.
    A positive lower margin imposes amplitude, not only direction. No epsilon
    can manufacture a target from zero or sub-threshold delta.
    """
    if d.ndim < 2 or d.shape != delta.shape or d.shape != mask.shape:
        raise ValueError('expected identical [batch, ...] shapes')
    if not d.is_floating_point() and not d.is_complex():
        raise ValueError('floating or complex tensors required')
    if d.dtype != delta.dtype or mask.dtype != d.real.dtype:
        raise ValueError('matching d/delta dtype and real mask dtype required')
    if any(x.device.type != 'cpu' for x in (d, delta, mask)):
        raise ValueError('prototype is CPU only')
    if not all(bool(torch.isfinite(x).all()) for x in (d, delta, mask)):
        raise ValueError('non-finite input')
    if not bool(((mask >= 0) & (mask <= 1)).all()):
        raise ValueError('mask must lie in [0,1]')
    params = [lower_margin, upper_margin, upper_weight, orthogonal_weight,
              outside_weight, min_delta_energy]
    if not all(math.isfinite(float(x)) for x in params):
        raise ValueError('non-finite parameter')
    if lower_margin < 0 or upper_margin < lower_margin or min_delta_energy < 0:
        raise ValueError('invalid margins or energy threshold')
    if min(upper_weight, orthogonal_weight, outside_weight) < 0:
        raise ValueError('negative weight')
    dims = tuple(range(1, d.ndim))
    pd, pt = mask * d, mask * delta
    norm2 = lambda x: (x.conj() * x).real.sum(dim=dims)
    energy = norm2(pt)
    valid = energy > min_delta_energy
    denominator = torch.where(valid, energy, torch.ones_like(energy))
    g_raw = (pd.conj() * pt).real.sum(dim=dims) / denominator
    g = torch.where(valid, g_raw, torch.zeros_like(g_raw))
    expand = g.reshape((-1,) + (1,) * (d.ndim - 1))
    lower = torch.relu(lower_margin - g).square()
    upper = torch.relu(g - upper_margin).square()
    orthogonal = norm2(pd - expand * pt) / denominator
    outside = norm2((1 - mask) * d)
    zero_invalid = lambda x: torch.where(valid, x, torch.zeros_like(x))
    lower, upper, orthogonal, outside = map(zero_invalid,
                                          (lower, upper, orthogonal, outside))
    per_sample = (lower + upper_weight * upper +
                  orthogonal_weight * orthogonal + outside_weight * outside)
    total = per_sample.sum() / valid.sum().clamp_min(1)
    return dict(total=total, per_sample=per_sample, gain=g, lower=lower,
                upper=upper, orthogonal=orthogonal, outside=outside,
                delta_energy=energy, valid=valid, skipped=(~valid).sum())


def clean_readout(q, u, anchor, a, sigma):
    """OUVE D=(u+sigma^2 q-(1-a)A)/a, scalar a>0,sigma>0.

    Caller MUST detach shared state/anchor before use when desired. Parameter
    freezing is not no_grad: trainable Score operations must retain autograd.
    """
    if q.shape != u.shape or q.shape != anchor.shape:
        raise ValueError('identical q/u/anchor shapes required')
    if any(x.device.type != 'cpu' for x in (q, u, anchor)):
        raise ValueError('prototype is CPU only')
    if not (math.isfinite(float(a)) and math.isfinite(float(sigma)) and
            float(a) > 0 and float(sigma) > 0):
        raise ValueError('positive finite scalar a/sigma required')
    return (u + sigma ** 2 * q - (1 - a) * anchor) / a
