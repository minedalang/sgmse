"""Tensor-only CPEL losses; no model, checkpoint, audio or runtime side effects."""
import torch


def native_dsm(score, epsilon, sigma):
    return 0.5 * (sigma * score + epsilon).abs().square().mean()


def common_point(score, state, clean, anchor, alpha, sigma):
    # Auxiliary regression at common inputs, NOT an unbiased native DSM draw.
    residual = sigma * score + (state - alpha * clean - (1-alpha) * anchor) / sigma
    return 0.5 * residual.abs().square().mean()


def pair_loss(q1, q0, delta, projection, alpha, sigma):
    """Single-pair energy-scaled whitened constraints, with soft temporal P.

    D is the full tensor number of complex coefficients; norms are sums. Soft
    P is not an orthogonal projection. Invalid delta skips every pair term.
    """
    pdelta = projection * delta
    valid = bool((pdelta.abs().square().mean().detach() > 1e-12).item())
    zero = (q1.real.sum() + q0.real.sum()) * 0.0
    if not valid:
        return zero, dict(valid=False, lower=0., upper=0., orth=0., outside=0., response=0.)
    e = (alpha/sigma) * pdelta
    radius = e.abs().square().sum().sqrt()
    # valid input and positive finite t ensure radius>0; no NaN branch evaluation.
    if not bool(torch.isfinite(radius).item()) or float(radius.detach()) <= 0:
        raise ValueError('Non-finite or zero whitened radius for valid pair')
    v = e / radius
    w = sigma * projection * (q1-q0)
    b = (w * v.conj()).real.sum()
    d = delta.numel()
    lower = torch.relu(.1*radius-b).square()/d
    upper = torch.relu(b-1.5*radius).square()/d
    orth = (w-b*v).abs().square().sum()/d
    outside = (sigma*(1-projection)*(q1-q0)).abs().square().sum()/d
    loss = lower + upper + .25*orth + .2*outside
    stats = dict(valid=True, response=float((b/radius).detach()),
                 lower=float(lower.detach()), upper=float(upper.detach()),
                 orth=float(orth.detach()), outside=float(outside.detach()))
    return loss, stats
