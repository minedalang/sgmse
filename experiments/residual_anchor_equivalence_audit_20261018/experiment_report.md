# Residual-Anchored OUVE equivalence audit — 2026-10-18

## Question

The repository contains a `residual_refine_enable` branch with a time-dependent learned residual anchor. This audit checks whether it is a new stochastic process or a reparameterization of the existing OUVE process.

## Code evidence

`sgmse/sdes.py` implements, for a fixed conditional tensor `y`,

```text
dX = [-theta * X + a(t) * y] dt + g(t) dW
a(t) = theta * lambda(t) + (1 - exp(-theta*t)) * lambda'(t)
```

and its marginal mean is

```text
m(t) = alpha(t) * x0 + beta(t) * y
alpha(t) = exp(-theta*t)
beta(t) = (1-alpha(t)) * lambda(t).
```

Since `beta'(t) + theta*beta(t) = a(t)`, the translated state

```text
U(t) = X(t) - beta(t) * y
```

satisfies

```text
dU = -theta * U dt + g(t) dW.
```

The diffusion coefficient `g(t)` is unchanged. `prior_sampling` and the adaptive sampler only translate the initial center by `lambda(T) * y`; `sampling/__init__.py` does not add a new noise law.

## Interpretation

The branch can still change the learned conditional representation, the score network's optimization problem, and the practical finite-capacity behavior. Those are empirical questions. The SDE itself is a translated/reparameterized OUVE process, so “new SDE” or “new stochastic dynamics” would be inaccurate. The constant schedule with strength 1 recovers the historical direct residual anchor; power and smoothstep schedules change the coordinate trajectory but not this equivalence.

## Decision

Do not present Residual-Anchored OUVE as an independent innovation or start a training run solely because the equations look new. It may be revisited only as a representation/conditioning ablation after a separate, falsifiable headroom signal and a matched no-loss control are available.

No dataset, checkpoint, source code, training state, or audio was modified.
