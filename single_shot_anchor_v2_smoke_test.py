import torch

from sgmse.sampling import get_pc_sampler


class _DummyReverseSDE:
    def __init__(self, score_fn, n_steps):
        self.score_fn = score_fn
        self.N = n_steps

    def discretize(self, x, y, t, stepsize):
        # Calling the score function is what exercises sampler step metadata.
        _ = self.score_fn(x, y, t)
        f = torch.zeros_like(x)
        g = torch.zeros(x.shape[0], device=x.device, dtype=x.real.dtype)
        return f, g


class _DummySDE:
    T = 1.0

    def __init__(self, n_steps):
        self.N = n_steps

    def reverse(self, score_fn, probability_flow=False):
        return _DummyReverseSDE(score_fn, self.N)

    def prior_sampling(self, shape, y):
        return torch.zeros(shape, device=y.device, dtype=y.dtype)


def main():
    n_steps = 30
    seen_steps = []
    fired_steps = []
    target_1_based = 15

    def score_fn(x, y, t):
        return torch.zeros_like(x)

    def score_calibrator(x, y, t, score_base, step_index=None, num_steps=None):
        assert num_steps == n_steps
        seen_steps.append(int(step_index))
        if int(step_index) == target_1_based - 1:
            fired_steps.append(int(step_index))
        return score_base

    y = torch.zeros(1, 1, 4, 4, dtype=torch.complex64)
    sampler = get_pc_sampler(
        predictor_name="reverse_diffusion",
        corrector_name="none",
        sde=_DummySDE(n_steps),
        score_fn=score_fn,
        y=y,
        eps=0.03,
        score_calibrator=score_calibrator,
    )
    sample, nfe = sampler()

    assert sample.shape == y.shape
    assert seen_steps == list(range(n_steps)), seen_steps
    assert fired_steps == [target_1_based - 1], fired_steps
    assert nfe == n_steps

    # Verify ratio mapping used by Single-shot Anchor V2.
    ratio = 0.5
    target_from_ratio = int(round(ratio * (n_steps - 1)))
    assert target_from_ratio == 14  # 15th predictor step for N=30.

    # Verify the training-time sampling interval for center=0.5, width=0.1.
    T, t_eps = 1.0, 0.03
    center, width = 0.5, 0.1
    progress = torch.linspace(center - width, center + width, 101)
    t = T - progress * (T - t_eps)
    recovered_progress = (T - t) / (T - t_eps)
    assert torch.all(recovered_progress >= center - width - 1e-6)
    assert torch.all(recovered_progress <= center + width + 1e-6)

    print("Single-shot Anchor V2 smoke test passed.")
    print(f"N={n_steps}, explicit step={target_1_based}, ratio=0.5 -> step={target_from_ratio + 1}")


if __name__ == "__main__":
    main()
