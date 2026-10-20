"""Small helpers shared by joint training code and focused unit tests."""


def rrsd_aux_loss_enabled(router, weight: float, disabled: bool) -> bool:
    """Whether the training-only RRSD auxiliary objective should run."""
    return router is not None and float(weight) > 0.0 and not bool(disabled)
