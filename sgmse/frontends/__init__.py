from .registry import FrontendRegistry
from .base import BaseFrontend, match_length
from .semamba import SemambaFrontend
from .storm import StormFrontend
from .tfgridnet import TFGridNetFrontend


@FrontendRegistry.register("semamba")
class _RegisteredSemambaFrontend(SemambaFrontend):
    pass


def build_frontend(frontend_name: str, **kwargs):
    frontend_cls = FrontendRegistry.get_by_name(frontend_name)
    return frontend_cls(**kwargs)


__all__ = [
    "BaseFrontend",
    "FrontendRegistry",
    "SemambaFrontend",
    "StormFrontend",
    "TFGridNetFrontend",
    "build_frontend",
    "match_length",
]
