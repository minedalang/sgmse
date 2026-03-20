from __future__ import annotations

from .external import ExternalWaveformFrontend
from .registry import FrontendRegistry


@FrontendRegistry.register("storm")
class StormFrontend(ExternalWaveformFrontend):
    """StoRM 前端适配器。

    注意：官方 StoRM 仓库也使用 `sgmse` 作为顶层包名，和当前项目同名。
    因此不建议在当前进程里直接默认 import 官方仓库的 `sgmse.model`，
    否则很容易和你自己的项目包发生命名冲突。

    这版保留通用外部前端接口：
    - 若你已经把 StoRM 的前端模型单独封装成一个独立 module/class，
      这里可以直接加载；
    - 若你后续决定把 StoRM 的 denoiser 阶段 vendoring 到当前项目里，
      也可以继续复用这个接口。
    """

    default_module_name = ""
    default_class_name = ""
