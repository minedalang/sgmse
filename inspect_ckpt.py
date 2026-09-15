import sys
import torch
from collections import OrderedDict


def print_obj(obj, name="root", indent=0, max_depth=10):
    prefix = "    " * indent

    if indent > max_depth:
        print(f"{prefix}{name}: <max depth>")
        return

    if torch.is_tensor(obj):
        print(
            f"{prefix}{name}: Tensor("
            f"shape={tuple(obj.shape)}, "
            f"dtype={obj.dtype}, "
            f"device={obj.device})"
        )
        return

    if isinstance(obj, (dict, OrderedDict)):
        print(
            f"{prefix}{name}: {type(obj).__name__} "
            f"({len(obj)} keys)"
        )

        for key, value in obj.items():
            print_obj(
                value,
                name=str(key),
                indent=indent + 1,
                max_depth=max_depth
            )
        return

    if isinstance(obj, (list, tuple)):
        print(
            f"{prefix}{name}: "
            f"{type(obj).__name__}({len(obj)})"
        )

        for i, value in enumerate(obj):
            print_obj(
                value,
                name=f"[{i}]",
                indent=indent + 1,
                max_depth=max_depth
            )
        return

    print(
        f"{prefix}{name}: "
        f"{type(obj).__name__} = {repr(obj)}"
    )


def main():
    if len(sys.argv) < 2:
        print("用法:")
        print("python inspect_ckpt.py xxx.ckpt")
        return

    ckpt_path = sys.argv[1]

    print("=" * 100)
    print("CKPT:", ckpt_path)
    print("=" * 100)

    # 仅对你自己可信的 ckpt 使用
    try:
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False
        )
    except TypeError:
        # 兼容旧版 PyTorch
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu"
        )

    print("\nCheckpoint Python 类型:")
    print(type(ckpt))

    print("\n" + "=" * 100)
    print("完整结构")
    print("=" * 100)

    print_obj(ckpt)

    print("\n" + "=" * 100)
    print("可能的模型权重")
    print("=" * 100)

    state_dict = None

    if isinstance(ckpt, dict):

        candidates = [
            "state_dict",
            "model_state_dict",
            "model",
            "net",
            "network",
            "weights",
            "ema_state_dict",
            "ema"
        ]

        for key in candidates:
            if key in ckpt:
                value = ckpt[key]

                if isinstance(value, (dict, OrderedDict)):
                    if any(torch.is_tensor(v) for v in value.values()):
                        print(f"\n发现可能的权重字段: {key}")
                        state_dict = value
                        break

        # 有些 ckpt 本身就是 state_dict
        if state_dict is None:
            if all(
                isinstance(k, str)
                for k in ckpt.keys()
            ) and any(
                torch.is_tensor(v)
                for v in ckpt.values()
            ):
                print("\nCheckpoint 本身可能就是 state_dict")
                state_dict = ckpt

    if state_dict is not None:

        print(f"\n参数数量: {len(state_dict)}")

        print("\n完整参数列表:")
        print("-" * 100)

        for key, value in state_dict.items():

            if torch.is_tensor(value):
                print(
                    f"{key:80s} "
                    f"shape={str(tuple(value.shape)):25s} "
                    f"dtype={value.dtype}"
                )
            else:
                print(
                    f"{key:80s} "
                    f"type={type(value).__name__}"
                )

    else:
        print("没有自动识别到 state_dict。")


if __name__ == "__main__":
    main()