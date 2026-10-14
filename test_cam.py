import time
import torch
from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder

device = "cuda"

model = WeSpeakerCAMPPlusEncoder(
    "/data/coding/sgmse/pre_models/wespeaker-voxceleb-campplus"
).to(device).eval()

B = 16
L = 32000

for step in range(8):
    wav = torch.randn(
        B, L,
        device=device,
        requires_grad=True
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # 1. FBank
    feat = model._differentiable_fbank(wav)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # 2. CAM++ 网络
    emb = model.model(feat)

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    # 3. 梯度回传
    loss = emb.square().mean()
    loss.backward()

    torch.cuda.synchronize()
    t3 = time.perf_counter()

    if step >= 2:
        print(
            f"FBank: {(t1-t0)*1000:.1f} ms | "
            f"CAM++: {(t2-t1)*1000:.1f} ms | "
            f"Backward: {(t3-t2)*1000:.1f} ms"
        )