
import time
import torch
from transformers import UniSpeechSatModel

device = "cuda:0"
model_name = "microsoft/unispeech-sat-base-plus-sv"

print("Loading UniSpeech-SAT-SV...")

model = UniSpeechSatModel.from_pretrained(
    model_name,
    local_files_only=True
).to(device).eval()

# 冻结编码器参数，但保留输入梯度
for p in model.parameters():
    p.requires_grad_(False)

B = 16
L = 32000
warmup = 2
steps = 8

forward_times = []
backward_times = []
total_times = []

print("Device:", torch.cuda.get_device_name(0))
print("Batch size:", B)
print("Audio length:", L)
print("Parameters:", sum(p.numel() for p in model.parameters()))

torch.cuda.reset_peak_memory_stats()

for step in range(steps):

    wav = torch.randn(
        B, L,
        device=device,
        requires_grad=True
    )

    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # UniSpeech-SAT 编码器前向传播
    output = model(
        input_values=wav,
        return_dict=True
    )

    emb = output.last_hidden_state.mean(dim=1)

    torch.cuda.synchronize()
    t1 = time.perf_counter()

    # 反向传播至输入语音
    loss = emb.square().mean()
    loss.backward()

    torch.cuda.synchronize()
    t2 = time.perf_counter()

    forward_ms = (t1 - t0) * 1000
    backward_ms = (t2 - t1) * 1000
    total_ms = (t2 - t0) * 1000

    if step >= warmup:
        forward_times.append(forward_ms)
        backward_times.append(backward_ms)
        total_times.append(total_ms)

        print(
            f"Step {step:02d} | "
            f"Forward: {forward_ms:.1f} ms | "
            f"Backward: {backward_ms:.1f} ms | "
            f"Total: {total_ms:.1f} ms"
        )

    del wav, output, emb, loss

print("\n========== Average Time ==========")

print(
    f"Forward: {sum(forward_times)/len(forward_times):.1f} ms"
)
print(
    f"Backward: {sum(backward_times)/len(backward_times):.1f} ms"
)
print(
    f"Total: {sum(total_times)/len(total_times):.1f} ms"
)

peak_mem = torch.cuda.max_memory_allocated() / 1024**3

print(f"Peak GPU allocated memory: {peak_mem:.2f} GB")