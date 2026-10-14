"""Fair speaker-encoder speed comparison; excludes SGMSE diffusion network.

Example:
python -m scripts.benchmark_speaker_encoders --plm_type xvector \
  --model speechbrain/spkrec-xvect-voxceleb --batch-size 16 --length 32000
"""
import argparse
import statistics
import time
import torch
import torch.nn.functional as F


def make_model(kind, name):
    if kind == "xvector":
        from sgmse.xvector_encoder import SpeechBrainXVectorEncoder
        return SpeechBrainXVectorEncoder(name)
    if kind == "campplus":
        from sgmse.speaker_encoders import WeSpeakerCAMPPlusEncoder
        return WeSpeakerCAMPPlusEncoder(name)
    from transformers import UniSpeechSatModel
    model = UniSpeechSatModel.from_pretrained(name)
    class MeanHidden(torch.nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone
        def forward(self, wav):
            return self.backbone(input_values=wav, return_dict=True).last_hidden_state.mean(1)
    return MeanHidden(model)


def main():
    parser = argparse.ArgumentParser(description="Benchmark 16kHz speaker encoders (not enhancement inference)")
    parser.add_argument("--plm_type", choices=["xvector", "campplus", "unispeech_sat"], default="xvector")
    parser.add_argument("--model", default=None, help="Pretrained Hub ID or local folder")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--length", type=int, default=32000, help="Samples per utterance, at 16 kHz")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["off", "bf16"], default="off")
    args = parser.parse_args()
    if args.batch_size < 1 or args.steps < 1 or args.warmup < 0:
        parser.error("batch-size and steps must be positive; warmup must be nonnegative")
    if args.device.startswith("cpu") and args.amp == "bf16":
        parser.error("For a reproducible comparison, use --amp off on CPU")

    names = {
        "xvector": "speechbrain/spkrec-xvect-voxceleb",
        "campplus": "Wespeaker/wespeaker-voxceleb-campplus",
        "unispeech_sat": "microsoft/unispeech-sat-base-plus-sv",
    }
    source = args.model or names[args.plm_type]
    print("Loading:", args.plm_type, source, flush=True)
    model = make_model(args.plm_type, source).to(args.device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    device = torch.device(args.device)
    use_cuda = device.type == "cuda"
    if use_cuda:
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    wav = torch.randn(args.batch_size, args.length, device=device)
    clean = torch.randn(args.batch_size, args.length, device=device)

    def sync():
        if use_cuda:
            torch.cuda.synchronize(device)

    def measure_one():
        # Inference-only: one batched forward, no gradients (typical X-Vector inference).
        sync()
        t0 = time.perf_counter()
        with torch.inference_mode(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.amp == "bf16")):
            inference_embedding = model(wav)
        sync()
        t1 = time.perf_counter()
        shape = tuple(inference_embedding.shape)
        del inference_embedding

        # Training-like: enhanced forward with autograd, clean forward with no
        # grad, then backward to the enhanced input; pretrained weights frozen.
        enhanced = wav.detach().requires_grad_(True)
        sync()
        t2 = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.amp == "bf16")):
            emb_enh = model(enhanced)
        sync()
        t3 = time.perf_counter()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=(args.amp == "bf16")):
            emb_clean = model(clean)
        sync()
        t4 = time.perf_counter()
        loss = F.mse_loss(emb_enh.float(), emb_clean.float())
        loss.backward()
        sync()
        t5 = time.perf_counter()
        assert enhanced.grad is not None and torch.isfinite(enhanced.grad).all()
        del enhanced, emb_enh, emb_clean, loss
        return ((t1-t0)*1000, (t3-t2)*1000, (t4-t3)*1000, (t5-t4)*1000), shape

    records = []
    for i in range(args.warmup + args.steps):
        ms, shape = measure_one()
        if i >= args.warmup:
            records.append(ms)
            print(f"iter {i-args.warmup+1:02d}: infer={ms[0]:.1f}ms, enh_fw={ms[1]:.1f}ms, "
                  f"clean_fw={ms[2]:.1f}ms, backward={ms[3]:.1f}ms, train_sum={sum(ms[1:]):.1f}ms")

    def avg(index):
        return statistics.mean(x[index] for x in records)

    inference = [x[0] for x in records]
    train = [sum(x[1:]) for x in records]
    print("\n========== SUMMARY ==========")
    print(f"encoder={args.plm_type}, embedding_shape={shape}")
    print(f"device={args.device}, B={args.batch_size}, L={args.length}, amp={args.amp}")
    print(f"INFERENCE 1 forward: mean={statistics.mean(inference):.1f} ms, median={statistics.median(inference):.1f} ms")
    print(f"Enhance forward:      mean={avg(1):.1f} ms")
    print(f"Clean forward:        mean={avg(2):.1f} ms")
    print(f"Backward to waveform: mean={avg(3):.1f} ms")
    print(f"TRAIN 2fw+1bw:       mean={statistics.mean(train):.1f} ms, median={statistics.median(train):.1f} ms")
    print(f"INFER throughput:     {args.batch_size * 1000 / statistics.mean(inference):.1f} utterances/s")
    if use_cuda:
        print(f"Peak allocated GPU memory: {torch.cuda.max_memory_allocated(device)/(1024**3):.2f} GiB")
    print("Note: excludes STORM, diffusion, wave reconstruction, DataLoader and InfoNCE.")


if __name__ == "__main__":
    main()
