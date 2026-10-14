"""Frozen speaker encoders for differentiable enhancement-training losses.

WeSpeaker CAM++ uses 80-bin Kaldi FBanks and produces a 512-D speaker
embedding.  Crucially, this wrapper does NOT use wespeaker's CLI
extract_embedding() method: that inference helper detaches tensors and would
prevent contrastive gradients from reaching the speech enhancement network.
"""
from pathlib import Path
 
import torch
from torch import nn
from torchaudio.compliance import kaldi


CAMPLUS_REPO = "Wespeaker/wespeaker-voxceleb-campplus"


def _resolve_campplus_weights(model_name_or_path: str) -> Path:
    candidate = Path(model_name_or_path).expanduser()
    if candidate.is_dir():
        checkpoint = candidate / "avg_model.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"CAM++ requires avg_model.pt under {candidate}")
        return checkpoint
    if candidate.is_file():
        return candidate
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise ImportError("Install huggingface_hub or pass a local CAM++ checkpoint path") from exc
    try:
        return Path(hf_hub_download(repo_id=model_name_or_path, filename="avg_model.pt"))
    except Exception as exc:
        raise RuntimeError(
            f"Could not download CAM++ checkpoint from {model_name_or_path!r}. "
            "On an offline training machine, pre-download avg_model.pt and "
            "pass its directory or file via --plm_name_or_path."
        ) from exc


class WeSpeakerCAMPPlusEncoder(nn.Module):
    """Differentiable 16-kHz waveform -> CAM++ speaker embedding adapter.

    Args:
      model_name_or_path: Hugging Face repo, local directory containing
        avg_model.pt, or a direct path to avg_model.pt.
      sample_rate: must be 16000 Hz (the published VoxCeleb model setting).

    A gradient is tracked through the *enhanced* waveform; the pretrained
    CAM++ weights are frozen.  Kaldi FBanks have deterministic dither=0 and
    utterance cepstral mean normalization (CMN), matching the WeSpeaker
    evaluation pipeline's 80-mel/25-ms/10-ms feature conventions.
    """

    def __init__(self, model_name_or_path=CAMPLUS_REPO, sample_rate=16000):
        super().__init__()
        if int(sample_rate) != 16000:
            raise ValueError("WeSpeaker CAM++ pretrained speaker model requires 16 kHz input")
        from sgmse.campplus_standalone import CAMPPlus

        self.model = CAMPPlus(feat_dim=80, embed_dim=512, pooling_func="TSTP")
        weight_path = _resolve_campplus_weights(model_name_or_path)
        # The official avg_model.pt is a tensor state dictionary; weights_only
        # avoids unpickling non-tensor objects from untrusted files.
        checkpoint = torch.load(str(weight_path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]
        if not isinstance(checkpoint, dict):
            raise TypeError(f"CAM++ checkpoint at {weight_path} is not a state dictionary")
        missing, unexpected = self.model.load_state_dict(checkpoint, strict=False)
        # A classification projection head is allowed in training checkpoints;
        # missing CAM++ backbone tensors must NOT silently become random weights.
        backbone_missing = [k for k in missing if not k.endswith("num_batches_tracked")]
        if backbone_missing:
            raise RuntimeError(
                f"CAM++ backbone has {len(backbone_missing)} missing tensors. "
                f"First missing keys: {backbone_missing[:8]}"
            )
        bad_unexpected = [k for k in unexpected if not k.startswith("projection.")]
        if bad_unexpected:
            raise RuntimeError(f"Unexpected CAM++ checkpoint keys: {bad_unexpected[:8]}")

        for param in self.model.parameters():
            param.requires_grad_(False)
        self.eval()
        self.sample_rate = 16000
        self.embedding_dim = 512
        print(f"[CAM++] Loaded frozen WeSpeaker weights: {weight_path}")

    def train(self, mode=True):
        super().train(False)
        return self

    @staticmethod
    def _differentiable_fbank(waveform: torch.Tensor) -> torch.Tensor:
        """Waveform [B,L] -> fbank [B,T,80], with input autograd preserved."""
        if waveform.ndim != 2:
            raise ValueError(f"Expected [B, L] waveform, got {tuple(waveform.shape)}")
        if waveform.size(-1) < 400:
            raise ValueError("CAM++ needs audio longer than 25 ms at 16 kHz")
        # torchaudio.compliance.kaldi.fbank expects [channels, samples] per
        # utterance. Operations are PyTorch tensors, NOT NumPy or no_grad.
        # Disable mixed precision for feature extraction to avoid FP16/BF16 FFT
        # accuracy issues; a cast to float32 still retains input gradients.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            waveform = waveform.float()
            fbanks = []
            for wav in waveform:
                mel = kaldi.fbank(
                    wav.unsqueeze(0),
                    sample_frequency=16000,
                    num_mel_bins=80,
                    frame_length=25.0,
                    frame_shift=10.0,
                    window_type="povey",
                    dither=0.0,
                    use_energy=False,
                )
                fbanks.append(mel - mel.mean(dim=0, keepdim=True))
            return torch.stack(fbanks, dim=0)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        feat = self._differentiable_fbank(waveform)
        embeddings = self.model(feat)
        if isinstance(embeddings, (tuple, list)):
            embeddings = embeddings[-1]
        if embeddings.ndim != 2 or embeddings.shape[0] != waveform.shape[0]:
            raise RuntimeError(f"Unexpected CAM++ embedding shape: {tuple(embeddings.shape)}")
        return embeddings
