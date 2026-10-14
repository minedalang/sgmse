"""Frozen SpeechBrain VoxCeleb X-Vector encoder for differentiable SGMSE losses.

Unlike SpeechBrain's high-level file classification interface, this adapter
calls the *batched, differentiable* Fbank -> sentence normalization -> TDNN
pipeline directly. No detach/no_grad is used for the enhanced-audio branch.

Upstream pretrained model: speechbrain/spkrec-xvect-voxceleb (Apache-2.0).
"""

from pathlib import Path
import torch
from torch import nn


XVECTOR_REPO = "speechbrain/spkrec-xvect-voxceleb"


class SpeechBrainXVectorEncoder(nn.Module):
    """16-kHz waveform [B,L] -> pretrained 512-D embedding [B,512].

    `model_name_or_path` may be the Hub repository or a local directory
    containing the SpeechBrain model's hyperparams.yaml and checkpoint files.
    Uses the same 24-bin Fbank / sentence-CMN as the official model.
    """

    def __init__(self, model_name_or_path=XVECTOR_REPO, sample_rate=16000):
        super().__init__()
        if int(sample_rate) != 16000:
            raise ValueError("SpeechBrain VoxCeleb X-Vector requires 16-kHz audio")
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as exc:
            raise ImportError(
                "Install SpeechBrain: python -m pip install speechbrain; "
                "if installation attempts to change PyTorch, use --no-deps "
                "and ensure SpeechBrain's other dependencies are installed."
            ) from exc

        source = str(model_name_or_path)
        if Path(source).expanduser().is_absolute() and not Path(source).expanduser().is_dir():
            raise FileNotFoundError(
                f"X-Vector local model directory does not exist: {source}. "
                "Download 'speechbrain/spkrec-xvect-voxceleb' to it first."
            )
        local = Path(source).expanduser().is_dir()
        if local:
            source = str(Path(source).expanduser().resolve())
            if not (Path(source) / "hyperparams.yaml").is_file():
                raise FileNotFoundError(f"Missing {source}/hyperparams.yaml")
        try:
            # Load once on CPU. Lightning/model.to(device) then moves all
            # registered modules to the trainer device.
            # The official hyperparams.yaml contains a *remote* pretrained_path.
            # Override it for fully offline loading from a locally downloaded
            # repository; otherwise SpeechBrain might still contact the Hub.
            loader_args = {"source": source, "run_opts": {"device": "cpu"}}
            if local:
                loader_args["overrides"] = {"pretrained_path": source}
            self.classifier = EncoderClassifier.from_hparams(**loader_args)
        except Exception as exc:
            raise RuntimeError(
                f"Could not load SpeechBrain X-Vector from {source!r}. "
                "For offline use, download all files of "
                "speechbrain/spkrec-xvect-voxceleb and pass the local directory."
            ) from exc

        modules = self.classifier.mods
        for name in ("compute_features", "mean_var_norm", "embedding_model"):
            if name not in modules:
                raise RuntimeError(f"SpeechBrain X-Vector checkpoint lacks '{name}'")
        # Neither the 7205-class verification head nor its output normalizer
        # is required for 512-D embeddings; discard them after loading.
        for unused in ("classifier", "mean_var_norm_emb"):
            if unused in modules:
                del modules[unused]
        for param in self.classifier.parameters():
            param.requires_grad_(False)
        self.sample_rate = 16000
        self.embedding_dim = 512
        self.eval()
        print(f"[X-Vector] Frozen SpeechBrain weights loaded: {source}")

    def train(self, mode=True):
        # Never update the pretrained TDNN or normalization statistics.
        super().train(False)
        return self

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim != 2:
            raise ValueError(f"Expected waveform [B,L], got {tuple(waveform.shape)}")
        if waveform.size(1) < 1600:
            raise ValueError("X-Vector needs at least 0.1 s of 16-kHz audio")
        modules = self.classifier.mods
        # SpeechBrain's Fbank is vectorized in the batch dimension, unlike the
        # one-utterance-at-a-time torchaudio.compliance.kaldi.fbank adapter.
        # FP32 features match the pretrained network's frontend and preserve
        # the gradient w.r.t. the enhanced waveform.
        with torch.autocast(device_type=waveform.device.type, enabled=False):
            feats = modules.compute_features(waveform.float())
            lengths = torch.ones(waveform.shape[0], device=feats.device, dtype=feats.dtype)
            feats = modules.mean_var_norm(feats, lengths)
        embedding = modules.embedding_model(feats, lengths)
        if embedding.ndim == 3 and embedding.size(1) == 1:
            embedding = embedding[:, 0, :]
        if embedding.ndim != 2 or embedding.size(0) != waveform.size(0):
            raise RuntimeError(f"Unexpected X-Vector embedding shape: {tuple(embedding.shape)}")
        if embedding.size(1) != 512:
            raise RuntimeError(f"Expected 512-dimensional X-Vectors, got {embedding.size(1)}")
        return embedding
