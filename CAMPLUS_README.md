# CAM++ encoder option for SGMSE joint contrastive training

This extension adds **`--plm_type campplus`** while preserving the original
`--plm_type unispeech_sat` default. The InfoNCE loss, positive/negative
sampling logic, and `--contrastive_speaker_weight` are unchanged.
This patch is rebased against the latest user-uploaded SGMSE code and retains
`SGMSE_PLM_CHECKPOINT` support for UniSpeech-SAT and CAM++.

## 1. Install in your SGMSE training environment

```bash
python -m pip install wespeaker huggingface_hub
```

`wespeaker` must expose `wespeaker.models.campplus.CAMPPlus`. If your package
release does not expose this class, install the current official source:

```bash
python -m pip install 'git+https://github.com/wenet-e2e/wespeaker.git'
```

The CAM++ checkpoint can be downloaded on any machine with internet access.
For offline training servers, copy `avg_model.pt` into a local directory, e.g.
`/data/models/wespeaker-voxceleb-campplus/`:

```bash
huggingface-cli download Wespeaker/wespeaker-voxceleb-campplus avg_model.pt --local-dir /data/models/wespeaker-voxceleb-campplus
```

Alternatively, `--plm_name_or_path Wespeaker/wespeaker-voxceleb-campplus`
will try to fetch the weights from Hugging Face on demand.

The official WeSpeaker CAM++ model is fed **80-dimensional Kaldi FBank**
(25-ms windows, 10-ms hop, 16 kHz) and produces a **512-dimensional**
speaker embedding. We use `dither=0`, waveform-wise CMN, and a PyTorch
feature pipeline so gradients propagate to enhanced waveforms. This is a
training-time adapter, not a byte-identical reproduction of the offline
WeSpeaker embedding extraction pipeline (which may use different waveform
normalization and dither settings).

## 2. Minimal change to your existing command

Keep your other arguments unchanged. In the existing contrastive command,
replace/add:

```bash
--plm_type campplus \
--plm_name_or_path /data/models/wespeaker-voxceleb-campplus \
--contrastive_weight 10 \
--contrastive_temperature 0.1 \
--contrastive_clean_weight 1.0 \
--contrastive_speaker_weight 0.5 \
--max_plm_length 32000
```

To run paired-only InfoNCE instead: `--contrastive_speaker_weight 0`.
For the original large encoder: `--plm_type unispeech_sat --plm_name_or_path microsoft/unispeech-sat-base-plus-sv`.

For a controlled comparison, initialize both experiments from the same
`5ul7u03z` 300k checkpoint, with **the same optimizer/training config**.
Do not use `--ckpt` to resume an optimizer checkpoint from a different
encoder architecture; use `--init_ckpt` to start a fresh comparison.

## 3. Verify before a long training run

```bash
python scripts/check_campplus_grad.py /data/models/wespeaker-voxceleb-campplus
python -m unittest discover -s tests -p 'test_campplus_adapter.py'
```

The first script loads the actual weights and confirms enhanced-audio
input gradients; the unit test uses tiny synthetic weights and does not
require downloading CAM++.

During training, check `contrastive_loss_step` is nonzero and that the
enhancement model has nonzero gradients. The frozen CAM++ parameters should
have `requires_grad=False` and no parameter gradients.

## 4. Important implementation notes

- The CAM++ adapter is deliberately imported **only** when enabled; the
  original UniSpeech-SAT path is still available.
- A missing CAM++ package, incompatible checkpoint or download failure
  raises an error instead of silently disabling the contrastive loss.
- In this latest ZIP, `self.plm_model = None` is already correctly indented
  inside the exception handler; this patch preserves that fixed behavior.
- CAM++ is used only to calculate the **training loss**; it is not needed
  for the normal enhancement forward path. Some inference checkpoints may
  still construct the encoder at model initialization, depending on the
  saved hyperparameters and inference script; test deployment independently.
- Speaker-weighted contrastive learning only uses additional positives if
  same-speaker examples and valid speaker IDs occur in a mini-batch.
