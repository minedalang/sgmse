# CAM++ standalone / WeSpeaker-free integration

This patch assumes you already applied the prior **SGMSE_CAMPlus_适配最新代码.patch**,
which added `sgmse/speaker_encoders.py` and the `--plm_type campplus` switch.
It replaces the WeSpeaker import with an in-project CAM++ architecture.
There is **no dependency on importing or installing** `wespeaker`, `s3prl`,
`espnet`, or on changing your existing Torch/TorchAudio versions.

## Apply

```bash
cd /data/coding/sgmse
git apply --check SGMSE_CAMPlus_完全独立版.patch
git apply SGMSE_CAMPlus_完全独立版.patch
```

## Test

```bash
python -c "from sgmse.campplus_standalone import CAMPPlus; print('CAM++ OK')"
python -m scripts.check_campplus_grad /data/models/wespeaker-voxceleb-campplus
```

`/data/models/wespeaker-voxceleb-campplus` needs the original `avg_model.pt`.
This is an official *weight file*, not a `wespeaker` Python dependency. A
local checkpoint avoids any Hub network requirement.

## Training flags (same as prior adapter)

```bash
--plm_type campplus \
--plm_name_or_path /data/models/wespeaker-voxceleb-campplus \
--contrastive_weight 10 \
--contrastive_temperature 0.1 \
--contrastive_clean_weight 1.0 \
--contrastive_speaker_weight 0.5 \
--max_plm_length 32000
```

Start a **new experiment initialized from your base 5ul7u03z checkpoint**;
do not resume an optimizer checkpoint trained with UniSpeech-SAT.

## Compatibility and scope

- Architecture is based on the Apache-2.0-licensed WeSpeaker CAM++ design,
  topology/state_dict names follow the upstream CAMPlus class:
  https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/campplus.py
- TSTP statistical pooling is adapted from the upstream Apache-2.0 code:
  https://github.com/wenet-e2e/wespeaker/blob/master/wespeaker/models/pooling_layers.py
- License: https://github.com/wenet-e2e/wespeaker/blob/master/LICENSE
- The encoder uses `torchaudio.compliance.kaldi.fbank`, but **not** SoX or
  `torchaudio.sox_effects`. It uses 16 kHz waveforms and 80-bin fbank.
- The parameters of CAM++ are frozen; gradients are preserved through the
  enhanced audio path. Full integration is not guaranteed until running the
  real-weight gradient script on your machine.
- If the official pretrained `avg_model.pt` uses a different checkpoint
  schema, the adapter intentionally raises an error rather than silently
  training with random encoder weights.
- We deliberately do not alter `model.py`, RRSD/Future-grad, or the existing
  UniSpeech-SAT checkpointing code.
