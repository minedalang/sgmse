CPU preparation implementation is ready for Root review; actual input preparation has not run.

Read-only source/header/hash checks: four prescribed official-derived clean files and both historical full30s480000-sample PCM16 noise files match prior locks. Thread provider codex_tools_compat; bridge running; dirty worktree preserved; ~1.9GB free; no model process found. No repeated completed experiment.

Driver uses standard library only; actual audio sample reads occur only within supervised Worker execution. It writes 16 noisy WAVs, four scaled-clean WAVs and20 little-endian float32 raw arrays with explicit shape/dtype and hashes. Four conditions share each utterance scale. Registry retains every coefficient, source/text hash, component/float32/quantized SNR, quantizer/residual guard, native basename seed. Inputs explicitly floor float64 to int16 before WAV writing; no saturation. Historical decoded noise float64 hash is checked as well as full source bytes.

Supervisor enforces single exclusive INPUT_PREPARATION_STARTED marker,118s execution+2s reserve,12MB directory budget and preserves failure/partial artifacts. Child blocks network sockets/DNS and subprocesses, and rejects model/numeric audio imports. Completed preparation uses input_metrics.json and input_preparation_report.md; implementation metadata cannot be mistaken for an execution. No final model execution manifest is created.

Synthetic-only tests passed: four mock PCM roundtrips; order and whole-array SNR; common downward scale and no upward scaling; floor for negative half-LSB; rejection of zero/nonfinite power, mismatched length and saturation; actual socket/DNS and torch/whisper import guards. Initial repeated-key syntax error was fixed before passing tests. No real sample decode, resample, model, network, data synthesis, training or source edits. Independent validator is statically compiled only and will reconstruct actual20 artifacts after Worker preparation, checking hashes, byte-exact PCM/raw arrays, full recipe/order/seeds.

Worker CPU command after Root approval:
python3 -B /data/coding/sgmse/experiments/p8_voicebank_native_content_pilot_20261019/prepare_inputs.py
Independent CPU validation after success:
python3 -B /data/coding/sgmse/experiments/p8_voicebank_native_content_pilot_20261019/validate_inputs.py

Future model driver remains unimplemented and unauthorized. Output native SoundFile serialization will require same-libsndfile memory re-encoding proof, not ideal floor bit equality. Failed local conversion gate remains failed; reused noise/speakers are not new holdout.
