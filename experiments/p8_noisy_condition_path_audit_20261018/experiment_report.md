# Noisy condition path audit

Completed read-only audit. The first CPU load failed with an import-path error; the Research Lead authorized one corrected launcher retry with PYTHONPATH pointing to the repository. Attempt 02 succeeded. There were two load attempts, one successful checkpoint metadata read, and no model initialization, inference, training, GPU use, dependency install, or checkpoint write. failures.json retains the initial failure, and attempt02_provenance.json records the correction. The checkpoint SHA256 was identical before and after the successful read: ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9.

The completed 20dB pilot metrics record the loaded model as mode2, frontend storm, residual_refine_enable=false. All five recorded pilot runtime source hashes still match the current files (joint_enhance_by_dir.py, sgmse/joint_model.py, sgmse/joint_data_module.py, sgmse/model.py, sgmse/frontends/storm.py). Its backbone source hash was not recorded, limiting historical reconstruction. Current git HEAD is 4ff156483ec35302ba070cffffb27093debf8e52; the working tree contains pre-existing changes.

JointSemambaSgmseModel aliases JointFrontendSgmseModel (joint_model.py:5888). Its enhance method delegates to ScoreModel.enhance when residual_refine_enable=false (2827–2857), and it inherits ScoreModel.forward. enhance aligns the condition waveform, normalizes both inputs using the noisy condition peak in mode2 (model.py:1453–1489), transforms them to complex spectrograms, and binds score_fn=partial(self,y_cond=y_cond) (1491). The sampler's y argument remains the frontend spectrogram Y (1543). Thus the conditioning waveform is separate from the SDE anchor.

ScoreModel.forward (1183–1215) sets y_extra=y_cond for mode2 with mode2_include_noisy_main enabled, and sends scaled x, anchor, condition, and extra input to the backbone. ncsnpp_v2.forward (896–922) falls back to y_extra=y_cond when extra is absent and concatenates exactly [x.real,x.imag,y_extra.real,y_extra.imag,y.real,y.imag], then applies x=2*x-1. Zero-based channels 2 and 3 are noisy evidence, and channels 4 and 5 are the anchor. The six-channel input also enters progressive input paths (943–975) when progressive_input=input_skip. The successful metadata read confirms dnn.all_modules.3.weight=[128,6,3,3] and the progressive input combiner dnn.all_modules.6.Conv_0.weight=[128,6,1,1].

Optional mechanisms are gated separately: selector_type creates a learned evidence selector or residual/noise-prior updates (ncsnpp_v2.py:639–643,730–746,823–841), cross_attn creates bottleneck_cross_attn (787–792), and film_modulate creates conditional decoder blocks (805–818). At forward time, a TF-grid selector builds y_hint from anchor and condition; otherwise y_hint=y_cond (1001–1010). Cross-attention consumes y_hint (1015–1018); FiLM consumes y_cond (1033); residual paths consume y_cond-y (1013,1049+). Saved hyperparameters and the state key snapshot establish that all these optional condition paths are inactive in this checkpoint. Bottleneck cross-attention, TF-grid selector/context, FiLM decoder wrappers/condition networks, residual regeneration, noise-prior guidance, and backend adapter key tests each have zero matching DNN keys. This does not mean the backbone lacks ordinary self-attention: its standard unconditional bottleneck attention is a separate mechanism.

In the 20dB pilot, run_pilot.py:140–149 retains the frontend anchor and seed while editing only the waveform condition in the fixed tail window. It explicitly asserts unchanged outside-window condition samples and unchanged global peak. This prevents normalization from changing the anchor or the global amplitude scale for that recorded intervention. Score trajectory x_t remains free to change after altered score evaluations. The STFT window and network receptive fields can spread an initially localized waveform edit in time; output changes beyond the edited window are expected possibilities. The intervention establishes sensitivity to condition evidence under matched anchor/randomness; it cannot identify a learned selection mechanism without module evidence, and it does not hold the reverse trajectory fixed.


## Actual checkpoint configuration

|Field|Saved in checkpoint|Effective value and basis|
|---|---|---|
|backbone|True|ncsnpp_v2; saved hyperparameter|
|train_type|True|mode2; saved hyperparameter|
|mode2_include_noisy_main|True|True; saved hyperparameter|
|cross_attn|True|False; saved hyperparameter|
|film_modulate|True|False; saved hyperparameter|
|film_levels|True|first; saved hyperparameter|
|selector_type|True|none; saved hyperparameter|
|evidence_selector_type|False|not an argument; not a constructor argument; do not invent saved/default value|
|use_tfgrid_selector|False|False; not a constructor argument; do not invent saved/default value|
|progressive_input|False|input_skip; current constructor default; see source-specific defaults|
|progressive|False|output_skip; current constructor default; see source-specific defaults|
|c_in|True|1; saved hyperparameter|
|residual_refine_enable|True|False; saved hyperparameter|
|backend_adapter_enable|False|False; current constructor default; see source-specific defaults|

`evidence_selector_type` is absent and is not an accepted argument in the actual constructors. The actual argument is `selector_type`. `use_tfgrid_selector` is a derived attribute, resolved false from `selector_type=none`, not an invented saved flag. Absent values are resolved only against the source constructors, which were parsed with AST without importing or instantiating the model. metrics.json records constructor source names and values. film_levels parses to {0} but has no effect with film_modulate=false.

The active noisy evidence path is ordinary learned convolutional processing of concatenated noisy channels, including progressive input skips. Consequently the condition-tail replacement changes this direct evidence path, not attention/FiLM/both and not a learned evidence selector. It holds the frontend waveform and sampled randomness fixed in the recorded pilot while allowing the score outputs and reverse trajectory to change. The pilot's fixed global peak also keeps the normalized anchor spectrogram fixed; this must be checked anew for other interventions. No causal conclusion about content correction, generalization, or selector quality follows from this architecture audit alone.
