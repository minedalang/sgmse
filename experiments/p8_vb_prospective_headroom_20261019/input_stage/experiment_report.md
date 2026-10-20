CPU input preparation: completed

32 prescribed inputs and eight scaled-clean controls; historical noises, full-utterance SNR, common per-utterance downward scale. All source/preparation locks and guards are in metrics.json and input_registry.json. No model/network/training/test access. Local16k clean from registry; ID association only, conversion/spoken alignment unproven. Old failed local-conversion gate unchanged. Model phase requires separate Root approval after input-registry review. Failure never authorizes retry.

## Worker execution and independent verification

Unique preparation session 8975 exited 0 in 4.347545699 seconds. Frozen supervisor records PID 13285 and child PID 13286; process had exited before independent ps, so these are supervisor-recorded PIDs, not fabricated observations. Exact command: `python3 -B /data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/input_stage/prepare_inputs.py`.

Unique prepared independent standard-library validator session 28813 exited 0, reconstructing 40 complete signals (32 noisy + 8 scaled-clean), 80 PCM16/f32le files, original publisher text bytes, 32 seed/basename/order/formula cases, historical full decoded-noise hashes, all peak/rounding/residual/SNR fields. It does not import driver functions. Validation time was not measured. Input registry SHA256 `cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998`.

Source PCM decode count is 10 (8 clean + 2 historical noises); generated PCM files are additionally decoded for roundtrip/independent validation. Zero resampling, model, ASR, training, online/test access, network bytes or retries. All 7 preparation locks and 32 source locks matched pre/post/supervisor/current; original runtime metrics and registry unchanged. Old VB/B metrics preserved. Git entries outside this current experiment unchanged; concurrent Engineer model_stage draft files permitted and not touched. Model stage not authorized or executed.

{
  "maximum_component_snr_error_db": 3.552713678800501e-15,
  "maximum_float32_residual_snr_difference_db": 7.610481134179281e-09,
  "maximum_pcm_residual_snr_difference_db": 7.630831746808616e-05,
  "maximum_quantization_error": 3.051755239760756e-05,
  "saturation_count": 0
}

Scales in frozen utterance order:

* p227_030: 0.31777565471899805
* p227_031: 0.3476628574259755
* p228_026: 0.0959163154210414
* p228_032: 0.0902926435756524
* p230_027: 0.2049204786071085
* p230_029: 0.16837522204850483
* p231_029: 0.3019252336291157
* p231_032: 0.22379661088287464

These are local16k/publisher-ID-associated inputs, not proof of spoken alignment, source-conversion equivalence, actual phone labels or pretraining non-exposure. SNR is full-utterance component SNR; quantized residual SNR is separately named, not physical component truth. All 32 prescribed conditions retained without outcome filtering.

Final sizes are recorded in worker_accounting.json; current parent includes concurrent model draft and excludes symlink target duplication. Input-stage cap24MiB and parent64MiB remain satisfied. No further stage started.
