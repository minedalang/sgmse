Model phase implementation prepared only; no model/audio-source execution.

Native main runs once over16 symlinked noisy PCM inputs only. All arguments/returns remain original; observations capture actual file seed, noisy peak, frontend shape before match, native center crop/pad, matched rawfrontend used by backend, actual condition, OFF kwargs, sampler tuple NFE and preclip backend. Return profiler reads only float norm_factor/int T_orig for locked ScoreModel.enhance code; frame and locals are never retained. EMA eval(no_ema=False) and mode2/nonresidual/storm/frontend_ckpt=None are checked at runtime. No timeit or trace_sink changes.

All32 raw output arrays retained. Frontend uses native SoundFile float32 PCM_16 conversion without gain; OFF uses unchanged main clip/write. Same-libsndfile BytesIO reencoding proves exact serialized bytes. Counts for preclip >1/<-1, endpoints, PCM representability saturation and saved peaks remain visible. Independent validator recomputes32 serialization proofs,52 independent canonical word/character DP scores, weighted/sentence aggregates and case/reference-position transitions. Adverse content/clean-control scores do not gate/exclude cases.

Enhancement weakrefs must die after native.main, gc/synchronize/empty_cache, before local tiny.en load. Actual calibration source/scorer/decode settings and options locked;52 full native PCM transcribe requests only (4 new scaledclean,16noisy,16matchedfrontend,16OFF). No old unscaledclean redecoding. API counts are not neural-forward counts.

Supervisor root STARTED.json is independent of existing CPU marker;410s execution+10s finalization,40MB includes original preparation and logs/validation; no retries. Source/input/driver locks checked pre/post/supervisor-post. Current input registry and every preexisting experiment file frozen. Native loader torch.load accepts str/Path or seekable real local file whose descriptor inode/name identifies frozen source, without seeking/changing original args or object. Unknown filelike refused.

Network sockets/DNS blocked before numeric/model imports. Explicit allowlist permits only native local ninja/compiler version and ninja build subprocesses, with locked executables and cached extension cwd. Native successful PATH reused. XDG_CACHE_HOME is redirected only after enhancement release, to preserve native existing extension cache. No dependency probing or compilation was executed during preparation. Runtime native import may legitimately invoke local cached-build tools; unknown subprocess or shape/source failure stops and preserves failure, never retries. Future actual environment/kernel loading remains untested.

Mock tests:16 independent native context dicts, identical wrapper args/kwargs/return identity, NFE30, float norm profiler, weakref model release,32 synthetic libsndfile encodings,225 independently implemented DP pairs, local filelike position preservation and unknown-filelike/network-process rejection. Fake torch/model used; actual model loads=0, real source audio decodes=0, network=0. Initial static-review issues (dict aliasing, filelike loader, blanket subprocess block) fixed before freeze.

Authorized future Worker command after Root review:
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_voicebank_native_content_pilot_20261019/model_stage/run_pilot.py
Independent CPU validation after successful model run:
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_voicebank_native_content_pilot_20261019/model_stage/validate_model_run.py
