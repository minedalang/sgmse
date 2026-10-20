# P8 checkpoint base identity — software preparation only

One standard-library script, nine synthetic tests, no comparison run. No true storage hashing, model import, torch/numpy import, waveform or dataset access, network, training, checkpoint write or extraction. During design only actual ZIP directories, data.pkl and byteorder were read: 300k metadata 295222 bytes, 80k metadata 113258 bytes, both little-endian; observed GLOBALs exactly the four Root-listed symbols. No semantic weight identity conclusion yet.

Restricted Unpickler resolves only collections.OrderedDict, torch.FloatStorage, torch._utils._rebuild_tensor_v2, and sgmse.joint_data_module.JointWaveScpDataModule to standard-library/inert local symbols. It never imports these original modules. Unknown GLOBAL, persistent storage format, tensor rebuild flags and trailing pickle fail closed. Supports float32 contiguous logical tensors with checked shape/offset/strides/storage bounds and explicit byteorder. Non-contiguous/negative/unknown layouts are unknown, not equal. Effective element bytes are streamed from ZIP storage without extraction; big-endian bytes canonicalized per float32 word. No floating-value approximation is used for equality.

Primary result is same-key RAW dnn.* and frontend.* identity. Missing keys/unsupported layouts retained as unknown. Per tensor contains dtype, shape, offset, stride, byteorder and effective SHA; module summaries retain equal/different/unknown. EMA list entries preserve serialized index/order/length and complete effective hash descriptors. Content matches can return multiple RAW names; duplicate content/shape or equal list length does not resolve a parameter-name mapping. Named EMA identity remains unknown. This audit does not prove historical initialization, matched training budget, runtime options, or output equivalence.

Current source explanation only: sgmse/model.py _get_ema_named_parameters skips requires_grad=False; sgmse/joint_model.py rrsd_freeze_base marks only rrsd_router.* parameters trainable. Thus a Router-only EMA list must not be described as applying original 300k EMA to the frozen dnn/frontend under current eval logic. This code observation alone does NOT establish the actual 80k/300k weight identity or historical code path. Runtime records train_type, mode2_include_noisy_main, frontend, frontend_trainable, rrsd_freeze_base, backbone, sde, transform_type, spec_abs_exponent and spec_factor; absent saved fields explicitly unknown rather than assumed defaults.

Worker invocation after separate Root review, once:
```
python3 -B /data/coding/sgmse/experiments/p8_checkpoint_base_identity_20261019/compare_checkpoint.py --execute-after-root-review
```
Preparation self-test (already completed, no real checkpoint reads):
```
python3 -B /data/coding/sgmse/experiments/p8_checkpoint_base_identity_20261019/compare_checkpoint.py --self-test
```

Runtime: exclusive STARTED, absolute 120-second alarm covering source full-file hashes pre/post, metadata parse, storage streaming and final records. Output directory <=2 MiB, no retries. Full checkpoint hashes, data.pkl hashes, storage bytes actually read (including offset skips), unique effective tensor bytes hashed, source/config/script SHA pre/post are separate counters. Storage spans reused through a descriptor cache; cached use is not counted as another storage read. Partial-span streaming does not claim whole-member CRC verification; full source checkpoint SHA is independently checked before/after. If complete output cannot fit, the run fails rather than presenting a filtered successful subset. Config still says execute_authorized=false; this preparation report does not grant Worker permission.

Nine tests: effective offset exact bytes; equal/different; duplicate EMA content matches; unsupported stride unknown; endian canonicalization; out-of-storage range; arbitrary GLOBAL rejection; invalid persistent ID; trailing pickle rejection. All use temporary synthetic ZIPs only. No new manifest or second framework. Independent content evaluation/human-listening permissions unchanged.

Script SHA256: a73fc545ecc1ca73c05d400e9f6cdf0dcfc8637990e4fa1fe5bcdf28a384b8cd

Current explanatory source SHA256:
- `/data/coding/sgmse/sgmse/model.py`: `fcf6e6ab1efefc3615d3ed4044bf33ad1195c43a62573c1f8311f9abf1af7b5d`
- `/data/coding/sgmse/sgmse/joint_model.py`: `0e714397676a04be0e8f8e393fb139df810b2dd41761aaf3fae8673f8d30e9d8`
