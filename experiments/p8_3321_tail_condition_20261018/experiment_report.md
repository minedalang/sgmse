# 3321 original condition-gap mechanism probe

Status: failed before model load or inference.

The single approved execution stopped after5.91s with KeyError('off') at run_pilot.py112. PyYAML interpreted the unquoted source_sha256 mapping key off as boolean False; the driver expects the string key off. This is a configuration parsing bug. No frontend was computed, neither backend run executed, and NFE/energy effects are unavailable.

Required config.yaml and metrics.json plus failure_archive/failure_snapshot.json preserve the failed attempt. worker_failure_diagnosis.json records parsed key types and independent post-failure hashes. The four source WAV hashes and checkpoint hash match configured values. No retry was performed. The failed driver and config remain unchanged; an engineer must quote the off key and validate all arm keys before a separately approved new attempt.

Planned only: original3321 PCM16 input, seed513132500, condition gap22.62–23.66s with20ms internal fades, globally held float frontend, exact frontend/no_op PCM16 gates, two off NFE30 runs. No claims about this causal effect can be drawn from this failed attempt. Prior results and negatives are preserved.
