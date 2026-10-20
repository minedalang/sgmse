# CPEL stage 0: bounded objective software prototype

Preparation only. No SGMSE import/model/data/audio/checkpoint access; CPU tensor
arithmetic and gradients, no fitting, optimizer, network or enhancement result.
Read current entry, AGENTS, full attachment (SHA256
75e785ebb749a77f37c91741fd2a0c7fc606416903ac9e6b1116d07dce12a613), CPEL topic
and detailed assessment. Existing dirty git state untouched.

`objective.py` calculates projection gain, squared lower/upper hinges,
normalized orthogonal energy and unnormalized outside energy. P is elementwise
soft weighting, NOT an idempotent projection. Coordinates sum within each
sample; equal mean over valid samples. Invalid zero/sub-threshold target energy
skips the entire sample, including outside penalty, explicitly reports validity
and skipped count. No epsilon creates a direction. Threshold and all margins /
weights are **software examples, not approved training hyperparameters**.

Lower margin >0 imposes a minimum amplitude; lower=0 permits zero. Pure cosine
admits tiny aligned output. Noisy passthrough responds to speech AND noise. In the actual noise-only case,
clean delta is zero: valid=false/skipped=1, so this directional objective does
NOT penalize the nonzero noise response. Its input-direction gain=1 and response
energy are separate diagnostics, never the undefined clean-direction gain.
Zero edit with identical inputs and identical RNG yields d=0 by consistency;
that is not evidence of learned acoustic selectivity.
Shared errors cancel. No term here establishes identifiability or rules out
these shortcuts. No extra loss was added to conceal them.

OUVE helper uses D=(u+sigma²q−(1−a)A)/a. At shared u, kernel target differences
are a delta/sigma²; shared native epsilon instead yields distinct states.
These are kernel identities, not unbiased common-proposal DSM or posterior
recoverability guarantees. Caller detaches u/A; helper does not silently detach.
A trainable scalar Score retains gradients through a frozen coefficient and
fixed state/anchor; eval/freezing must not be confused with no_grad.

`synthetic_check.py`: 18 bounded engineering checks, including real/complex
finite-difference gradcheck, no optimizer. Fixed tensors are written in JSON
for `reference_check.py`, which uses only scalar Python complex arithmetic,
imports neither torch nor objective, and checks every saved arithmetic field
and OUVE identities. Independent checker does not independently prove gradients.

Engineer may run in-memory selftests, not create final metrics/report. After
Root freezes code, Worker executes once from this directory:

```
../../env/bin/python -B synthetic_check.py --output metrics.json
../../env/bin/python -B reference_check.py metrics.json --output independent_validation.json
```

Outputs use exclusive creation. Worker records commands/environment/timing,
config/code hashes and result limitations in experiment_report.md. Root must
review before any actual model integration; no training authorization follows
from passing these software tests.
