# Local reverb resource audit — 2026-10-18

## Question

Can the training/test reverb-mismatch hypothesis be tested with an independently sourced real room impulse response (RIR) already present on this machine, without modifying `/data/datasets/dns_ins20`?

## Read-only checks

- The DNS Ins20 README states that the official `with_reverb` split uses internally recorded Microsoft RIRs with RT60 between 300 ms and 1300 ms.
- The local generator configurations reference RIR locations on the original machine under `/home/xyf/...`; both referenced directories are absent in the current environment.
- No RIR or impulse-response WAV files were found in `/data/coding/sgmse` outside its environment directories.
- The official local test tree contains 150 clean/noisy pairs for each of `no_reverb` and `with_reverb`; these are already generated evaluation data and were not copied or modified.
- Configuration and README SHA256 values are recorded in `config.yaml`.

## Decision

The earlier four-speaker fixed-FIR smoke cannot be upgraded to a real-RIR validation with the resources currently available. Do not extend that artificial-FIR route and do not start reverb training or a reverb adapter. A new reverb experiment would first require an independently supplied, fixed RIR resource and a fresh-CV manifest; obtaining that resource is an external dependency, not a code or permission blocker.

## Limits

This audit does not disprove training/test reverb mismatch. It only establishes that the current workspace lacks the source RIRs needed for a clean causal follow-up. The official `with_reverb` audio remains usable for descriptive paired analysis, but not for constructing a new training view without introducing data leakage or undocumented transformations.
