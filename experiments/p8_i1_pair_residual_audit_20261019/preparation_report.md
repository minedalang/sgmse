# I1a pair/residual audit: PREPARED ONLY

Root design/config preserved. No actual source WAV/header/sample read, no main/validator execution, no source selection, no models/ASR/network/training/mixing/editing or output waveform.20I0record locks actually rehashed;24audio SHA/header/bytes merely inherited from frozen I0/config, NOT revalidated now.

Minimal two scripts: audit_pairs.py computes321directdot lag rows per pair, all20ms/10mshop frames including final partial, exact source stats, rawX-S and full/common zero/best OLS sensitivities. validate_audit.py imports no driver functions: separate int16 decoder uses struct.iter_unpack vs main np.frombuffer, independently computes every statistic and CSV row with centered dot/residual arithmetic and fixed tolerances. Both use numpy float64 but do not share statistical implementation. Fits do not alter source or create aligned audio. Numerical tolerances are not scientific thresholds. All12fixedpairs/6speakers remain; noiseidentityunknown and all scientific/noise/split gatesfalse.

Full/noisy/raw signal mean/power/RMS/peak/endpoints/zeros retained. Full/common OLS residual innerproduct/centeredPearson/mean/power explicitly separated: forced inner orthogonality does not imply Pearson0 or noise purity.321maxcorr selection has winner-selection bias, no range expansion. Algebraic reconstruction is not pure-noise validation.

Single CPU process each, NumPy thread env=1 before import.118s work deadline plus hard absolute120s finalization alarm across postchecks/report, hardtimeout record+exit124. Exclusive main/validation markers; no retries.8MiB complete directory including preparation/records, reserve checks. All sockets/models/unknown processes denied and forbidden-attempt container retained; caught attempts still fail. network_bytes0 is refusal-policy accounting, not packet capture proof. Source write forbidden. PCM sample decode/file/byte counters separate from full-WAV hash bytes; decode-prehash and source-posthash distinct, extra Worker preflight hashes not represented as decodes. Originalmetrics hash protected by validator.

36named synthetic checks in synthetic_test_results.json: known signed/boundary lags, non1gain+DC, constant(exact equality before centered mean, noepsilon)/allzero/negativecorr, tie,3sampletail, OLSinner/Pearson distinction, nonfinite null+reason strictJSON, generated PCM endpoints/truncation, path/hash guards, existingmarker, workdeadline/byte reserve/finalizationalarm, refused model/network/sourcewrite and caught network event. No test accesses actual source audio. Preparation under256KiB.

Future Worker main ONLY after separate Root approval:
```
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_i1_pair_residual_audit_20261019/audit_pairs.py --execute-after-root-review
```
Then independently, exactly once:
```
/data/coding/sgmse/env/bin/python -B /data/coding/sgmse/experiments/p8_i1_pair_residual_audit_20261019/validate_audit.py --validate-after-completed-run
```
Futuremain24PCM file decodes + validator24=48, plus each sourcepre/posthash. No actual run authorized by this report. Full runtime config/metrics/report are standardized; failures retained, hardtimeout records invalidate success.
