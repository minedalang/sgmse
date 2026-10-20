# CPU input preparation worker verification

Completed exactly once; supervisor PID 44927, exec session 46463, exit 0. Runtime 2.523976451018825 s. CPU preparation only: zero model/network/training calls; no test content. No retry, no enhancement/ASR or future model phase. Child PID was not recorded by the frozen supervisor.

Frozen input registry SHA256: `0af790bf7392c8b24697cf1cb4c40e537d424c09d82940661a18ad37dfacdf13`. Input metrics SHA256: `59f41885a76e611c1aef6b71b17d10adc00a22e74565af3f7af8d24ece13d209`. Original registry, metrics, preparation report and frozen files were not changed.

Independent `validate_inputs.py` completed once and reconstructed all 20 WAV/float32 artifact pairs, recipes, order and seeds. Supplemental `worker_validate.py` verified all 4 publisher raw TXT bytes/hashes against the original ZIP and manifest, all 12 source and 8 preparation locks at child pre/post, supervisor post and current, and all recorded float32/PCM residuals, component SNRs and peaks. Zero saturation across all 20 audio artifacts.

Historical failed waveform gate and prior clean-ASR results retained without reruns. Only four repeated utterances/two speakers and reused noises; these inputs establish no content/enhancement benefit.

| Utterance | Samples | Seconds | Shared scale | Group max before scale |
|---|---:|---:|---:|---:|
| p226_027 | 63447 | 3.9654375 | 0.19272256208834648 | 4.6699254630468667 |
| p226_028 | 67653 | 4.2283125 | 0.12425738974272588 | 7.2430299868961052 |
| p287_037 | 60008 | 3.7505000 | 0.23551915311994656 | 3.8213452624876023 |
| p287_054 | 51089 | 3.1930625 | 0.19129400031449756 | 4.7047999337164361 |

| Basename | Samples | Seed | Gain | Component SNR dB | float32 residual SNR dB | PCM residual SNR dB |
|---|---:|---:|---:|---:|---:|---:|
| p226_027_N1_snr_m18.wav | 63447 | 1486336889 | 17.427961317758928 | -18 | -18.000000001476 | -17.999998876101 |
| p226_027_N1_snr_p0.wav | 63447 | 3131377406 | 2.1940503378692475 | 0 | 4.7468151643772e-10 | -3.5306779618728e-05 |
| p226_027_N2_snr_m18.wav | 63447 | 3031475568 | 15.495699398591551 | -18 | -18.000000001011 | -18.000002412572 |
| p226_027_N2_snr_p0.wav | 63447 | 1225915657 | 1.9507929746410495 | 0 | -1.2286481788181e-09 | -3.3554024813908e-05 |
| p226_028_N1_snr_m18.wav | 67653 | 4166214792 | 15.666312494134058 | -18 | -17.999999998901 | -17.999999284023 |
| p226_028_N1_snr_p0.wav | 67653 | 2303162876 | 1.9722718907973824 | -9.6432746655329e-16 | -1.385525453288e-09 | -4.4130688754915e-07 |
| p226_028_N2_snr_m18.wav | 67653 | 2418667005 | 12.379505719013439 | -18 | -17.999999997137 | -18.000010778005 |
| p226_028_N2_snr_p0.wav | 67653 | 521132841 | 1.558487433511724 | 9.6432746655329e-16 | 8.5213085871861e-10 | -0.00010391117451761 |
| p287_037_N1_snr_m18.wav | 60008 | 3194511354 | 14.580549551685374 | -18 | -17.999999996805 | -18.000005480132 |
| p287_037_N1_snr_p0.wav | 60008 | 447921833 | 1.8355824348540768 | 0 | 7.2331792441472e-10 | -5.9169160935174e-05 |
| p287_037_N2_snr_m18.wav | 60008 | 1074088042 | 12.696686104636607 | -18 | -18.000000001368 | -18.000001074023 |
| p287_037_N2_snr_p0.wav | 60008 | 949415772 | 1.5984180782700919 | 1.9286549331066e-15 | -8.6738724343487e-09 | 3.3704417427575e-07 |
| p287_054_N1_snr_m18.wav | 51089 | 2525130446 | 19.367131787412387 | -18 | -18.000000000601 | -18.000002797682 |
| p287_054_N1_snr_p0.wav | 51089 | 3211441175 | 2.4381774360740041 | -9.6432746655329e-16 | -1.5806918320005e-09 | -1.622596535551e-05 |
| p287_054_N2_snr_m18.wav | 51089 | 4034286842 | 15.643519454898039 | -18 | -17.999999996967 | -18.000002127788 |
| p287_054_N2_snr_p0.wav | 51089 | 611393327 | 1.9694024171667577 | 0 | 2.1509478428539e-09 | 7.2351687355407e-06 |

All 20 unquantized peaks <= 0.90000000000000002; PCM peaks <= 0.899993896484375. Max quantizer residual 3.0517570538721173e-05; max float32 rounding 2.9751431984692545e-08; max PCM-vs-float32 residual 3.0517578125e-05. All finite/mono16kPCM16/unchanged-length guards passed. Full per-artifact values are in worker_validation.json and the immutable registry.

Bridge PID 39452 and listening server PID 16022 remained active. Existing git dirty state was inspected in full and recorded in worker_preflight.json; no source, weight, service, previous experiment, research plan, commit or push was changed. Model runtime remains unimplemented/unauthorized.

Final directory bytes (including this fixed-width accounting line): 000007614542; cap 12000000.
