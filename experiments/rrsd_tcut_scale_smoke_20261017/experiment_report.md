# Partial RRSD cutoff compatibility smoke

A one-file CUDA inference with `N=3` compared the pre-patch invocation (omitting `--rrsd_t_cut_scale`) against the explicit new option `--rrsd_t_cut_scale 0.0`. Both used the same 80k EMA checkpoint, seed, input and `rrsd_t_cut=0.05`.

The output WAVs were byte-identical (SHA256 `cf50c1baaedcd88fce2c403cca8f9802ed2b17e37c1de3f2a4443b55b6b49559`). Python compilation of the modified inference/model files also passed. This confirms the new partial-gate option preserves the historical hard-cut behavior at scale 0.0.
