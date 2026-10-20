# Harm Teacher finite-difference probe

This is a read-only diagnostic using the existing Router-anchor finite-difference logs. No data, checkpoint, or model code was changed. The probe perturbs selected conflict tiles by `fd_eps=0.005` and checks whether increasing alpha decreases Future error while increasing layer-3 WavLM error.

- Source rows: **86**; rows with selected conflict tiles: **64**.
- All rows: plus condition `0.535`, minus condition `0.616`, bidirectional `0.442`.
- Conflict rows only: plus `0.719`, minus `0.828`, bidirectional `0.594`.
- Low t `[0.05,0.2)`: plus `0.786`, minus `0.929`, bidirectional `0.714`.
- Mid t `[0.2,0.5)`: plus `0.667`, minus `0.750`, bidirectional `0.500`.

## Interpretation

The signs support a local harm label: in selected conflicts, increasing Router alpha usually improves the Future proxy and worsens WavLM layer-3 distance, while decreasing alpha reverses those changes. The agreement is stronger at low t than mid t, so a learned Harm Teacher should include diffusion time and should be compared against a time-only gate. This does **not** establish DNSMOS benefit; WavLM remains an imperfect perceptual proxy, and the rows come from a small diagnostic subset.

The next gate is an oracle sampler test: use the locally measured harm score to attenuate alpha and evaluate PESQ, STOI, SI-SDR, and DNSMOS before training a deployable head.
