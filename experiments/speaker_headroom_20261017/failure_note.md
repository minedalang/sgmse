# Initial launch failure

The first launch failed immediately with `ModuleNotFoundError: No module named 'sgmse'` because Python executed `probe.py` with `experiments/` as its import path. No audio, checkpoint, or dataset was modified. The reproducible launch must set `PYTHONPATH=/data/coding/sgmse`.
