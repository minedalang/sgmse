#!/usr/bin/env python
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path


ROOT = Path("/data/coding/sgmse")
LOG = ROOT / "future_gd_smooth.log"
OUT = ROOT / "experiments/rrsd_future_grad_diagnostic_audit_20261020"


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    text = LOG.read_text(errors="replace")
    diag = re.findall(r"\[RRSD-DIAG\] batch=(\d+)/16 .*?grad_abs=([0-9.eE+-]+)", text)
    fd = re.findall(r"\[RRSD-FD\] eps=([0-9.eE+-]+) selected=([0-9.eE+-]+).*?delta_zero=([+0-9.eE-]+) delta_plus=([+0-9.eE-]+)", text)
    smooth = re.findall(r"\[RRSD-SMOOTH\] k=(3|5) .*?budget_rel_err=([0-9.eE+-]+)", text)
    result = {
        "experiment_id": "rrsd_future_grad_diagnostic_audit_20261020",
        "source_log": str(LOG),
        "source_log_sha256": sha(LOG),
        "diagnostic_batches": len(diag),
        "diagnostic_batch_ids": [int(x[0]) for x in diag],
        "grad_abs_values": [float(x[1]) for x in diag],
        "finite_difference_records": len(fd),
        "finite_difference_eps_values": sorted(set(float(x[0]) for x in fd)),
        "finite_difference_selected_mean": sum(float(x[1]) for x in fd) / len(fd),
        "finite_difference_delta_zero_nonzero": sum(abs(float(x[2])) > 0.0 for x in fd),
        "finite_difference_delta_plus_nonzero": sum(abs(float(x[3])) > 0.0 for x in fd),
        "smooth_records": len(smooth),
        "smooth_budget_rel_error_max": max(float(x[1]) for x in smooth),
        "completed_without_save": "[RRSD-DIAG] Completed; exiting diagnostic run without saving." in text,
        "dataset_modified": False,
        "checkpoint_modified": False,
        "training_repeated": False,
    }
    assert result["diagnostic_batches"] == 16
    assert result["diagnostic_batch_ids"] == list(range(1, 17))
    assert result["finite_difference_records"] == 16
    assert result["smooth_records"] == 32
    assert result["completed_without_save"]
    assert max(result["grad_abs_values"]) > 0.0
    (OUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
