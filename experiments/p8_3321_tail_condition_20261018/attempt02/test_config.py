"""CPU parser regression: no model import/load or inference."""
from pathlib import Path
import copy, importlib.util, yaml
p=Path(__file__).with_name('run_pilot.py')
spec=importlib.util.spec_from_file_location('pilot',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
cfg=yaml.safe_load(p.with_name('config.yaml').read_text());assert m.validate_config(cfg)
assert all(type(k) is str for k in cfg['source_sha256']) and len(cfg['source_sha256'])==5
for field,bad in [('backend',False),('rrsd','false'),('rrsd',0),('use_ema','true'),('use_ema',1)]:
    broken=copy.deepcopy(cfg);broken[field]=bad
    try:m.validate_config(broken)
    except AssertionError:pass
    else:raise AssertionError(f'Accepted invalid type for {field}')
broken=copy.deepcopy(cfg);broken['source_sha256'][False]=broken['source_sha256'].pop('off')
try:m.validate_config(broken)
except AssertionError:pass
else:raise AssertionError('Accepted boolean source key')
print(m.synthetic_checks())
print('PASS: five string source keys; strict boolean/backend types; boolean key regression rejected; no inference')
