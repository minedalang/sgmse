"""Matched-fileID supplementary uncertainty only; frozen methods/metrics/data."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import csv,re,json,math
from pathlib import Path
import numpy as np

def load_tables(paths,metrics,expected_n):
    tables={}
    for key,path in paths.items():
        rows={}
        with Path(path).open(newline='') as f:
            reader=csv.DictReader(f)
            if not {'utt',*metrics}.issubset(reader.fieldnames or []):raise ValueError('Missing CSV fields')
            count=0
            for row in reader:
                match=re.search(r'(?:^|_)fileid_([0-9]+)\.wav$',row['utt'])
                if match is None:raise ValueError('Malformed final fileid suffix')
                number=int(match[1]);values=[float(row[m]) for m in metrics]
                if number in rows or not all(math.isfinite(v) for v in values):raise ValueError('Duplicate or nonfinite row')
                rows[number]=values;count+=1
        if count!=expected_n:raise ValueError('CSV must have exact150 rows')
        tables[key]=rows
    ids=sorted(next(iter(tables.values())))
    if any(sorted(v)!=ids for v in tables.values()):raise ValueError('Matched-ID sets differ')
    return ids,{key:np.asarray([rows[k] for k in ids],dtype=np.float64) for key,rows in tables.items()}

def contrasts(tables,cfg):
    matrices=[];columns=[]
    for arm in cfg['arms'][1:]:
        delta=sum((tables[(arm,split)]-tables[('off',split)] for split in cfg['splits']))/2
        matrices.append(delta);columns.extend(arm+'_vs_off__'+m for m in cfg['metrics'])
    return np.column_stack(matrices),columns

def analyze(tables,cfg,original):
    matrix,columns=contrasts(tables,cfg);means=matrix.mean(axis=0);rng=np.random.default_rng(cfg['seed']);indices=rng.integers(0,cfg['matched_file_ids'],size=(cfg['replicates'],cfg['draw_size']),dtype=np.uint16)
    # Same gathered matchedID draws used for both arms and all sevenmetrics.
    draws=np.empty((len(indices),len(columns)),dtype=np.float64)
    for start in range(0,len(indices),250):draws[start:start+250]=matrix[indices[start:start+250]].mean(axis=1)
    endpoints=np.quantile(draws,[.025,.975],axis=0,method='linear');result=[]
    for j,column in enumerate(columns):
        arm,metric=column.split('_vs_off__');old=original['paired_bootstrap'][arm+'_vs_off'][metric]
        if abs(means[j]-old['mean_delta'])>cfg['original_mean_absolute_tolerance']:raise ValueError('Original pooled mean mismatch')
        result.append(dict(contrast=arm+'_vs_off',metric=metric,mean_delta=float(means[j]),historical_mean_delta=old['mean_delta'],historical_CI95=old['ci95'],matched_file_CI95=endpoints[:,j].tolist(),mean_reproduction_absolute_error=float(abs(means[j]-old['mean_delta']))))
    return dict(columns=columns,comparisons=result,matched_file_n=cfg['matched_file_ids'],replicates=cfg['replicates'],seed=cfg['seed'],bootstrap_unit=cfg['bootstrap_unit'],quantile_method='linear'),dict(indices=indices,draw_means=draws)
if __name__=='__main__':
    import sys
    from bootstrap_io import execute
    if sys.argv[1:]!=['--execute-after-root-review']:raise RuntimeError('Worker-only explicit invocation')
    sys.exit(execute('main',load_tables,analyze))
