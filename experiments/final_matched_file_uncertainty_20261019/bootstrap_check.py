"""Independent CSV join, per-ID arithmetic, bincount bootstrap and linear endpoints.
Does not import bootstrap_main. Shared IO/lifecycle only; shared NumPy RNG backend."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
import csv,re,json,math
from pathlib import Path
import numpy as np

def load_tables(paths,metrics,expected_n):
    parsed={};allids=None
    for key,file in paths.items():
        record={}
        with Path(file).open(newline='') as f:
            reader=csv.DictReader(f)
            for row in reader:
                m=re.search(r'fileid_(\d+)\.wav\Z',row['utt'])
                if m is None or (m.start()>0 and row['utt'][m.start()-1]!='_'):raise ValueError('Bad utt terminal ID')
                identity=int(m[1])
                if identity in record:raise ValueError('Repeated ID')
                data=[]
                for column in metrics:
                    value=float(row[column])
                    if not math.isfinite(value):raise ValueError('Nonfinite metric')
                    data.append(value)
                record[identity]=data
        if len(record)!=expected_n:raise ValueError('Wrong source row count')
        keys=sorted(record)
        if allids is not None and keys!=allids:raise ValueError('Unmatched fileIDs')
        allids=keys;parsed[key]=record
    return allids,{key:np.asarray([rows[i] for i in allids],dtype=np.float64) for key,rows in parsed.items()}

def analyze(tables,cfg,original):
    n=cfg['matched_file_ids'];matrix=np.empty((n,2*len(cfg['metrics'])));columns=[]
    for ai,arm in enumerate(cfg['arms'][1:]):
        for mi,metric in enumerate(cfg['metrics']):
            column=ai*len(cfg['metrics'])+mi;columns.append(arm+'_vs_off__'+metric)
            for row in range(n):matrix[row,column]=((tables[(arm,cfg['splits'][0])][row,mi]-tables[('off',cfg['splits'][0])][row,mi])+(tables[(arm,cfg['splits'][1])][row,mi]-tables[('off',cfg['splits'][1])][row,mi]))/2
    rng=np.random.Generator(np.random.PCG64(cfg['seed']));indices=rng.integers(0,n,size=(cfg['replicates'],cfg['draw_size']),dtype=np.uint16);draws=np.empty((cfg['replicates'],len(columns)))
    for j,index in enumerate(indices):weights=np.bincount(index.astype(np.int64),minlength=n)/cfg['draw_size'];draws[j]=weights@matrix
    means=np.sum(matrix,axis=0)/n;result=[]
    for j,col in enumerate(columns):
        arm,metric=col.split('_vs_off__');old=original['paired_bootstrap'][arm+'_vs_off'][metric];sorted_draws=np.sort(draws[:,j]);ci=[]
        for q in [.025,.975]:
            position=q*(len(sorted_draws)-1);lo=math.floor(position);hi=math.ceil(position);ci.append(float(sorted_draws[lo]+(sorted_draws[hi]-sorted_draws[lo])*(position-lo)))
        if abs(means[j]-old['mean_delta'])>cfg['original_mean_absolute_tolerance']:raise ValueError('Historical mean mismatch')
        result.append(dict(contrast=arm+'_vs_off',metric=metric,mean_delta=float(means[j]),historical_mean_delta=old['mean_delta'],historical_CI95=old['ci95'],matched_file_CI95=ci,mean_reproduction_absolute_error=float(abs(means[j]-old['mean_delta']))))
    return dict(columns=columns,comparisons=result,matched_file_n=n,replicates=cfg['replicates'],seed=cfg['seed'],bootstrap_unit=cfg['bootstrap_unit'],quantile_method='linear'),dict(indices=indices,draw_means=draws)

def compare(a,b):
    if isinstance(a,dict):
        if set(a)!=set(b):raise AssertionError('Keys mismatch')
        for key in a:compare(a[key],b[key])
    elif isinstance(a,list):
        if len(a)!=len(b):raise AssertionError('Lengths mismatch')
        for x,y in zip(a,b):compare(x,y)
    elif isinstance(a,(int,float)) and not isinstance(a,bool):
        if not math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12):raise AssertionError((a,b))
    elif a!=b:raise AssertionError((a,b))
if __name__=='__main__':
    import sys
    from bootstrap_io import execute
    if sys.argv[1:]!=['--validate-after-completed-run']:raise RuntimeError('Explicit independent validation')
    sys.exit(execute('validation',load_tables,analyze,compare))
