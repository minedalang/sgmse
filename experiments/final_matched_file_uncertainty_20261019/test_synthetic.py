import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import csv,tempfile,json,numpy as np
import bootstrap_main as main,bootstrap_check as check
D=Path(__file__).resolve().parent;tests=[]
def ok(x,name):assert x,name;tests.append(name)
cfg=dict(arms=['off','router','tcut005'],splits=['no_reverb','with_reverb'],metrics=['metric'],matched_file_ids=3,seed=20261019,replicates=11,draw_size=3,bootstrap_unit='matched',original_mean_absolute_tolerance=2e-10)
tables={('off',s):np.zeros((3,1)) for s in cfg['splits']};tables.update({('router','no_reverb'):np.asarray([[1],[2],[3.]]),('router','with_reverb'):np.asarray([[-1],[4],[5.]]),('tcut005','no_reverb'):np.ones((3,1))*2,('tcut005','with_reverb'):np.ones((3,1))*4})
original=dict(paired_bootstrap=dict(router_vs_off=dict(metric=dict(mean_delta=7/3,ci95=[-8,9])),tcut005_vs_off=dict(metric=dict(mean_delta=3,ci95=[2,4]))))
a,aa=main.analyze(tables,cfg,original);b,bb=check.analyze(tables,cfg,original);check.compare(a,b);ok(np.array_equal(aa['indices'],bb['indices']) and np.allclose(aa['draw_means'],bb['draw_means'],rtol=1e-12,atol=1e-12),'gathermeans vsindependentbincount sums alldraws')
ok(a['comparisons'][0]['mean_delta']==7/3 and a['comparisons'][1]['mean_delta']==3 and a['comparisons'][0]['historical_CI95']==[-8,9],'pairedsplit arithmetic preservesmeans historicalCI')
ok(np.all(aa['draw_means'][:,1]==3) and len(a['comparisons'][0]['matched_file_CI95'])==2,'constantpairedcontrast interval endpoints')
with tempfile.TemporaryDirectory(prefix='synthetic_',dir=D) as tmp:
 p=Path(tmp)/'test.csv'
 def write(rows):
  with p.open('w',newline='') as f:
   w=csv.writer(f);w.writerow(['utt','metric']);w.writerows(rows)
 write([['foo_fileid_10.wav',1],['foo_fileid_2.wav',2],['foo_fileid_3.wav',3]])
 ids,values=main.load_tables({('x','y'):p},['metric'],3);iid,iv=check.load_tables({('x','y'):p},['metric'],3);ok(ids==iid==[2,3,10] and np.array_equal(values[('x','y')],iv[('x','y')]),'terminalnumericfileID independentjoin numericordering')
 rejected=0
 for rows in [[['a_fileid_1.wav',1],['b_fileid_1.wav',2],['a_fileid_2.wav',3]],[['a_fileid_1.wav.extra',1],['b_fileid_2.wav',2],['a_fileid_3.wav',3]],[['a_fileid_1.wav','nan'],['b_fileid_2.wav',2],['a_fileid_3.wav',3]],[['a_fileid_1.wav',1]]]:
  write(rows)
  for loader in [main.load_tables,check.load_tables]:
   try:loader({('x','y'):p},['metric'],3)
   except ValueError:rejected+=1
   else:raise AssertionError('invalidCSV accepted')
 ok(rejected==8,'duplicate malformedterminal nonfinite insufficientrows failboth')
assert not (D/'STARTED.json').exists();result=dict(status='passed',count=len(tests),checks=tests,true_final_bootstrap_runs=0,real_CSV_values_analyzed=0,audio_reads=0,model_calls=0)
(D/'synthetic_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
