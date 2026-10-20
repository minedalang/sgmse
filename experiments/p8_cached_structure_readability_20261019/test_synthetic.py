"""10 small synthetic checks. No true source audio/header read or real fitting."""
import os
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import ast,json,numpy as np
import readout as main,independent_readout as independent
checks=[]
def ok(value,name):assert value,name;checks.append(name)
z=main.representation(np.zeros(803));ok(z['full'].shape==(4,119) and z['simple'].shape==(4,6) and np.all(z['identity']==0) and z['unused_tail_samples']==3,'zero targets dimensions fullframes tail and powerfloor')
rng=np.random.default_rng(20261019);wave=rng.normal(0,.1,1127);a=main.representation(wave);b=independent.representation(wave);ok(all(np.allclose(a[k],b[k],rtol=1e-10,atol=1e-12) for k in ['full','simple','identity']),'independent FFT feature bookkeeping')
ok(a['identity'][0,1]==0 and np.array_equal(a['full'][0,:17],a['full'][0,17:34]) and np.array_equal(a['full'][0,:17],a['full'][0,34:51]),'firstflux0 edgeframecontext repetition')
# One-sided framepower satisfies exact DFT Parseval normalization.
h=.5-.5*np.cos(2*np.pi*np.arange(320)/319);expected=np.dot((wave[:320]*h),(wave[:320]*h))/np.dot(h,h);ok(np.isclose(10**a['simple'][0,0],expected,rtol=1e-12),'onesided DC Nyquist weights and power normalization')
xx=[rng.normal(size=(3,6)),rng.normal(size=(7,6))];xx[0][:,0]=.1;xx[1][:,0]=.1;yy=[rng.normal(size=(3,2)),rng.normal(size=(7,2))]
m=main.fit(xx,yy);n=independent.fit(xx,yy);ok(np.allclose(m['coefficients'],n['coefficients'],rtol=1e-9,atol=1e-10) and m['std'][0]==1 and abs(m['weight_sum']-1)<1e-15,'normalEq versus weightedaugmentedlstsq and exactconstant std1')
ok(np.allclose(m['train_target_mean'],(yy[0].mean(axis=0)+yy[1].mean(axis=0))/2),'caseequal frameequal targetmean not pooledframe mean')
# Duplicating frames within one case leaves normalized objective/lambda unchanged.
m2=main.fit([np.repeat(xx[0],2,axis=0),xx[1]],[np.repeat(yy[0],2,axis=0),yy[1]]);ok(np.allclose(m['coefficients'],m2['coefficients'],rtol=1e-10,atol=1e-10),'weight sum1 lambda invariant to within-case duplicateframes')
constant=main.fit([np.zeros((2,2)),np.zeros((3,2))],[np.ones((2,2))*2,np.ones((3,2))*4]);p=main.predict(np.zeros((1,2)),constant);ok(np.allclose(p,3) and np.all(p>1),'intercept unpenalized predictions unclipped')
#32tiny cases (2frames x6/119) enforce full fixed LOSO count, not a large synthetic matrix.
cases=[]
for speaker in range(4):
 for j in range(8):
  f=rng.normal(size=(2,119));simple=f[:,:6];identity=rng.random((2,2));cases.append(dict(basename=f'synthetic{speaker}_{j}',id=f'u{speaker}_{j//4}',speaker='s'+str(speaker),noise='N1' if j%2 else 'N2',snr_db=-18 if j%4<2 else 0,target=rng.random((2,2)),noisy=dict(full=f,simple=simple,identity=identity),frontend=dict(full=f*.9,simple=simple*.9,identity=identity*.9),tail=3))
r,pred=main.evaluate(cases);rr,pp=independent.evaluate(cases);independent.compare(r,rr);ok(len(r['fits'])==16 and len(r['cases'])==32 and len(pred)==256 and all(np.allclose(pred[k],pp[k],rtol=1e-8,atol=1e-9) for k in pred),'16dualoutputfits32cases OOF independent scope')
for fold in r['fits']:
 train=set(fold['train_cases']);held=set(fold['heldout_cases']);assert not train&held and len(train)==24 and len(held)==8
# Train-only fitting must not change under heldout modifications.
first=r['fits'][0];modified=[dict(c) for c in cases]
for c in modified:
 if c['speaker']=='s0':c['target']=c['target']+100
r2,_=main.evaluate(modified);ok(np.array_equal(first['mean'],r2['fits'][0]['mean']) and np.array_equal(first['coefficients'],r2['fits'][0]['coefficients']),'LOSO train-only standardization coefficients constant isolate heldouttargets')
D=Path(__file__).resolve().parent
for p in D.glob('*.py'):ast.parse(p.read_text())
assert not (D/'STARTED.json').exists() and not (D/'metrics.json').exists()
result=dict(status='passed',count=len(checks),checks=checks,actual_cache_pcm_reads=0,real_probe_fits=0,synthetic_probe_fits='smallarrays only',enhancement_training_calls=0,network_bytes=0)
(D/'synthetic_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
