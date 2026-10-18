import csv,json,time,sys
from pathlib import Path
import numpy as np,torch,soundfile as sf
ROOT=Path('/data/coding/sgmse');E=ROOT/'experiments/frontend_state_representation_probe_20261020_online16';M=ROOT/'experiments/tr0_fresh_cv_manifest_20261018/manifest.csv';CK=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt';sys.path.insert(0,str(ROOT))
from sgmse.joint_data_module import JointWaveDataModule
from sgmse.joint_model import JointSemambaSgmseModel
from sgmse.util.other import pad_spec
D=torch.device('cuda' if torch.cuda.is_available() else 'cpu');start=time.time(); model=JointSemambaSgmseModel.load_from_checkpoint(str(CK),map_location='cpu',strict=False,data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False);model.eval(no_ema=False).to(D)
allrows=list(csv.DictReader(M.open())); rows=[]; seen=set()
for rr in allrows:
 if rr['speaker'] not in seen: rows.append(rr); seen.add(rr['speaker'])
 if len(rows)==16: break
assert len(set(r['speaker'] for r in rows))==16
leaves=[(n,m) for n,m in model.frontend.backbone.named_modules() if n and len(list(m.children()))==0][:3]; hook_data={}
def hk(name):
 def f(mod,inp,out):
  z=out[0] if isinstance(out,(tuple,list)) else out
  if torch.is_tensor(z): hook_data[name]=(tuple(z.shape),float(z.float().mean()),float(z.float().std()))
 return f
handles=[m.register_forward_hook(hk(n)) for n,m in leaves]; rec=[]
for j,r in enumerate(rows):
 no,_=sf.read(r['noisy'],dtype='float32');cl,_=sf.read(r['clean'],dtype='float32');n=min(len(no),len(cl));no=no[:n];cl=cl[:n]; x=torch.from_numpy(no)[None,None,:].to(D); c=torch.from_numpy(cl)[None,None,:].to(D); norm=x.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12); hook_data.clear()
 with torch.inference_mode(): pre=model._frontend_forward(x/norm)*norm
 noisy_hooks=dict(hook_data); hook_data.clear()
 with torch.inference_mode(): _=model._frontend_forward(c/norm)
 clean_hooks=dict(hook_data)
 with torch.inference_mode():
  ym=pad_spec(torch.unsqueeze(model._forward_transform(model._stft(pre.squeeze(1)/norm.squeeze(1))),0)); yc=pad_spec(torch.unsqueeze(model._forward_transform(model._stft(x.squeeze(1)/norm.squeeze(1))),0)); xc=pad_spec(torch.unsqueeze(model._forward_transform(model._stft(c.squeeze(1)/norm.squeeze(1))),0))
 ns=[]
 def sink(it):
  if int(it['step_index']) in (3,5,7,9):
   with torch.inference_mode():
    sc=model(it['x_t'],ym,it['t'],y_cond=yc); x0=model._rrsd_score_to_x0(it['x_t'],ym,it['t'],sc)[0]; ns.append(float((torch.abs(x0-xc).float().square().mean()/torch.abs(xc).float().square().mean().clamp_min(1e-12)).item()))
 torch.manual_seed(20261020+j)
 with torch.inference_mode(): model.enhance(pre.squeeze(0),y_cond=x.squeeze(0),sampler_type='pc',predictor='reverse_diffusion',corrector='none',N=30,snr=0.5,trace_sink=sink)
 rec.append({'filename':r['filename'],'speaker':r['speaker'],'snr_db':float(10*np.log10((np.mean(cl**2)+1e-12)/(np.mean((no-cl)**2)+1e-12))),'noisy_rms':float(np.sqrt(np.mean(no**2))),'cosine':float(np.dot(no,cl)/(np.linalg.norm(no)*np.linalg.norm(cl)+1e-12)),'nmse_mean':float(np.mean(ns)),'nmse_states':ns,'hook_noisy':noisy_hooks,'hook_clean':clean_hooks})
for h in handles:h.remove()
# fixed feature sets and speaker LOSO ridge
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
y=np.array([r['nmse_mean'] for r in rec]); groups=[r['speaker'] for r in rec]
sets={ 'simple':['snr_db','noisy_rms','cosine'], 'hook':[], 'combined':['snr_db','noisy_rms','cosine']}
for n in ['act','output_layer','pyramid_upsample']:
 sets['hook'] += [f'{n}_nm','{0}_sd'.format(n)]
 sets['combined'] += [f'{n}_nm','{0}_sd'.format(n)]
for r in rec:
 for n in ['act','output_layer','pyramid_upsample']:
  r[f'{n}_nm']=r['hook_noisy'][n][1];r[f'{n}_sd']=r['hook_noisy'][n][2]
results={}
for name,keys in sets.items():
 X=np.array([[r[k] for k in keys] for r in rec]); pred=np.zeros(16)
 for g in sorted(set(groups)):
  tr=np.array([q!=g for q in groups]);te=~tr; mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd<1e-8]=1; z=Ridge(alpha=1).fit((X[tr]-mu)/sd,y[tr]);pred[te]=z.predict((X[te]-mu)/sd)
 results[name]={'mae':float(np.mean(abs(pred-y))),'r2':float(r2_score(y,pred)),'spearman':float(spearmanr(y,pred).statistic)}
with (E/'per_utterance.csv').open('w',newline='') as f:
 keys=['filename','speaker','snr_db','noisy_rms','cosine','nmse_mean','nmse_states'];w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows([{k:r[k] for k in keys} for r in rec])
out={'experiment_id':'frontend_state_representation_probe_20261020_online16','n':16,'n_speakers':16,'selected_steps':[3,5,7,9],'results':results,'target_mean':float(y.mean()),'runtime_seconds':time.time()-start,'hook_modules':[n for n,m in leaves]};(E/'metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
