import csv,json,time,sys
from pathlib import Path
import numpy as np,torch,soundfile as sf
ROOT=Path('/data/coding/sgmse');E=ROOT/'experiments/frontend_state_representation_probe_20261020';M=ROOT/'experiments/tr0_fresh_cv_manifest_20261018/manifest.csv';CK=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt';sys.path.insert(0,str(ROOT))
def cos(a,b): return float(np.dot(a,b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12))
def snr(no,cl): return float(10*np.log10((np.mean(cl**2)+1e-12)/(np.mean((no-cl)**2)+1e-12)))
from sgmse.joint_data_module import JointWaveDataModule
from sgmse.joint_model import JointSemambaSgmseModel
D=torch.device('cuda' if torch.cuda.is_available() else 'cpu');start=time.time()
model=JointSemambaSgmseModel.load_from_checkpoint(str(CK),map_location='cpu',strict=False,data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False);model.eval(no_ema=False).to(D)
allrows=list(csv.DictReader(M.open())); rows=allrows[:8]; rec=[]; hook_data={}
# choose three leaf modules with parameters from Storm backbone
leaves=[(n,m) for n,m in model.frontend.backbone.named_modules() if n and len(list(m.children()))==0][:3]
hooks=[]
def hook(name):
 def f(mod,inp,out):
  z=out[0] if isinstance(out,(tuple,list)) else out
  if torch.is_tensor(z): hook_data[name]=(tuple(z.shape),float(z.float().mean().item()),float(z.float().std().item()))
 return f
for n,m in leaves: hooks.append(m.register_forward_hook(hook(n)))
for r in rows:
 no,_=sf.read(r['noisy'],dtype='float32');cl,_=sf.read(r['clean'],dtype='float32');n=min(len(no),len(cl));no=no[:n];cl=cl[:n];stats={}
 for label,w in [('noisy',no),('clean',cl)]:
  x=torch.from_numpy(w)[None,None,:].to(D);norm=x.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12);hook_data.clear()
  with torch.inference_mode(): pre=model._frontend_forward(x/norm)*norm
  stats[label]={'hooks':dict(hook_data),'rms':float(torch.sqrt(torch.mean(pre**2)).item())}
 rec.append({'filename':r['filename'],'speaker':r['speaker'],'snr_db':snr(no,cl),'noisy_clean_cos':cos(no,cl),'hook_stats':stats})
for h in hooks:h.remove()
(E/'per_utterance.json').write_text(json.dumps(rec,indent=2)+'\n')
out={'experiment_id':'frontend_state_representation_probe_20261020','status':'representation_smoke_completed','n':8,'n_speakers':4,'device':str(D),'runtime_seconds':time.time()-start,'representation':'StormFrontend backbone hook outputs; 3 leaf modules; noisy and clean passes','hook_modules':[n for n,m in leaves],'metrics':{'mean_snr_db':float(np.mean([r['snr_db'] for r in rec])),'mean_noisy_clean_cos':float(np.mean([r['noisy_clean_cos'] for r in rec]))},'limitations':['first 8 manifest rows contain 4 speakers (2 utterances each), not 8 distinct speakers','no valid OOF proxy label joined; no training or official test access','hook embeddings are descriptive; no explanatory-power claim']}
(E/'metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
