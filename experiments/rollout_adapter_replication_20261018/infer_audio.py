#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch

ROOT=Path(__file__).resolve().parents[2]; EXP=Path(__file__).resolve().parent
CKPT=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
MANIFEST=ROOT/'experiments/tr0_fresh_cv_manifest_20261018/manifest.csv'
ADAPTER=EXP/'adapter_state.pt'; OUT=EXP/'audio'
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); sys.path.insert(0,str(ROOT))
def sha256(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def load_model():
 from sgmse.joint_data_module import JointWaveDataModule
 from sgmse.joint_model import JointSemambaSgmseModel
 m=JointSemambaSgmseModel.load_from_checkpoint(str(CKPT),map_location='cpu',strict=False,data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False,backend_adapter_enable=True,backend_adapter_module=33)
 m.eval(no_ema=False).to(DEVICE); m.dnn.backend_adapter.eval(); return m
def infer(m,path,seed):
 wav,sr=sf.read(path,dtype='float32',always_2d=False); y=torch.from_numpy(np.asarray(wav,dtype=np.float32)).to(DEVICE)[None,None,:]
 norm=y.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)
 with torch.inference_mode(): pre=m._frontend_forward(y/norm)*norm
 cur=pre.shape[-1]; ref=y.shape[-1]
 if cur>ref: pre=pre[...,(cur-ref)//2:(cur-ref)//2+ref]
 elif cur<ref: pre=torch.nn.functional.pad(pre,( (ref-cur)//2, ref-cur-(ref-cur)//2))
 torch.manual_seed(seed)
 if DEVICE.type=='cuda': torch.cuda.manual_seed(seed)
 with torch.inference_mode(): out=m.enhance(pre.squeeze(0),y_cond=y.squeeze(0),sampler_type='pc',predictor='reverse_diffusion',corrector='none',N=30,snr=0.5)
 return np.asarray(out,dtype=np.float32),sr
def main():
 OUT.joinpath('baseline').mkdir(parents=True,exist_ok=True); OUT.joinpath('adapter').mkdir(parents=True,exist_ok=True)
 with MANIFEST.open(newline='',encoding='utf-8') as f: rows=list(csv.DictReader(f))
 m=load_model(); state=torch.load(ADAPTER,weights_only=False)['adapter_state_dict']; start=time.perf_counter(); records=[]
 for i,row in enumerate(rows[:8]):
  name=Path(row['noisy']).name; seed=20261018+i
  m.dnn.backend_adapter.load_state_dict({k:torch.zeros_like(v) for k,v in state.items()})
  base,sr=infer(m,row['noisy'],seed); sf.write(OUT/'baseline'/name,np.clip(base,-1,1),sr)
  m.dnn.backend_adapter.load_state_dict(state); adapted,_=infer(m,row['noisy'],seed); sf.write(OUT/'adapter'/name,np.clip(adapted,-1,1),sr)
  records.append({'filename':name,'uttid':row['uttid'],'noisy_path':row['noisy'],'clean_path':row['clean'],'baseline_path':str(OUT/'baseline'/name),'adapter_path':str(OUT/'adapter'/name),'baseline_sha256':sha256(OUT/'baseline'/name),'adapter_sha256':sha256(OUT/'adapter'/name)})
  print(i+1,name,flush=True)
 payload={'experiment_id':'rollout_adapter_fresh_cv_audio_20261018','status':'completed','device':str(DEVICE),'checkpoint_sha256':sha256(CKPT),'adapter_sha256':sha256(ADAPTER),'n_utterances':len(records),'runtime_seconds':time.perf_counter()-start,'records':records,'protocol':'same PC/N30/seed per file, exact zero adapter versus trained stable adapter'}
 (EXP/'fresh_cv_audio_manifest.json').write_text(json.dumps(payload,indent=2)+'\n')
if __name__=='__main__': main()
