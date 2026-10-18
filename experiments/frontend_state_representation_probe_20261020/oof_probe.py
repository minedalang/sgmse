import json,time,sys,csv
from pathlib import Path
import numpy as np,torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.stats import spearmanr
ROOT=Path('/data/coding/sgmse');E=ROOT/'experiments/frontend_state_representation_probe_20261020';CACHE=ROOT/'experiments/rollout_adapter_replication_20261018/rollout_state_cache.pt';CK=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt';sys.path.insert(0,str(ROOT))
from sgmse.joint_data_module import JointWaveDataModule
from sgmse.joint_model import JointSemambaSgmseModel
items=torch.load(CACHE,weights_only=False); D=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); m=JointSemambaSgmseModel.load_from_checkpoint(str(CK),map_location='cpu',strict=False,data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False);m.eval(no_ema=False).to(D)
vals=[]
with torch.inference_mode():
 for it in items:
  x=it['x_t'].to(D); ym=it['y_main'].to(D); yc=it['y_cond'].to(D); t=it['t'].to(D); target=it['x_clean'].to(D); score=m(x,ym,t,y_cond=yc); x0=m._rrsd_score_to_x0(x,ym,t,score)[0]; nm=float((torch.abs(x0-target).float().square().mean()/torch.abs(target).float().square().mean().clamp_min(1e-12)).item()); vals.append({'utt_index':it['utt_index'],'step_index':it['step_index'],'nmse':nm})
# aggregate 4 states per utterance
agg={i:[] for i in range(8)}
for v in vals: agg[v['utt_index']].append(v['nmse'])
hooks=json.loads((E/'per_utterance.json').read_text()); X=[];y=[];groups=[]
for i,r in enumerate(hooks):
 feat=[r['snr_db'],r['noisy_clean_cos']]
 # derive noisy hook means/std/rms; clean differences
 for name in ['act','output_layer','pyramid_upsample']:
  hn=r['hook_stats']['noisy']['hooks'][name]; hc=r['hook_stats']['clean']['hooks'][name]; feat += [hn[1],hn[2],hc[1]-hn[1],hc[2]-hn[2]]
 feat += [r['hook_stats']['noisy']['rms']]
 X.append(feat); y.append(np.mean(agg[i])); groups.append(r['speaker'])
X=np.asarray(X);y=np.asarray(y); pred=np.full(8,np.nan); alphas=[1e-3,1e-2,1e-1,1,10,100]
for g in sorted(set(groups)):
 tr=np.array([x!=g for x in groups]);te=~tr; mu=X[tr].mean(0);sd=X[tr].std(0);sd[sd<1e-8]=1; model=Ridge(alpha=1.0).fit((X[tr]-mu)/sd,y[tr]);pred[te]=model.predict((X[te]-mu)/sd)
base=np.full(8,y.mean()); mae=float(np.mean(np.abs(pred-y))); bmae=float(np.mean(np.abs(base-y))); r2=float(r2_score(y,pred)); spr=float(spearmanr(y,pred).statistic)
out={'oof_n':8,'speakers':4,'target':'4-state mean clean-x0 NMSE','baseline_mean_nmse':float(y.mean()),'ridge_loso_oof':{'mae':mae,'r2':r2,'spearman':spr,'baseline_mae':bmae,'predictions':pred.tolist(),'targets':y.tolist()},'limitations':['exploratory low power: 4 speakers and 8 utterances','features include descriptive hook pooled statistics; no diffusion-time feature','same cache/manifest alignment verified by utt_index only']}
(E/'oof_metrics.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
