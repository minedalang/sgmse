"""Two additional fixed-seed repeats for the P8-M target mask."""
from pathlib import Path
import sys, hashlib, json, time
import numpy as np, soundfile as sf, torch
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from joint_ablation_by_dir import load_mono, _match_length
from sgmse.joint_model import JointSemambaSgmseModel
from sgmse.joint_data_module import JointWaveDataModule
from run_probe import smooth_mask, CKPT, NOISY, SR, OUT

def main():
    model=JointSemambaSgmseModel.load_from_checkpoint(str(CKPT),map_location='cpu',strict=True,
        data_module_cls=JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False)
    model.eval(no_ema=False).to('cuda')
    y=load_mono(str(NOISY),SR).cuda()[None,None,:]
    norm=y.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)
    with torch.inference_mode(): yp=_match_length(model._frontend_forward(y/norm)*norm,y)
    rows=[]
    for seed in (2027,2028):
      for name,masked in (('baseline',False),('target_zero_10.0_10.5',True)):
        ypm=yp.clone()
        if masked: ypm=ypm*smooth_mask(ypm.shape[-1],10.0,10.5,0.0).to(ypm.device)[None,None,:]
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); t=time.time()
        with torch.inference_mode(): out=model.enhance(ypm.squeeze(0),y_cond=y.squeeze(0),sampler_type='pc',predictor='reverse_diffusion',corrector='none',N=30,snr=0.5)
        out=np.clip(out,-1,1).astype(np.float32); p=OUT/f'{name}_seed{seed}.wav'; sf.write(p,out,SR)
        rows.append({'seed':seed,'arm':name,'runtime_s':time.time()-t,'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    p=Path(__file__).resolve().parent/'multiseed_metrics.json'; p.write_text(json.dumps({'seeds':[2027,2028],'arms':rows,'interpretation':'repeatability only; no WER claim','dataset_modified':False,'checkpoint_modified':False},indent=2))
if __name__=='__main__': main()
