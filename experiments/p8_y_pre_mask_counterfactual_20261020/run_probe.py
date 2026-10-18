"""Small P8-M counterfactual probe; never writes to the dataset."""
from pathlib import Path
import sys
import csv, hashlib, json, time
import numpy as np
import soundfile as sf
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from joint_ablation_by_dir import load_mono, _match_length
from sgmse.joint_model import JointSemambaSgmseModel
from sgmse.joint_data_module import JointWaveDataModule

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
NOISY = Path("/data/datasets/dns_ins20/dns_snr_-20_-5/synthetic_snr-20_-5/noisy/book_02324_chp_0015_reader_10658_7_CGPxaXQVd_U-fan_Freesound_validated_350631_0-oHvYwexeY6k_snr-19_fileid_5243.wav")
SR = 16000
SEED = 2026

def smooth_mask(n, start, end, scale, edge=0.04):
    """Time-domain smooth attenuation mask, with cosine ramps."""
    t = torch.arange(n, dtype=torch.float32) / SR
    m = torch.ones(n)
    a, b = float(start), float(end)
    e = min(edge, max((b-a)/4, 1/SR))
    left = (t >= a) & (t < a+e)
    hold = (t >= a+e) & (t <= b-e)
    right = (t > b-e) & (t <= b)
    m[hold] = scale
    if left.any():
        u = (t[left]-a)/e
        m[left] = 1 - (1-scale)*0.5*(1-torch.cos(torch.pi*u))
    if right.any():
        u = (t[right]-(b-e))/e
        m[right] = scale + (1-scale)*0.5*(1-torch.cos(torch.pi*u))
    m[t > b] = 1
    return m

def main():
    model = JointSemambaSgmseModel.load_from_checkpoint(
        str(CKPT), map_location="cpu", strict=True, data_module_cls=JointWaveDataModule,
        base_dir=".", batch_size=1, num_workers=0, dummy=True, weights_only=False)
    model.eval(no_ema=False).to("cuda")
    y = load_mono(str(NOISY), SR).cuda()[None, None, :]
    norm = y.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    with torch.inference_mode():
        y_pre = _match_length(model._frontend_forward(y/norm)*norm, y)
    arms = {
        "baseline": None,
        "target_zero_10.0_10.5": (10.0, 10.5, 0.0),
        "target_half_10.0_10.5": (10.0, 10.5, 0.5),
        "negative_zero_16.5_17.0": (16.5, 17.0, 0.0),
    }
    records=[]
    for name, spec in arms.items():
        yp = y_pre.clone()
        if spec:
            a,b,s=spec
            yp = yp * smooth_mask(yp.shape[-1],a,b,s).to(yp.device)[None,None,:]
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        t0=time.time()
        with torch.inference_mode():
            out=model.enhance(yp.squeeze(0), y_cond=y.squeeze(0), sampler_type="pc",
                              predictor="reverse_diffusion", corrector="none", N=30, snr=0.5)
        out=np.clip(out,-1,1).astype(np.float32)
        path=OUT/f"{name}.wav"; sf.write(path,out,SR)
        records.append({"arm":name,"mask":spec,"runtime_s":time.time()-t0,
                        "samples":int(out.size),"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    (Path(__file__).resolve().parent/"metrics.json").write_text(json.dumps({
        "experiment_id":"p8_y_pre_mask_counterfactual_20261020",
        "checkpoint":str(CKPT),"checkpoint_sha256":hashlib.sha256(CKPT.read_bytes()).hexdigest(),
        "input":str(NOISY),"input_sha256":hashlib.sha256(NOISY.read_bytes()).hexdigest(),
        "seed":SEED,"sampler":"pc","N":30,"snr":0.5,"corrector":"none",
        "arms":records,"dataset_modified":False,"checkpoint_modified":False,
        "interpretation":"OOD counterfactual acoustic probe; no content/WER claim"
    },indent=2))

if __name__ == "__main__": main()
