#!/usr/bin/env python3
"""Stable-step follow-up after the deliberately bounded 1e-3 pilot failed."""
from __future__ import annotations
import json, sys, time
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
EXP = Path(__file__).resolve().parent
CKPT = ROOT / "joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt"
CACHE = EXP / "rollout_state_cache.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.insert(0, str(ROOT))

def nmse(model, item):
    x=item["x_t"].to(DEVICE); y=item["y_main"].to(DEVICE); c=item["y_cond"].to(DEVICE); t=item["t"].to(DEVICE); target=item["x_clean"].to(DEVICE)
    score=model(x,y,t,y_cond=c); x0=model._rrsd_score_to_x0(x,y,t,score)[0]
    return torch.abs(x0-target).float().square().mean()/torch.abs(target).float().square().mean().clamp_min(1e-12)

def evaluate(model, items):
    model.dnn.backend_adapter.eval()
    with torch.inference_mode(): vals=[float(nmse(model,x)) for x in items]
    model.dnn.backend_adapter.train()
    return float(np.mean(vals)), vals

def main():
    from sgmse.joint_data_module import JointWaveDataModule
    from sgmse.joint_model import JointSemambaSgmseModel
    start=time.perf_counter(); items=torch.load(CACHE,weights_only=False)
    model=JointSemambaSgmseModel.load_from_checkpoint(str(CKPT),map_location="cpu",strict=False,data_module_cls=JointWaveDataModule,base_dir=".",batch_size=1,num_workers=0,dummy=True,weights_only=False,backend_adapter_enable=True,backend_adapter_module=33)
    model.eval(no_ema=False).to(DEVICE)
    for p in model.parameters(): p.requires_grad=False
    for p in model.dnn.backend_adapter.parameters(): p.requires_grad=True
    model.dnn.backend_adapter.train()
    train=[x for x in items if x["utt_index"]<6]; hold=[x for x in items if x["utt_index"]>=6]
    base_train,_=evaluate(model,train); base_hold,hold_before=evaluate(model,hold)
    opt=torch.optim.Adam(model.dnn.backend_adapter.parameters(),lr=1e-5)
    gen=torch.Generator().manual_seed(20261019); history=[]
    for step in range(100):
        item=train[int(torch.randint(len(train),(1,),generator=gen).item())]
        opt.zero_grad(set_to_none=True); loss=nmse(model,item); loss.backward()
        grad=torch.nn.utils.clip_grad_norm_(model.dnn.backend_adapter.parameters(),1.0); opt.step()
        if step in (0,9,49,99): history.append({"step":step+1,"loss":float(loss.detach()),"grad_norm_before_clip":float(grad)})
    trained_train,_=evaluate(model,train); trained_hold,hold_after=evaluate(model,hold)
    torch.save({"adapter_state_dict": model.dnn.backend_adapter.state_dict(), "base_checkpoint": str(CKPT), "lr": 1e-5, "steps": 100, "gradient_clip": 1.0}, EXP / "stable_adapter_state.pt")
    payload={"experiment_id":"rollout_adapter_stable_pilot_20261018","status":"completed","device":str(DEVICE),"lr":1e-5,"steps":100,"gradient_clip":1.0,"baseline":{"train_nmse":base_train,"holdout_nmse":base_hold},"trained":{"train_nmse":trained_train,"holdout_nmse":trained_hold},"holdout_before":hold_before,"holdout_after":hold_after,"holdout_relative_change":(trained_hold-base_hold)/max(base_hold,1e-12),"history":history,"runtime_seconds":time.perf_counter()-start,"interpretation":"Follow-up stability check after the 1e-3 pilot. This remains a clean x0 proxy only; no final audio metric or test-set inference."}
    (EXP/"stable_metrics.json").write_text(json.dumps(payload,indent=2)+"\n")
    print(json.dumps({k:payload[k] for k in ('baseline','trained','holdout_relative_change','history')},indent=2))
if __name__=='__main__': main()
