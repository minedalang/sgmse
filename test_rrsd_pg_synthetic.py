"""Tiny synthetic smoke-test: verifies backprop and positive FD signs without a GPU/model downloads."""
import importlib.util
import os
from types import SimpleNamespace
import torch
from pathlib import Path
from unittest.mock import patch

src=Path(__file__).parent/'rrsd_pg_agreement_probe.py'
spec=importlib.util.spec_from_file_location('pgdiag',src)
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FakeWavLM(torch.nn.Module):
    def forward(self,input_values,output_hidden_states=True,return_dict=True):
        # Delay line that keeps sample/time structure for differentiability.
        z=torch.stack((input_values,2*input_values),dim=-1)
        return SimpleNamespace(hidden_states=[z,z,z,z])
class Model:
    sr=16000; rrds=0
    rrsd_lookahead_dt=0.03; t_eps=0.01; rrsd_oracle_eps=1e-6
    data_module=SimpleNamespace(num_frames=9,hop_length=2)
    sde=SimpleNamespace(T=1.,N=30)
    def _score_step_geometry(self,s):
        return s['score'],s['x'],s['y'],s['t'],s['sigma'],s['wx'],None,s['x0_base']
    def _rrsd_reverse_mean_step(self,*,x,y_main,t,score,dt):
        return x + (dt[:,None,None,None]*score),t-dt
    def __call__(self,x,y,t,*,y_cond):
        return -.1*x
    def _rrsd_score_to_x0(self,x,y,t,score):
        return x+score, None,None,None
    def to_audio(self,spec,length):
        # Real/reversible complex representation: use real+imag, flatten.
        v=(spec.real+spec.imag).flatten()
        if v.numel()<length:v=torch.nn.functional.pad(v,(0,length-v.numel()))
        return v[:length]

torch.manual_seed(3)
B=2; x=torch.randn(B,1,4,5).to(torch.cfloat)
target=torch.randn_like(x)*.2
s={ 'score': torch.ones_like(x)*(.3+.2j), 'x': x, 'y':torch.zeros_like(x),
    't':torch.tensor([.15,.37]),'sigma': torch.ones(B,1,1,1)*.8,
    'wx':torch.ones(B,1,1,1), 'x0_base':x }
with patch.dict(os.environ,{'PG_MAX_SAMPLES':'16','PG_FD_BATCHES':'5','PG_FD_EPS':'0.01','PG_T_MIN':'0.05','PG_T_MAX':'0.5'}):
    with patch.object(m,'_encoder',return_value=FakeWavLM()):
        m._run_probe(Model(),target=target,x0_front=x+torch.randn_like(x)*.4,
              step_out=s,pred={'alpha_tile':torch.rand(B,1,2,2)*.05},batch_num=1,max_batches=2)
print('Synthetic smoke test OK')
