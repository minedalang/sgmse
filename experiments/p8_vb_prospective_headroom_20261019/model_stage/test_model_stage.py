"""Synthetic/mock-only observer and scoring tests, never real model/audio inputs."""
from pathlib import Path
import ast,importlib.util,itertools,json,sys,tempfile,types
from helpers import module,observe,sha,transitions,summarize
STAGE=Path(__file__).resolve().parent
for p in STAGE.glob('*.py'):ast.parse(p.read_text())
# args/kwargs and return identity (including sampler tuple) are preserved.
x=object();y=object();ret=(object(),30);seen=[]
def fn(*a,**kw):assert a==(x,) and kw['y'] is y;return ret
wrapped=observe(fn,lambda a,k:seen.append((a,k)),lambda r,a,k:seen.append(r))
assert wrapped(x,y=y) is ret and seen[-1] is ret
cal=module('cal_score_test',STAGE.parent.parent/'p8_voicebank_clean_text_calibration_20261019/run_calibration.py');ind=module('ind_score_test',STAGE.parent.parent/'p8_voicebank_clean_text_calibration_20261019/validate_completed_run.py')
seq=[list(x) for n in range(4) for x in itertools.product(['a','b'],repeat=n)]
for ref in seq:
 for hyp in seq:assert cal.alignment(ref,hyp)==ind.independent_align(ref,hyp)
f=cal.score('a b c d','a x c');b=cal.score('a b c d','x b c z extra');t=transitions(f,b);assert sum(t['counts'].values())==4 and t['frontend_insertions']==f['word']['I'] and t['off_insertions']==b['word']['I']
# Fully mocked native model/torch; real numpy+libsndfile handle only generated numeric arrays.
import numpy as np
import soundfile as sf
import run_pilot as run
class Tensor:
 def __init__(self,a):self.a=np.asarray(a,dtype=np.float32)
 @property
 def shape(self):return self.a.shape
 def abs(self):return Tensor(abs(self.a))
 def max(self):return Tensor(self.a.max())
 def clamp_min(self,v):return Tensor(np.maximum(self.a,v))
 def item(self):return self.a.item()
 def detach(self):return self
 def cpu(self):return self
 def numpy(self):return self.a
 def squeeze(self):return Tensor(self.a.squeeze())
class OUVESDE:N=30
sampling=types.ModuleType('sgmse.sampling');sampling.get_pc_sampler=lambda *a,**k:lambda:(None,30)
class ScoreModel:
 def enhance(self,y,**kw):
  norm_factor=kw['y_cond'].abs().max().clamp_min(1e-12).item();T_orig=int(y.shape[-1]);sampler=sampling.get_pc_sampler('reverse_diffusion','none',sde=OUVESDE(),snr=.5);sampler();return y.numpy().copy()
class Model(ScoreModel):
 training=True;train_type='mode2';residual_refine_enable=False;hparams=dict(frontend='storm',frontend_ckpt=None)
 @classmethod
 def load_from_checkpoint(cls,*a,**k):return cls()
 def eval(self,**kw):self.training=False;return self
 def _frontend_forward(self,x):return x
native=types.ModuleType('joint_enhance_by_dir');native.JointSemambaSgmseModel=Model;native.load_mono=lambda p,*a:Tensor(np.array([.25,-.4,.1,.0],np.float32));native._match_length=lambda a,b:a
fake_torch=types.ModuleType('torch');fake_torch.is_tensor=lambda x:isinstance(x,Tensor);fake_torch.initial_seed=lambda:seed[0];fake_torch.cuda=types.SimpleNamespace(synchronize=lambda:None,empty_cache=lambda:None,memory_allocated=lambda:0,memory_reserved=lambda:0)
seed=[0]
def main():
 args=sys.argv;folder=Path(args[args.index('--in_dir')+1]);out=Path(args[args.index('--out_dir')+1]);model=Model.load_from_checkpoint('mock.ckpt',strict=True,weights_only=False);model.eval(no_ema=False)
 for p in sorted(folder.glob('*.wav')):
  seed[0]=next(x['seed'] for x in registry['cases'] if x['basename']==p.name);y=native.load_mono(p,16000);pre=model._frontend_forward(Tensor(y.a.reshape(1,1,-1)));pre=native._match_length(pre,Tensor(y.a.reshape(1,1,-1)));result=model.enhance(pre.squeeze(),y_cond=y,**expected);sf.write(out/p.name,np.clip(result,-1.,1.),16000)
native.main=main
mods={'torch':fake_torch,'joint_enhance_by_dir':native,'sgmse':types.ModuleType('sgmse'),'sgmse.sampling':sampling,'sgmse.model':types.ModuleType('sgmse.model')};mods['sgmse'].__path__=[];mods['sgmse.model'].ScoreModel=ScoreModel
old={k:sys.modules.get(k) for k in mods};sys.modules.update(mods)
expected=dict(sampler_type='pc',predictor='reverse_diffusion',corrector='none',N=30,snr=.5,rrsd=False,anchor_calib=False,anchor_v2=False,regen_strength=None)
try:
 with tempfile.TemporaryDirectory() as temp:
  run.OUT=Path(temp);run.INPUT=run.OUT;(run.OUT/'outputs').mkdir();(run.OUT/'inputs').mkdir();registry=dict(cases=[])
  for i in range(32):
   name='synthetic_'+str(i).zfill(2)+'.wav';path=run.OUT/'inputs'/name;path.write_bytes(b'mock-source-not-audio');registry['cases'].append(dict(basename=name,id='mock',speaker='mock',seed=i+10,wav=str(path.relative_to(run.OUT)),wav_sha256=sha(path),samples=4))
  d=dict(frontend_calls=0,backend_calls=0,enhancement_model_loads=0,native_cases=[]);run.enhance_native(dict(checkpoint='mock.ckpt',expected_enhance_kwargs=expected),registry,d,lambda:None)
  assert len({id(x) for x in d['native_cases']})==32 and len(d['native_cases'])==32
  assert d['frontend_calls']==d['backend_calls']==32 and d['enhancement_release']['model_weakrefs_dead']
  assert all(x['nfe']==30 and x['backend_actual_norm_factor']==x['noisy_peak_native_reference'] for x in d['native_cases'])
finally:
 for k,v in old.items():
  if v is None:sys.modules.pop(k,None)
  else:sys.modules[k]=v
print(json.dumps(dict(status='passed',native_mock_cases=32,observer_args_returns_identity=True,model_release_weakref=True,independent_alignment_pairs=len(seq)**2,synthetic_serialization_pairs=64,real_model_loads=0,real_audio_decodes=0,network_bytes=0)))
# Weight stream observer accepts actual Lightning local file objects, preserves position.
from helpers import weight_path,allow_local_build
import io,os
with tempfile.TemporaryDirectory() as tmp:
 p=Path(tmp)/'mock.ckpt';p.write_bytes(b'not-real-weights');locked={str(p.resolve()):sha(p)}
 with p.open('rb') as f:
  f.seek(3);assert weight_path(f,locked)==str(p.resolve()) and f.tell()==3
 assert weight_path(p,locked)==str(p.resolve())
 try:weight_path(io.BytesIO(b'unknown'),locked)
 except RuntimeError:pass
 else:raise AssertionError('Unknown stream accepted')
 cfg=dict(local_build_executables=['/usr/bin/true'],native_build_directories=[tmp])
 try:allow_local_build('/usr/bin/curl',['/usr/bin/curl','https://example.invalid'],tmp,cfg)
 except RuntimeError:pass
 else:raise AssertionError('Network command accepted')
print(json.dumps(dict(weight_filelike_mock='passed',position_preserved=True,unknown_filelike_rejected=True,network_process_rejected=True)))
