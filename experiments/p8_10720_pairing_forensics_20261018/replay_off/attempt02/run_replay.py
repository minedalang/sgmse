"""One bounded forensic replay; read-only probes wrap existing inference entry point."""
from pathlib import Path
import sys,os,json,hashlib,time,datetime,subprocess,traceback,io
ROOT=Path('/data/coding/sgmse'); OUT=Path(__file__).resolve().parent
if (OUT/'metrics.json').exists():raise RuntimeError('Refuse repeated replay')
sys.path.insert(0,str(ROOT));os.chdir(ROOT)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
original=json.loads((OUT.parent.parent/'metrics.json').read_text());src=Path(original['sources']['noisy']['path']);saved=Path(original['sources']['off']['path']);name=saved.name
ckpt=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
inp=OUT/'inputs';out=OUT/'output';inp.mkdir();out.mkdir();(inp/name).symlink_to(src)
argv=['joint_enhance_by_dir.py','--ckpt',str(ckpt),'--in_dir',str(inp),'--out_dir',str(out),'--device','cuda:0','--sample_rate','16000','--train_type','mode2','--sampler_type','pc','--N','30','--corrector','none','--snr','0.5','--seed','2026','--use_ema']
files=['joint_enhance_by_dir.py','sgmse/joint_model.py','sgmse/joint_data_module.py','sgmse/model.py','sgmse/frontends/storm.py']
d={'status':'running','purpose':'One explicitly authorized forensic reproduction to test causal input/output pairing; not a new research experiment','started_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'input_path':str(src),'input_symlink':str(inp/name),'input_sha256':sha(src),'saved_output_path':str(saved),'saved_output_sha256':sha(saved),'checkpoint_path':str(ckpt),'checkpoint_sha256':sha(ckpt),'script_sha256':{f:sha(ROOT/f) for f in files},'replay_wrapper_sha256':sha(__file__),'argv':argv,'literal_historical_command_available':False,'command_provenance':'Reconstructed from original config.yaml, off_retry.log and report; exact literal historical command/script hashes unavailable','effective_per_file_seed':(2026+int.from_bytes(hashlib.blake2b(name.encode(),digest_size=4).digest(),'little'))%2**32,'git_head':subprocess.getoutput('git rev-parse HEAD'),'git_status':subprocess.getoutput('git status --short'),'path':os.environ.get('PATH'),'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),'gpu_inventory':subprocess.getoutput('nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.used,memory.total --format=csv'),'failures':[]}
(OUT/'config.yaml').write_text('experiment_id: p8_10720_pairing_forensics_20261018_replay_off\npurpose: single_authorized_causal_provenance_replay\ncheckpoint: '+str(ckpt)+'\nuse_ema: true\nsampler: pc\nN: 30\ncorrector: none\nsnr: 0.5\nseed: 2026\nrrsd: false\ntrain_type: mode2\ndevice: cuda:0\ninput_basename: '+name+'\n')
(OUT/'launch_provenance.json').write_text(json.dumps(d,indent=2)+'\n')
t0=time.monotonic()
try:
 import torch,numpy as np,soundfile as sf
 import joint_enhance_by_dir as mod
 d['environment']={'python':sys.version,'torch':torch.__version__,'cuda_runtime':torch.version.cuda,'cudnn_version':torch.backends.cudnn.version(),'device_name':torch.cuda.get_device_name(0),'device_properties':str(torch.cuda.get_device_properties(0)),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark}
 def compare(a,b):
  a=np.asarray(a,dtype=np.float64).reshape(-1);b=np.asarray(b,dtype=np.float64).reshape(-1);delta=a-b
  return dict(samples=len(a),exact_pcm_equality=bool(np.array_equal(a,b)),max_absolute_difference=float(np.max(abs(delta))),difference_rms=float(np.sqrt(np.mean(delta*delta))),correlation=float(np.corrcoef(a,b)[0,1]),replay_rms=float(np.sqrt(np.mean(a*a))),saved_rms=float(np.sqrt(np.mean(b*b))))
 oldmatch=mod._match_length
 def match(a,b):
  z=oldmatch(a,b);v=z.detach().cpu().numpy().reshape(-1);bio=io.BytesIO();sf.write(bio,v,16000,format='WAV',subtype='PCM_16');quantized=sf.read(io.BytesIO(bio.getvalue()))[0];ref=sf.read(original['sources']['y_pre']['path'])[0]
  d['frontend']={'before_match_samples':int(a.shape[-1]),'after_match_samples':int(z.shape[-1]),'noisy_samples':int(b.shape[-1]),'float32_sha256':hashlib.sha256(v.tobytes()).hexdigest(),'pcm16_sha256':hashlib.sha256(bio.getvalue()).hexdigest(),'comparison_after_pcm16_quantization':compare(quantized,ref)};return z
 mod._match_length=match
 oldenhance=mod.JointSemambaSgmseModel.enhance
 def enhance(self,*args,**kwargs):
  d['loaded_model']={'train_type':self.train_type,'frontend':str(getattr(self,'frontend_name',getattr(self,'frontend',None))),'residual_refine_enable':getattr(self,'residual_refine_enable',None)}
  d['enhance_kwargs']=kwargs;tt=time.monotonic();kwargs['timeit']=True
  y,nfe,rtf=oldenhance(self,*args,**kwargs);torch.cuda.synchronize();d['nfe']=int(nfe);d['model_reported_rtf']=float(rtf);d['enhance_wall_seconds']=time.monotonic()-tt;return y
 mod.JointSemambaSgmseModel.enhance=enhance
 sys.argv=argv;mod.main()
 target=out/name;a,sr=sf.read(target);b,bsr=sf.read(saved);d['output_path']=str(target);d['output_sha256']=sha(target);d['output_bytes']=target.stat().st_size;d['byte_hash_matches_saved']=d['output_sha256']==d['saved_output_sha256'];d['sample_rates']=[sr,bsr];d['full_comparison']=compare(a,b);d['clip_13p75_15p75_comparison']=compare(a[220000:252000],b[220000:252000]);d['peak_cuda_memory_bytes']=torch.cuda.max_memory_allocated(0);d['status']='completed'
except Exception as exc:
 d['status']='failed';d['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});traceback.print_exc()
d['wall_seconds']=time.monotonic()-t0;d['completed_at_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();d['post_run_input_sha256']=sha(src);d['post_run_saved_output_sha256']=sha(saved);d['post_run_checkpoint_sha256']=sha(ckpt)
(OUT/'metrics.json').write_text(json.dumps(d,indent=2,default=str)+'\n');print(json.dumps({k:d.get(k) for k in ['status','wall_seconds','byte_hash_matches_saved','full_comparison','clip_13p75_15p75_comparison','failures']},indent=2))
