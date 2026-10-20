"""Single authorized zero-weight import debug; no models/tensors/audio inference."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,os,signal,subprocess,sys,time,traceback
A=Path(__file__).resolve().parent;S=A/'model_stage';BASE=A.parent
sys.path.insert(0,str(S));from helpers import read,write,verify,sha,module
from process_guard import allow_local_build

def size():return sum(p.stat().st_size for p in A.rglob('*') if p.is_file() and not p.is_symlink())
def child():
 d=dict(status='running',weight_load_attempts=0,model_load_attempts=0,model_forward_attempts=0,network_attempts=[],network_bytes=0,subprocesses=[],imports=[],scientific_calls=0)
 def snapshot():write(A/'smoke_progress.json',d)
 def audit(event,args):
  if event=='subprocess.Popen':
   record=dict(original_executable=str(args[0]),argv=args[1],cwd=None if args[2] is None else str(args[2]));d['subprocesses'].append(record);snapshot();record.update(allow_local_build(args[0],args[1],args[2],cfg));snapshot()
  if event in ['socket.connect','socket.getaddrinfo','socket.sendto','socket.bind','os.system','os.posix_spawn']:raise RuntimeError('Smoke network/shell forbidden '+event)
 try:
  cfg=read(S/'execution_config.json');d['dependency_pre']=verify(read(A/'dependency_source_locks.json'));d['old_pre']=verify(read(A/'original_preservation_locks.json'))
  cal=module('frozen_calibration_smoke',cfg['calibration_driver']);cal.offline_guard(d['network_attempts']);sys.addaudithook(audit)
  snapshot();import torch;d['imports'].append(dict(name='torch',version=torch.__version__,source=torch.__file__))
  def no_weight(*a,**k):d['weight_load_attempts']+=1;raise RuntimeError('Smoke forbids checkpoint loading')
  def no_model(*a,**k):d['model_load_attempts']+=1;raise RuntimeError('Smoke forbids model construction/load')
  def no_forward(*a,**k):d['model_forward_attempts']+=1;raise RuntimeError('Smoke forbids tensor/model inference')
  torch.load=no_weight;torch.serialization.load=no_weight;torch.jit.load=no_weight;torch.nn.Module.__init__=no_model;torch.nn.Module._call_impl=no_forward
  snapshot();sys.path.insert(0,'/data/coding/sgmse');import joint_enhance_by_dir as native;native.JointSemambaSgmseModel.load_from_checkpoint=no_model;d['imports'].append(dict(name='joint_enhance_by_dir',source=native.__file__))
  snapshot();import whisper;whisper.load_model=no_model;d['imports'].append(dict(name='whisper',version=whisper.__version__,source=whisper.__file__))
  d['cached_extension_modules']={name:getattr(mod,'__file__',None) for name,mod in sys.modules.items() if name=='upfirdn2d'};d['status']='passed';snapshot()
 except BaseException as e:d['status']='failed';d['error']=repr(e);d['traceback']=traceback.format_exc();traceback.print_exc()
 finally:
  try:d['dependency_post']=verify(read(A/'dependency_source_locks.json'));d['old_post']=verify(read(A/'original_preservation_locks.json'))
  except BaseException as e:d['status']='failed';d['post_error']=repr(e)
  write(A/'smoke_child_result.json',d)
 return 0 if d['status']=='passed' else 1

def supervise():
 start=time.monotonic()
 with (A/'IMPORT_SMOKE_STARTED.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),limit_seconds=60,limit_log_bytes=5000000,scope='zero-weight dependency import only'),f)
 cfg=read(S/'execution_config.json');env=os.environ.copy();env.update(PATH=cfg['native_path'],PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',NUMBA_CACHE_DIR=str(A/'smoke_numba_cache'))
 reason=None
 with (A/'import_smoke.log').open('x') as log:
  p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],cwd='/data/coding/sgmse',env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   if time.monotonic()-start>=58 or (A/'import_smoke.log').stat().st_size>=4_900_000:
    reason='60s/5MB smoke bound reserve';os.killpg(p.pid,signal.SIGKILL);p.wait();break
   time.sleep(.05)
 d=read(A/'smoke_child_result.json') if (A/'smoke_child_result.json').exists() else read(A/'smoke_progress.json') if (A/'smoke_progress.json').exists() else dict(status='failed')
 if reason or p.returncode:d['status']='failed';d['supervisor_error']=reason or 'child exit '+str(p.returncode)
 d.update(seconds=time.monotonic()-start,child_exit=p.returncode,log_bytes=(A/'import_smoke.log').stat().st_size,actual_scientific_model_calls=0);write(A/'import_smoke_metrics.json',d);print(json.dumps({k:d.get(k) for k in ['status','seconds','log_bytes','weight_load_attempts','model_load_attempts','model_forward_attempts','imports','subprocesses','error']}));return 0 if d['status']=='passed' else 1
if __name__=='__main__':sys.exit(child() if sys.argv[1:]==['--child'] else supervise())
