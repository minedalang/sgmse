"""One fixed dev crop, CPU source/rounding diagnosis; no fitting or model."""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import hashlib,json,signal,time,socket
from pathlib import Path
import numpy as np
import soundfile as sf

D=Path(__file__).resolve().parent
cfg=json.loads((D/'config.yaml').read_text())
if (D/'metrics.json').exists() or (D/'STARTED.json').exists():
    raise RuntimeError('Exclusive single execution; no retry')
begin=time.monotonic()
def timeout(*args):raise TimeoutError('30-second limit')
signal.signal(signal.SIGALRM,timeout);signal.alarm(30)
def deny(*args,**kwargs):raise RuntimeError('Network forbidden')
socket.socket.connect=deny;socket.socket.connect_ex=deny
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
out={'status':'running','pid':os.getpid(),'sources':{},'failures':[]}
with (D/'STARTED.json').open('x') as f:json.dump({'pid':os.getpid()},f)
try:
    out['source_sha_pre']={k:sha(p) for k,p in cfg['sources'].items()}
    arrays={};known={'PCM_U8':8,'PCM_S8':8,'PCM_16':16,'PCM_24':24,'PCM_32':32}
    for role,path in cfg['sources'].items():
        info=sf.info(path);bits=known.get(info.subtype)
        a,rate=sf.read(path,start=cfg['crop_start'],frames=cfg['crop_samples'],dtype='float64',always_2d=True)
        if a.shape!=(cfg['crop_samples'],1) or rate!=16000 or not np.isfinite(a).all():
            raise RuntimeError('Unexpected finite mono16k crop geometry')
        a=a[:,0];arrays[role]=a
        q=None if bits is None else 2.**(1-bits)
        out['sources'][role]={'path':path,'format':info.format,'format_info':info.format_info,'subtype':info.subtype,'subtype_info':info.subtype_info,'bits':bits,'samplerate':info.samplerate,'channels':info.channels,'source_frames':info.frames,'crop_samples':len(a),'quantum':q,'crop_min':float(a.min()),'crop_max':float(a.max()),'negative_endpoint_count':None if q is None else int(np.count_nonzero(a==-1.)),'positive_endpoint_count':None if q is None else int(np.count_nonzero(a==1.-q)),'endpoint_scope':'Selected crop only; endpoint equality indicates possible saturation, not proof of clipping.'}
    x,s,n=(arrays[k] for k in ('X','S','N'));r=x-s-n
    xrms=float(np.sqrt(np.mean(x*x)));rrms=float(np.sqrt(np.mean(r*r)))
    out['residual']={'raw_rms':rrms,'raw_max_abs':float(np.max(np.abs(r))),'X_raw_rms':xrms,'relative_rms':None if xrms==0 else rrms/xrms,'signed_quantiles':dict(zip(map(str,cfg['quantiles']),map(float,np.quantile(r,cfg['quantiles'])))),'absolute_quantiles':dict(zip(map(str,cfg['quantiles']),map(float,np.quantile(np.abs(r),cfg['quantiles']))))}
    qs=[out['sources'][k]['quantum'] for k in ('X','S','N')]
    if all(q is not None for q in qs):
        bound=.5*sum(qs);outside=np.abs(r)>bound+cfg['roundoff_tolerance'];count=int(outside.sum())
        out['rounding_bound']={'status':'known_integer_PCM_conditional_bound','worst_bound':bound,'numeric_tolerance':cfg['roundoff_tolerance'],'exceeding_samples':count,'total_samples':len(r),'exceeding_fraction':count/len(r),'max_excess_over_bound':float(np.maximum(np.abs(r)-bound,0).max()),'interpretation':'compatible_with_final_nearest_rounding_only' if count==0 else 'contradicts_exact_mixture_plus_final_nearest_rounding_only','qualification':'Conditional consistency only, not proof of noise purity or original synthesis identity; clipping or other processing invalidates premise.'}
    else:out['rounding_bound']={'status':'unknown_noninteger_or_unknown_bitdepth','interpretation':'no bound inferred'}
    out['status']='completed'
except BaseException as e:
    import traceback
    out['status']='failed';out['failures'].append({'error':repr(e),'traceback':traceback.format_exc()})
finally:
    out['source_sha_post']={k:sha(p) for k,p in cfg['sources'].items()}
    out['source_unchanged']=out.get('source_sha_pre')==out['source_sha_post']
    if not out['source_unchanged']:out['status']='failed'
    out.update(seconds=time.monotonic()-begin,code_sha256=sha(__file__),config_sha256=sha(D/'config.yaml'),numpy_version=np.__version__,soundfile_version=sf.__version__)
    with (D/'metrics.json').open('x') as f:json.dump(out,f,indent=2,allow_nan=False)
    signal.alarm(0)
print(json.dumps(out))
raise SystemExit(0 if out['status']=='completed' else 1)
