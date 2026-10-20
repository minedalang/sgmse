"""Frozen 64-row codec diagnostic. No model, checkpoint, fitting or updates."""
import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,hashlib,time,signal,socket,collections
from pathlib import Path
import numpy as np
import soundfile as sf
D=Path(__file__).resolve().parent
cfg=json.loads((D/'config.yaml').read_text())
if (D/'metrics.json').exists() or (D/'execution_receipt.json').exists():raise RuntimeError('No retry')
start=time.monotonic()
def deadline(*args):raise TimeoutError('60-second cap')
signal.signal(signal.SIGALRM,deadline);signal.alarm(60)
def refuse(*args,**kw):raise RuntimeError('Network forbidden')
socket.socket.connect=refuse;socket.socket.connect_ex=refuse
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
reg=Path(cfg['registry']);assert sha(reg)==cfg['registry_sha256']
records=[json.loads(s) for s in reg.read_text().splitlines()];assert len(records)==64
keys={'X':'wav_path','S':'clean_path_loader','N':'noise1_path'}
paths=[r[k] for r in records for k in keys.values()];assert len(set(paths))==192
old={p:sha(p) for p in [cfg['old_diagnosis'],cfg['old_attempt03']]}
receipt={'status':'running','pid':os.getpid(),'code_sha256':sha(__file__),'config_sha256':sha(D/'config.yaml'),'old_metadata_sha_pre':old}
with (D/'execution_receipt.json').open('x') as f:json.dump(receipt,f)
out={'status':'running','registry_sha256':sha(reg),'rows':[],'source_sha':{},'failures':[]}
known={'PCM_16':16,'PCM_24':24,'PCM_32':32}
try:
    for path in paths:out['source_sha'][path]={'pre':sha(path)}
    for index,row in enumerate(records):
        result={'index':index,'utt':row['utt'],'priority_bin':row['priority_bin'],'source_info':{},'strict_pass':None,'nearest_compatible':None,'codec_consistent':None,'processing_chain':'unverified','reasons':[]}
        try:
            infos={role:sf.info(row[key]) for role,key in keys.items()}
            for role,i in infos.items():
                result['source_info'][role]={'format':i.format,'subtype':i.subtype,'bits':known.get(i.subtype),'sr':i.samplerate,'channels':i.channels,'frames':i.frames}
            if any(i.samplerate!=16000 or i.channels!=1 or i.frames!=480000 for i in infos.values()):
                raise ValueError('unknown_geometry: each original source must be mono16k/480000frames')
            center=(480000-32640)//2;result['crop_start']=center
            arrays={}
            for role,key in keys.items():
                v,sr=sf.read(row[key],start=center,frames=32640,dtype='float64',always_2d=True)
                if v.shape!=(32640,1) or sr!=16000 or not np.isfinite(v).all():raise ValueError('unknown_invalid_crop')
                arrays[role]=v[:,0]
                bits=known.get(infos[role].subtype);q=None if bits is None else 2.**(1-bits)
                result['source_info'][role].update(quantum=q,negative_endpoint_count=None if q is None else int(np.count_nonzero(v==-1)),positive_endpoint_count=None if q is None else int(np.count_nonzero(v==1-q)))
            x,s,n=(arrays[k] for k in ('X','S','N'));res=x-s-n
            xrms=float(np.sqrt(np.mean(x*x)));rrms=float(np.sqrt(np.mean(res*res)));mx=float(np.max(np.abs(res)))
            relative=None if xrms==0 else rrms/xrms
            result.update(raw_residual_rms=rrms,raw_residual_max_abs=mx,X_raw_rms=xrms,relative_rms=relative,strict_pass=None if relative is None else relative<=.001)
            if relative is None:result['reasons'].append('strict_unknown_zero_X_rms')
            elif relative>.001:result['reasons'].append('strict_relative_rms_above_0.001')
            info=list(result['source_info'].values());q=[i['quantum'] for i in info]
            format_known=all(i['format'] in ('WAV','WAVEX','RF64') and i['bits'] is not None for i in info)
            endpoints=None if not format_known else sum(i['negative_endpoint_count']+i['positive_endpoint_count'] for i in info)
            result.update(known_integer_PCM_format=format_known,crop_endpoint_count=endpoints)
            if not format_known:result['reasons'].append('bounds_unknown_format_or_bitdepth')
            elif endpoints:result['reasons'].append('bounds_unknown_crop_code_endpoint_present')
            else:
                for label,factor,flag in [('conditional_nearest_bound',.5,'nearest_compatible'),('codec_bound_general',1.,'codec_consistent')]:
                    bound=factor*sum(q);count=int(np.count_nonzero(np.abs(res)>bound+1e-12))
                    result[label]={'absolute_bound':bound,'exceeding_samples':count,'total_samples':32640,'compatible':count==0}
                    result[flag]=count==0
                    if count:result['reasons'].append(label+'_exceeded')
            result['status']='evaluated'
        except Exception as e:
            result['status']='unknown';result['reasons'].append(repr(e))
        out['rows'].append(result)
    def summary(rows):
        ans={'records':len(rows),'evaluated':sum(r['status']=='evaluated' for r in rows),'unknown_rows':sum(r['status']=='unknown' for r in rows),'processing_chain_unverified':len(rows),'reasons':dict(collections.Counter(reason for r in rows for reason in r['reasons']))}
        for key in ('strict_pass','nearest_compatible','codec_consistent'):
            ans[key]={'pass':sum(r[key] is True for r in rows),'fail':sum(r[key] is False for r in rows),'unknown':sum(r[key] is None for r in rows),'total':len(rows)}
        ans['crop_endpoint_present']=sum((r.get('crop_endpoint_count') or 0)>0 for r in rows)
        return ans
    out['summary']={'overall':summary(out['rows'])}
    for b in sorted({r['priority_bin'] for r in out['rows']}):out['summary'][b]=summary([r for r in out['rows'] if r['priority_bin']==b])
    olddiag=json.loads(Path(cfg['old_diagnosis']).read_text());attempt=json.loads(Path(cfg['old_attempt03']).read_text());first=out['rows'][0]
    out['dev0_crosscheck']={'utt':first['utt'],'old_relative_rms':olddiag['residual']['relative_rms'],'new_relative_rms':first.get('relative_rms'),'absolute_difference':None if first.get('relative_rms') is None else abs(first['relative_rms']-olddiag['residual']['relative_rms']),'old_max_residual':olddiag['residual']['raw_max_abs'],'new_max_residual':first.get('raw_residual_max_abs'),'attempt03_float32_relative_rms':attempt['crop_additivity']['dev']['relative_rms'],'attempt03_premise_passed':attempt['crop_additivity']['dev']['premise_passed']}
    assert out['dev0_crosscheck']['absolute_difference'] is not None and out['dev0_crosscheck']['absolute_difference']<1e-12
    out['status']='completed'
except BaseException as e:
    import traceback
    out['status']='failed';out['failures'].append({'error':repr(e),'traceback':traceback.format_exc()})
finally:
    for path in paths:
        if path in out['source_sha']:out['source_sha'][path]['post']=sha(path)
    out['source_unchanged']=len(out['source_sha'])==192 and all(h.get('pre')==h.get('post') for h in out['source_sha'].values())
    receipt.update(status=out['status'],seconds=time.monotonic()-start,source_unique_files=192,old_metadata_sha_post={p:sha(p) for p in old},registry_sha_post=sha(reg))
    if not out['source_unchanged'] or old!=receipt['old_metadata_sha_post'] or sha(reg)!=cfg['registry_sha256']:out['status']=receipt['status']='failed'
    with (D/'metrics.json').open('x') as f:json.dump(out,f,separators=(',',':'),allow_nan=False)
    receipt['metrics_sha256']=sha(D/'metrics.json');(D/'execution_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');signal.alarm(0)
print(json.dumps({'status':out['status'],'seconds':receipt['seconds'],'pid':os.getpid(),'summary':out.get('summary'),'dev0':out.get('dev0_crosscheck'),'source_unchanged':out['source_unchanged']}))
raise SystemExit(0 if out['status']=='completed' else 1)
