"""CPU-only preparation; actual source PCM decoding is Worker-only, never on import."""
from pathlib import Path
import array, datetime, hashlib, importlib.abc, json, math, os, signal, socket, struct, subprocess, sys, time, traceback, wave, zipfile
OUT=Path(__file__).resolve().parent
IDS=['p227_030', 'p227_031', 'p228_026', 'p228_032', 'p230_027', 'p230_029', 'p231_029', 'p231_032']
CAP=24*1024**2
PARENT_CAP=64*1024**2
BASE=OUT.parent

def require(ok,msg):
    if not ok: raise RuntimeError(msg)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def size(): return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
def parent_size(): return sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
def locks(values):
    now={p:sha(p) for p in values};require(now==values,'Source/preparation lock mismatch');return now

def guard(events):
    class BlockModels(importlib.abc.MetaPathFinder):
        def find_spec(self,fullname,path=None,target=None):
            if fullname.split('.')[0] in {'torch','whisper','numpy','soundfile','torchaudio','transformers'}: raise RuntimeError('CPU preparation forbidden import '+fullname)
    sys.meta_path.insert(0,BlockModels())
    def audit(event,args):
        if event.startswith('socket.') or event in {'subprocess.Popen','os.system','os.posix_spawn','os.fork'}:
            events.append(event);raise RuntimeError('CPU preparation network/process forbidden: '+event)
    sys.addaudithook(audit)

def header(p):
    with wave.open(str(p),'rb') as w:
        h=dict(rate=w.getframerate(),channels=w.getnchannels(),width=w.getsampwidth(),frames=w.getnframes(),compression=w.getcomptype())
    require(h['rate']==16000 and h['channels']==1 and h['width']==2 and h['compression']=='NONE','Expected mono16kPCM16 '+str(p));return h

def decode(p):
    h=header(p)
    with wave.open(str(p),'rb') as w: b=w.readframes(h['frames'])
    require(len(b)==h['frames']*2,'Truncated PCM')
    return [v[0]/32768 for v in struct.iter_unpack('<h',b)]

def power(x):
    p=math.fsum(v*v for v in x)/len(x);require(math.isfinite(p) and p>0,'Nonpositive/nonfinite component energy');return p

def recipe(s,noises):
    ps=power(s);rows=[];M=max(abs(v) for v in s)
    for nid,n in noises:
        require(len(s)==len(n),'Noise/clean length mismatch');pn=power(n)
        for q in [-18,0]:
            g=math.sqrt(ps/pn)*10**(-q/20)
            x=[v+g*w for v,w in zip(s,n)]
            M=max(M,max(abs(g*w) for w in n),max(abs(v) for v in x))
            rows.append(dict(noise=nid,snr_db=q,gain=g,clean_power=ps,noise_power=pn,mix=x,noise_samples=n))
    a=min(1.,.9/M)
    return a,M,rows

def f32(x): return [v[0] for v in struct.iter_unpack('<f',struct.pack('<'+'f'*len(x),*x))]
def quantize(x):
    q=[math.floor(v*32768) for v in x];require(all(-32768<=v<=32767 for v in q),'Input saturation forbidden');return q

def store(folder,name,x):
    require(all(math.isfinite(v) for v in x),'Nonfinite input')
    peak=max(abs(v) for v in x);require(peak<=.90004,'Input peak exceeded')
    raw=f32(x);q=quantize(x);floatbound=max(abs(a-b) for a,b in zip(raw,x));err=max(abs(a/32768-b) for a,b in zip(q,x))
    require(err<=1/32768+1e-15,'Quantizer residual exceeded')
    require(max(abs(a/32768-b) for a,b in zip(q,raw))<=1/32768+floatbound+1e-15,'float32/PCM bound')
    wav=folder/(name+'.wav');arr=folder/(name+'.f32le')
    with wave.open(str(wav),'wb') as w:
        w.setparams((1,2,16000,0,'NONE','not compressed'));w.writeframes(struct.pack('<'+'h'*len(q),*q))
    arr.write_bytes(struct.pack('<'+'f'*len(raw),*raw))
    require(decode(wav)==[v/32768 for v in q],'PCM roundtrip mismatch')
    return dict(wav=str(wav.relative_to(OUT)),float32=str(arr.relative_to(OUT)),wav_sha256=sha(wav),float32_sha256=sha(arr),float32_format='little-endian IEEE754 float32, shape [samples], no header',samples=len(x),peak=peak,pcm_peak=max(abs(v/32768) for v in q),saturation_count=0,quantizer='floor(x_float64*32768), saturate forbidden',max_quantization_error=err,max_float32_rounding_error=floatbound,max_pcm_vs_float32_error=max(abs(a/32768-b) for a,b in zip(q,raw)),header=header(wav))

def references(cfg):
    selected={}
    for line in Path(cfg['text_manifest']).read_text().splitlines():
        r=json.loads(line)
        if r['id'] in IDS: require(r['id'] not in selected,'Duplicate reference');selected[r['id']]=r
    require(set(selected)==set(IDS),'Missing reference')
    out=[]
    with zipfile.ZipFile(cfg['text_zip']) as z:
        for sid in IDS:
            t=selected[sid]['transcript'];raw=z.read(t['archive_entry'])
            require(hashlib.sha256(raw).hexdigest()==t['raw_text_SHA256'] and len(raw)==t['raw_text_bytes'],'Reference lock mismatch')
            require(raw.decode('utf-8-sig')==t['original_text'],'Reference text mismatch')
            out.append(dict(id=sid,original_text=t['original_text'],raw_text_hex=raw.hex(),raw_text_sha256=t['raw_text_SHA256'],archive_entry=t['archive_entry']))
    return out

def child():
    require(os.environ.get('PILOT_PREP_SUPERVISOR')==str(os.getppid()),'Supervisor required')
    d=dict(status='running',failures=[],network_attempts=[],network_bytes=0,model_loads=0,training_calls=0,test_content_accessed=False,audio_source_decodes=0)
    def checkpoint():
        write(OUT/'input_progress.json',d);require(time.monotonic()<float(os.environ['PILOT_PREP_DEADLINE']),'120s deadline');require(size()<CAP-500000 and parent_size()<PARENT_CAP-1048576,'Stage/parent byte reserve')
    try:
        guard(d['network_attempts']);cfg=read(OUT/'config.yaml');manifest=read(OUT/'preparation_manifest.json')
        d['source_pre']=locks(cfg['source_locks']);d['prepared_pre']=locks({str(OUT/n):h for n,h in manifest.items()});checkpoint()
        folder=OUT/'inputs';folder.mkdir();refs=references(cfg);noises={}
        old=read(cfg['historical_recipe'])
        for n in cfg['noises']:
            prior=next(v for v in old['noise'] if v['noise_id']==n['id'])
            require(prior['source_path']==n['path'] and prior['source_sha256']==sha(n['path']),'Historical noise lock mismatch')
            require(header(n['path'])['frames']==480000,'Expected full30s historical noise')
            noises[n['id']]=decode(n['path']);d['audio_source_decodes']+=1
            require(hashlib.sha256(struct.pack('<'+'d'*len(noises[n['id']]),*noises[n['id']])).hexdigest()==n['historical_float64_sha256'],'Historical decoded noise hash mismatch')
        registry=dict(schema_version=1,phase='cpu_inputs_only',sources=cfg['source_locks'],references=refs,groups=[],cases=[],noise_sources=cfg['noises'],not_new_holdout=True,metadata_registry_sha256=sha(BASE/'prospective_registry.json'))
        for sample in cfg['samples']:
            sid=sample['id'];s=decode(sample['audio']);d['audio_source_decodes']+=1
            require(header(sample['audio'])==sample['header'],'Clean metadata header changed');require(len(s)==sample['header']['frames'],'Clean source length changed')
            ns=[(nid,noises[nid][80000:80000+len(s)]) for nid in ['N1','N2']]
            a,M,rows=recipe(s,ns);clean=[a*v for v in s];control=store(folder,sid+'_scaledclean',clean)
            group=dict(id=sid,speaker=sample['speaker'],scale=a,group_max_before_scale=M,noise_start_sample=80000,noise_end_sample_exclusive=80000+len(s),scaledclean=control)
            registry['groups'].append(group)
            for row in rows:
                name=sid+'_'+row['noise']+'_snr_'+('m18' if row['snr_db']==-18 else 'p0')
                mixed=[a*v for v in row['mix']];nc=[a*row['gain']*v for v in row['noise_samples']]
                component_snr=10*math.log10(power(clean)/power(nc));require(abs(component_snr-row['snr_db'])<1e-10,'Component SNR mismatch')
                artifact=store(folder,name,mixed);pcm=decode(folder/(name+'.wav'))
                residual=max(abs(v-c-n) for v,c,n in zip(pcm,clean,nc))
                require(residual<=1/32768+artifact['max_float32_rounding_error']+1e-14,'Mixture residual bound')
                residual_noise=[v-c for v,c in zip(pcm,clean)]
                float32_residual_noise=[v-c for v,c in zip(f32(mixed),clean)]
                registry['cases'].append(dict(id=sid,speaker=sample['speaker'],noise=row['noise'],snr_db=row['snr_db'],basename=name+'.wav',seed=(2026+int.from_bytes(hashlib.blake2b((name+'.wav').encode(),digest_size=4).digest(),'little'))%2**32,gain=row['gain'],scale=a,clean_power=row['clean_power'],noise_power=row['noise_power'],component_snr_db=component_snr,float32_residual_noise_snr_db=10*math.log10(power(clean)/power(float32_residual_noise)),pcm_residual_noise_snr_db=10*math.log10(power(clean)/power(residual_noise)),max_pcm_mixture_residual=residual,**artifact));checkpoint()
        require(len(registry['groups'])==8 and len(registry['cases'])==32,'Missing conditions')
        d['source_post']=locks(cfg['source_locks']);d['prepared_post']=locks({str(OUT/n):h for n,h in manifest.items()})
        write(OUT/'input_registry.json',registry);d.update(status='completed',input_registry_sha256=sha(OUT/'input_registry.json'),cases=32,scaledclean_controls=8)
    except BaseException as e:
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    write(OUT/'input_child_result.json',d);return 0 if d['status']=='completed' else 1

def supervise():
    manifest=read(OUT/'preparation_manifest.json');locks({str(OUT/n):h for n,h in manifest.items()})
    for n in ['INPUT_PREPARATION_STARTED.json','inputs','input_registry.json','metrics.json','input_progress.json','input_child_result.json','input_run.log']:
        require(not (OUT/n).exists() and not (OUT/n).is_symlink(),'Existing CPU execution artifact: '+n)
    start=time.monotonic()
    with (OUT/'INPUT_PREPARATION_STARTED.json').open('x') as f: json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),manifest_sha256=sha(OUT/'preparation_manifest.json')),f)
    env=os.environ.copy();env.update(PILOT_PREP_SUPERVISOR=str(os.getpid()),PILOT_PREP_DEADLINE=str(start+118),PYTHONDONTWRITEBYTECODE='1')
    reason=None
    with (OUT/'input_run.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while p.poll() is None:
            if time.monotonic()-start>=118 or size()>=CAP-500000 or parent_size()>=PARENT_CAP-1048576:
                reason='Preparation time/byte limit';os.killpg(p.pid,signal.SIGKILL);p.wait();break
            time.sleep(.05)
    d=read(OUT/'input_child_result.json') if (OUT/'input_child_result.json').exists() else dict(status='failed',failures=[])
    if reason or p.returncode: d['status']='failed';d['failures'].append(dict(supervisor_error=reason or str(p.returncode)))
    try: d['supervisor_source_post']=locks(read(OUT/'config.yaml')['source_locks']);d['supervisor_prepared_post']=locks({str(OUT/n):h for n,h in manifest.items()})
    except BaseException as e: d['status']='failed';d['failures'].append(dict(postcheck=repr(e)))
    d.update(seconds=time.monotonic()-start,child_exit=p.returncode,byte_cap=CAP,time_cap=120,parent_byte_cap=PARENT_CAP,parent_bytes=parent_size(),supervisor_pid=os.getpid(),child_pid=p.pid,model_phase_authorized=False)
    d['artifact_manifest']={str(p.relative_to(OUT)):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in OUT.rglob('*') if p.is_file() and p.name not in ['metrics.json','experiment_report.md']}
    write(OUT/'metrics.json',d)
    (OUT/'experiment_report.md').write_text('CPU input preparation: '+d['status']+'\n\n32 prescribed inputs and eight scaled-clean controls; historical noises, full-utterance SNR, common per-utterance downward scale. All source/preparation locks and guards are in metrics.json and input_registry.json. No model/network/training/test access. Local16k clean from registry; ID association only, conversion/spoken alignment unproven. Old failed local-conversion gate unchanged. Model phase requires separate Root approval after input-registry review. Failure never authorizes retry.\n')
    require(size()<=CAP and parent_size()<=PARENT_CAP,'Final preparation stage/parent budget');require(time.monotonic()-start<=120,'Final preparation exceeds120s');print(json.dumps(dict(status=d['status'],seconds=d['seconds'],bytes=size())));return 0 if d['status']=='completed' else 1

if __name__=='__main__':
    require(sys.argv[1:] in [[],['--child']],'Unsupported arguments')
    sys.exit(child() if sys.argv[1:]==['--child'] else supervise())
