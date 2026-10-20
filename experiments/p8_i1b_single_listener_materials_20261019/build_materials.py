"""Once-only Worker source-PCM slice export. No numeric sample conversion."""
from pathlib import Path
import hashlib,io,json,os,signal,sys,time,wave
ROOT=Path(__file__).resolve().parent
def read(p):return json.loads(Path(p).read_text())
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def size():return sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
def prepare_trial_order(observations):
    result=[]
    for stage,arm in enumerate(['noisy','clean'],1):
        rows=[]
        for r in observations:
            opaque=hashlib.sha256(('p8_i1b_single_listener_v1|'+arm+'|'+r['operator_observation_id']).encode()).hexdigest()
            rows.append(dict(stage=stage,arm=arm,trial_id='T'+opaque[:16],order_hash=opaque,observation=r))
        result.extend(sorted(rows,key=lambda r:r['order_hash']))
    if len(result)!=94 or len({r['trial_id'] for r in result})!=94:raise RuntimeError('Not94unique trials')
    return result
def crop_pcm(payload,start,stop):
    if type(start) is not int or type(stop) is not int or stop-start!=7680 or start<0 or stop*2>len(payload):raise RuntimeError('PCM crop geometry')
    return payload[start*2:stop*2]

def execute():
    cfg=read(ROOT/'config.yaml');allocation=read(cfg['allocation'])
    if sha(cfg['allocation'])!=cfg['allocation_sha256']:raise RuntimeError('Allocation changed')
    for n in ['STARTED.json','audio','trials.json','metrics.json','operator_sources.json']:
        if (ROOT/n).exists():raise RuntimeError('Existing execution artifact: no automaticretry '+n)
    begun=time.monotonic()
    def hard(*a):
        try:write(ROOT/'hard_timeout.json',dict(status='failed',reason='120sabsolute'))
        finally:os._exit(124)
    signal.signal(signal.SIGALRM,hard);signal.setitimer(signal.ITIMER_REAL,120)
    with (ROOT/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid()),f)
    allowed={str(Path(x['path']).resolve()) for r in allocation['observations'] for x in r['source_files'].values()}
    def audit(event,args):
        if event.startswith('socket.') or event in ['subprocess.Popen','os.system','os.fork','os.posix_spawn']:raise RuntimeError('Build forbids network/process')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not p.is_relative_to(ROOT):raise RuntimeError('Sourcewrite forbidden')
            if not writing and str(p).startswith('/data/datasets/') and str(p) not in allowed:raise RuntimeError('Unapproved datasetread')
    sys.addaudithook(audit);d=dict(status='running',failures=[],human_responses=0,model_calls=0,network_bytes=0,new_server=False,source_pcm_container_reads=0,source_pcm_sample_values_converted=0,crop_wavs=0)
    try:
        obs=allocation['observations']
        if len(obs)!=47 or len({r['operator_observation_id'] for r in obs})!=47:raise RuntimeError('Allocation scope')
        order=prepare_trial_order(obs);cache={};source_pre={}
        (ROOT/'audio').mkdir();public=[];operator=[]
        for row in order:
            if time.monotonic()-begun>118 or size()>cfg['max_directory_bytes']-65536:raise RuntimeError('Workdeadline/byte reserve')
            r=row['observation'];f=r['source_files'][row['arm']];p=Path(f['path'])
            if p.is_symlink() or p.resolve().parent!=Path('/data/datasets/vb/voicebank_demand/16k/train')/row['arm']:raise RuntimeError('Sourcepath escape')
            if str(p) not in cache:
                b=p.read_bytes();digest=hashlib.sha256(b).hexdigest()
                if digest!=f['sha256'] or len(b)!=f['bytes']:raise RuntimeError('Source lock mismatch')
                with wave.open(io.BytesIO(b),'rb') as w:
                    if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype(),w.getnframes())!=(16000,1,2,'NONE',f['header']['frames']):raise RuntimeError('Source header mismatch')
                    payload=w.readframes(w.getnframes())
                if len(payload)!=2*f['header']['frames']:raise RuntimeError('Truncated source')
                cache[str(p)]=payload;source_pre[str(p)]=dict(sha256=digest,bytes=len(b));d['source_pcm_container_reads']+=1
            start,stop=r['context_start'],r['context_stop']
            if stop-start!=7680 or start<0 or stop>f['header']['frames'] or r['target_relative_start']!=3200 or r['target_relative_stop']!=4480:raise RuntimeError('Context/markergeometry')
            crop=crop_pcm(cache[str(p)],start,stop);target=ROOT/'audio'/(row['trial_id']+'.wav')
            with wave.open(str(target),'wb') as w:w.setparams((1,2,16000,0,'NONE',''));w.writeframes(crop)
            d['crop_wavs']+=1;public.append(dict(trial_id=row['trial_id'],stage=row['stage'],audio='audio/'+target.name))
            operator.append(dict(trial_id=row['trial_id'],stage=row['stage'],arm=row['arm'],observation_id=r['operator_observation_id'],source_path=str(p),source_sha256=f['sha256'],context_start=start,context_stop=stop,target_relative_start=3200,target_relative_stop=4480,crop_sha256=sha(target),crop_pcm_sha256=hashlib.sha256(crop).hexdigest(),crop_samples=7680))
        if d['source_pcm_container_reads']!=24 or d['crop_wavs']!=94:raise RuntimeError('Incomplete24sources/94crops')
        material=hashlib.sha256((cfg['allocation_sha256']+'p8i1b-single-v1').encode()).hexdigest()
        write(ROOT/'trials.json',dict(material_id=material,trials=public));write(ROOT/'operator_sources.json',dict(private_operator_metadata=True,not_a_security_boundary=True,trials=operator,source_pre=source_pre,allocation_sha256=cfg['allocation_sha256']))
        if any(sha(Path(p))!=row['sha256'] for p,row in source_pre.items()):raise RuntimeError('Source changed post')
        d.update(status='completed',material_id=material,source_files=24,planned_judgments=94,human_result_status='unrated_optional_TODO',source_pre_post_match=True)
    except BaseException as e:
        import traceback
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    d.update(seconds=time.monotonic()-begun,pid=os.getpid());write(ROOT/'metrics.json',d)
    (ROOT/'experiment_report.md').write_text('# Optional single-listener materials: '+d['status']+'\n\n94originalPCM480ms crops, middle80ms marked, no normalization or editing.47Noisy must lock before47Clean in independently fixedhash order. Samehuman exploration with memory/expectancybias, no independentgold/accuracy/phonemerecovery claim. BrowserlocalStorage draft and localJSONdownload only; no serverupload. OptionalTODO does not block research. No humanresponses or browserplayback measured by generation. Source/operator maps are omitted from page/trials; staticfiles are not secureantipeeking. Existing8766server reused without restart.\n')
    if size()>cfg['max_directory_bytes'] or time.monotonic()-begun>120:raise RuntimeError('Final120s/2MiBcap')
    signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(dict(status=d['status'],seconds=d['seconds'],bytes=size())));return 0 if d['status']=='completed' else 1
if __name__=='__main__':
    if sys.argv[1:]!=['--build-once']:raise RuntimeError('Worker-only explicit build')
    sys.exit(execute())
