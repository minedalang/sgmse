"""Static128symlink materials, header/hash only. No source PCM decode."""
from pathlib import Path
import hashlib,json,os,signal,sys,time,wave
D=Path(__file__).resolve().parent

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def order_hash(salt,value):return hashlib.sha256((salt+'|'+value).encode()).hexdigest()
def allocation(reg,met,cfg):
    native={r['basename']:r for r in met['native_cases']};groups={g['id']:g['scaledclean'] for g in reg['groups']};refs={r['id']:r for r in reg['references']};ip=Path(cfg['input_registry']).parent;mp=Path(cfg['model_metrics']).parent
    if len(reg['cases'])!=32 or len(native)!=32 or len(groups)!=8:raise RuntimeError('Not fixed32case8controls')
    rows=[]
    for r in reg['cases']:
        n=native[r['basename']];g=groups[r['id']]
        if n['id']!=r['id'] or n['input_sha256']!=r['wav_sha256']:raise RuntimeError('Join mismatch')
        casehash=order_hash(cfg['case_order_salt'],r['basename']);parity=int(order_hash(cfg['AB_parity_salt'],r['basename']),16)%2;mapping={'A':'frontend' if parity==0 else 'off','B':'off' if parity==0 else 'frontend'}
        sources={'clean':dict(path=str(ip/g['wav']),sha256=g['wav_sha256']),'noisy':dict(path=str(ip/r['wav']),sha256=r['wav_sha256']),'frontend':dict(path=str(mp/n['frontend_pcm']['path']),sha256=n['frontend_pcm']['sha256']),'off':dict(path=str(mp/n['off_pcm']['path']),sha256=n['off_pcm']['sha256'])}
        rows.append(dict(case_id='C'+casehash[:16],order_hash=casehash,basename=r['basename'],id=r['id'],speaker=r['speaker'],noise=r['noise'],snr_db=r['snr_db'],samples=r['samples'],AB_mapping=mapping,sources=sources,reference=refs[r['id']]['original_text']))
    return sorted(rows,key=lambda r:r['order_hash'])

def build():
    cfg=read(D/'config.yaml')
    if any((D/n).exists() for n in ['STARTED.json','audio','cases.json','metrics.json']):raise RuntimeError('Existing build, no retry')
    start=time.monotonic()
    def deadline(*a):
        try:write(D/'hard_timeout.json',dict(status='failed',reason='120s'))
        finally:os._exit(124)
    signal.signal(signal.SIGALRM,deadline);signal.alarm(120);(D/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid())))
    d=dict(status='running',failures=[],human_responses=0,PCM_sample_decodes=0,model_calls=0,ASR_calls=0,training_calls=0,external_network_bytes=0)
    try:
        if sha(cfg['input_registry'])!=cfg['input_registry_sha256'] or sha(cfg['model_metrics'])!=cfg['model_metrics_sha256']:raise RuntimeError('Metadata source changed')
        rows=allocation(read(cfg['input_registry']),read(cfg['model_metrics']),cfg);unique={p['path']:p['sha256'] for r in rows for p in r['sources'].values()}
        if len(unique)!=104:raise RuntimeError('Not104unique sources')
        def guard(event,args):
            if event.startswith('socket.') or event in ['subprocess.Popen','os.system','os.fork','os.posix_spawn']:raise RuntimeError('Builder forbids network/process')
            if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
                path=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
                if writing and not path.is_relative_to(D):raise RuntimeError('Sourcewrite forbidden')
                if not writing and (path.suffix in ['.wav','.ckpt'] or str(path).startswith('/data/datasets/')) and str(path) not in unique:raise RuntimeError('Unapproved source')
        sys.addaudithook(guard)
        headers={};source_pre={}
        for path,h in unique.items():
            p=Path(path)
            if p.is_symlink() or not p.resolve().is_relative_to(Path(cfg['input_registry']).parent.parent) or p.suffix!='.wav':raise RuntimeError('Unapproved source path')
            if sha(p)!=h:raise RuntimeError('Source SHA mismatch')
            with wave.open(str(p),'rb') as w:
                if (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())!=(16000,1,2,'NONE'):raise RuntimeError('Header mismatch')
                headers[path]=dict(sample_rate=16000,channels=1,width=2,frames=w.getnframes())
            source_pre[path]=dict(sha256=h,bytes=p.stat().st_size)
        (D/'audio').mkdir();public=[];operator=[]
        for r in rows:
            audio={};entry=dict(r);entry['entries']={}
            for label,source in [('clean','clean'),('noisy','noisy'),('A',r['AB_mapping']['A']),('B',r['AB_mapping']['B'])]:
                value=r['sources'][source];name=order_hash('p8-vb-content-audio-v1',r['case_id']+'|'+label)+'.wav';link=D/'audio'/name;link.symlink_to(value['path']);audio[label]='audio/'+name;entry['entries'][label]=dict(link=str(link.relative_to(D)),source=source,path=value['path'],sha256=value['sha256'])
                if headers[value['path']]['frames']!=r['samples']:raise RuntimeError('Source length mismatch')
            public.append(dict(case_id=r['case_id'],reference=r['reference'],audio=audio,duration_seconds=r['samples']/16000));operator.append(entry)
        material=order_hash('p8-vb-content-single-v1',cfg['input_registry_sha256']+cfg['model_metrics_sha256']);write(D/'cases.json',dict(schema=cfg['public_schema'],material_id=material,cases=public));write(D/'operator_mapping.json',dict(operator_only_not_secure_hiding=True,cases=operator,source_pre=source_pre,headers=headers))
        if any(sha(p)!=h for p,h in unique.items()):raise RuntimeError('Source changed post')
        links=list((D/'audio').iterdir());regular=sum(p.stat().st_size for p in D.rglob('*') if p.is_file() and not p.is_symlink());d.update(status='completed',cases=32,playback_entries=128,unique_source_PCM=104,source_pre_post_unchanged=True,source_files_fullhash_bytes_eachpass=sum(r['bytes'] for r in source_pre.values()),regular_file_bytes_before_report=regular,symlink_own_bytes=sum(p.lstat().st_size for p in links),referenced_unique_source_bytes=sum(r['bytes'] for r in source_pre.values()),referenced_entry_total_bytes=sum(p.stat().st_size for p in links),AB_parity_frontend_A=sum(r['AB_mapping']['A']=='frontend' for r in rows),material_id=material)
    except BaseException as e:
        import traceback
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    d.update(seconds=time.monotonic()-start,pid=os.getpid());write(D/'metrics.json',d);(D/'experiment_report.md').write_text('# Cached32case single-listener materials: '+d['status']+'\n\n128opaque symlink entries/104unchanged fullPCM sources, header/hash only, no PCMdecode/reencoding/normalization/crop. Onehuman reference-aided exploration, notblindcontenttest/humanWER/independentgold/phonemerecovery/CAEG. Fixedhash32caseorder andABparity, no outcomeselection. Publicpublishertext may differ from realizedspeech. Contentlocked before optionalquality. BrowserlocalStorage+partialJSON only, NOTuploadedtoserver. User may takebreaks; optionalTODO not goalblocker. Mappingseparate not secureantipeeking. Existing8766staticserver unchanged. No generatedhumanresponses/model/ASR/training.\n')
    if sum(p.stat().st_size for p in D.rglob('*') if p.is_file() and not p.is_symlink())>cfg['max_new_regular_bytes'] or time.monotonic()-start>120:raise RuntimeError('Final2MiB/120s cap')
    signal.alarm(0);print(json.dumps(dict(status=d['status'],seconds=d['seconds'])));return 0 if d['status']=='completed' else 1
if __name__=='__main__':
    if sys.argv[1:]!=['--build-once']:raise RuntimeError('Worker-only explicit build')
    sys.exit(build())
