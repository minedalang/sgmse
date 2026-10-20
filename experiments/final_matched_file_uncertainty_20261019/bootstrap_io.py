"""Narrow6CSV/sourceJSON lifecycle shared, no statistical business functions."""
from pathlib import Path
import hashlib,importlib.abc,json,os,signal,sys,time
for k in ['OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[k]='1'
D=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def execute(stage,loader,analyze,compare=None):
    import numpy as np
    cfg=json.loads((D/'config.yaml').read_text());root=Path(cfg['source_root']);paths={(arm,split):root/'metrics'/arm/(split+'.csv') for arm in cfg['arms'] for split in cfg['splits']};source=[*paths.values(),root/'metrics.json'];code=[p for p in D.iterdir() if p.is_file() and (p.suffix=='.py' or p.name in ['analysis_plan.md','config.yaml'])]
    marker=D/('STARTED.json' if stage=='main' else 'VALIDATION_STARTED.json')
    if marker.exists():raise RuntimeError('Existing marker no retry')
    start=time.monotonic()
    def hard(*args):
        try:write(D/(stage+'_timeout.json'),dict(status='failed',reason='absolute60s'))
        finally:os._exit(124)
    signal.signal(signal.SIGALRM,hard);signal.setitimer(signal.ITIMER_REAL,60)
    with marker.open('x') as f:json.dump(dict(stage=stage,pid=os.getpid()),f)
    events=[]
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','sklearn','soundfile','librosa','whisper','torchaudio','transformers','pystoi','pesq'}:events.append(name);raise RuntimeError('Forbidden model/audio import')
    sys.meta_path.insert(0,Block());allowed=set(str(p.resolve()) for p in source)
    def guard(event,args):
        if event.startswith('socket.') or event in ['subprocess.Popen','os.system','os.posix_spawn','os.fork']:events.append(event);raise RuntimeError('Forbidden network/process')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];writing=(isinstance(mode,str) and any(c in mode for c in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if writing and not p.is_relative_to(D):events.append('sourcewrite');raise RuntimeError('Sourcewrite forbidden')
            if not writing and ((p.is_relative_to(root) and str(p) not in allowed) or p.suffix in ['.wav','.ckpt'] or str(p).startswith('/data/datasets/')):events.append('forbiddenread');raise RuntimeError('Only frozen6CSV+metrics allowed')
    sys.addaudithook(guard);d=dict(status='running',failures=[],forbidden_attempts=events,audio_reads=0,model_calls=0,training_calls=0,network_bytes=0)
    original_main=sha(D/'metrics.json') if stage!='main' else None
    try:
        d['source_sha_pre']={str(p):sha(p) for p in source};d['code_config_sha_pre']={p.name:sha(p) for p in code}
        if sha(root/'metrics.json')!=cfg['original_metrics_sha256']:raise RuntimeError('Frozen metrics SHA mismatch')
        original=json.loads((root/'metrics.json').read_text());ids,tables=loader(paths,cfg['metrics'],cfg['matched_file_ids']);result,arrays=analyze(tables,cfg,original)
        if arrays['indices'].shape!=(10000,150) or arrays['indices'].dtype!=np.uint16 or arrays['draw_means'].shape!=(10000,14):raise RuntimeError('Bootstrap scope mismatch')
        if stage=='main':np.savez_compressed(D/'matched_bootstrap.npz',**arrays);d['result']=result;d['matched_file_ids']=ids
        else:
            old=json.loads((D/'metrics.json').read_text())
            if old['status']!='completed' or old['matched_file_ids']!=ids:raise RuntimeError('Main incomplete')
            compare(result,old['result'])
            with np.load(D/'matched_bootstrap.npz',allow_pickle=False) as archive:
                if set(archive.files)!=set(arrays) or not np.array_equal(archive['indices'],arrays['indices']) or not np.allclose(archive['draw_means'],arrays['draw_means'],rtol=1e-10,atol=1e-12):raise RuntimeError('Independent drawmeans orindex mismatch')
            d.update(verified_draws=10000,verified_columns=14,protected_main_metrics_sha256=original_main)
        d['status']='completed' if stage=='main' else 'passed'
    except BaseException as e:
        import traceback
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    try:
        d['source_sha_post']={str(p):sha(p) for p in source};d['code_config_sha_post']={p.name:sha(p) for p in code}
        if d['source_sha_pre']!=d['source_sha_post'] or d['code_config_sha_pre']!=d['code_config_sha_post'] or events:raise RuntimeError('Source/codechanged or forbidden attempt')
        if stage!='main' and sha(D/'metrics.json')!=original_main:raise RuntimeError('Original mainmetricschanged')
    except BaseException as e:d['status']='failed';d['failures'].append(dict(final=repr(e)))
    d.update(seconds=time.monotonic()-start,pid=os.getpid(),numpy_version=np.__version__,interpretation='matchedfileID with bothsplits; notspeaker/noise/roomcluster, no multiplicitycorrection, no newmethodchoice or contentclaim')
    write(D/('metrics.json' if stage=='main' else 'independent_validation.json'),d)
    if stage=='main':(D/'experiment_report.md').write_text('# Final frozen metrics: matched-file uncertainty supplement — '+d['status']+'\n\nOriginalhistoricalCI and newmatchedfileCI retained side-by-side forRouter-off/tcut005-off and7metrics. Same150fileIDs keepbothsplits/allarms together, preserving originalequalconditionpooledmean.10000PCG64seed20261019draws/linear95percentiles, no change to methods ormetricdata/no outcome-selectedCI. Not speaker/noise/roomclusters, multiplecomparisoncorrection, crossseed/datasetconfirmation, audio/modelevaluation orhumancontentrecovery. All150rows finite/unique/matched orfailure, no deletion/retune/retry. Independentbincount-weighted sums checked all10000x14draws and endpoints; sharedNumPyRNG and lifecycleIO disclosed.\n')
    if sum(p.stat().st_size for p in D.iterdir() if p.is_file())>cfg['max_directory_bytes'] or time.monotonic()-start>60:raise RuntimeError('Final8MiB/60s cap')
    signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(dict(status=d['status'],seconds=d['seconds'])));return 0 if d['status'] in ['completed','passed'] else 1
