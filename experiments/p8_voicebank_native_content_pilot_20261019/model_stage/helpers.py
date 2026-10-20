"""Standard-library helpers; safe to import without touching input audio or models."""
from pathlib import Path
import hashlib,importlib.util,json

def require(ok,message):
    if not ok:raise RuntimeError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,x):Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def verify(locks):
    actual={p:sha(p) for p in locks};require(actual==locks,'Frozen source/input/code changed');return actual
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def observe(fn,before=None,after=None):
    """Pass identical original positional/keyword objects and original return object."""
    def wrapped(*args,**kwargs):
        if before is not None:before(args,kwargs)
        result=fn(*args,**kwargs)
        if after is not None:after(result,args,kwargs)
        return result
    return wrapped

def transitions(front,off):
    a={r['reference_index']:r['op'] for r in front['word']['alignment'] if r['reference_index'] is not None}
    b={r['reference_index']:r['op'] for r in off['word']['alignment'] if r['reference_index'] is not None}
    require(set(a)==set(b)==set(range(front['word']['N'])),'Reference positions differ')
    rows=[];counts=dict(bothH=0,frontendH_to_OFFnotH=0,frontendnotH_to_OFFH=0,bothnotH=0)
    for i in a:
        key='bothH' if a[i]==b[i]=='H' else 'frontendH_to_OFFnotH' if a[i]=='H' else 'frontendnotH_to_OFFH' if b[i]=='H' else 'bothnotH'
        counts[key]+=1;rows.append(dict(reference_index=i,reference=front['reference_word_tokens'][i],frontend_op=a[i],off_op=b[i],transition=key))
    return dict(counts=counts,positions=rows,frontend_insertions=front['word']['I'],off_insertions=off['word']['I'],interpretation='ASR canonical alignment positions; not verified speech deletion/recovery or phoneme times')

def summarize(items,cases,aggregate):
    bykey={x['key']:x for x in items};out={}
    for label,subset in [('all',cases)]+[('noise_'+n,[r for r in cases if r['noise']==n]) for n in ['N1','N2']]+[('snr_'+str(q),[r for r in cases if r['snr_db']==q]) for q in [-18,0]]+[('noise_'+n+'_snr_'+str(q),[r for r in cases if r['noise']==n and r['snr_db']==q]) for n in ['N1','N2'] for q in [-18,0]]:
        out[label]={arm:aggregate([bykey[r['basename']+'__'+arm]['score'] for r in subset]) for arm in ['noisy','frontend','off']}
    comparisons=[]
    for r in cases:
        f=bykey[r['basename']+'__frontend']['score'];b=bykey[r['basename']+'__off']['score']
        comparisons.append(dict(basename=r['basename'],id=r['id'],noise=r['noise'],snr_db=r['snr_db'],off_minus_frontend={k:{x:b[k][x]-f[k][x] for x in ['S','D','I','edit_distance','rate']} for k in ['word','character']},transitions=transitions(f,b)))
    controls=[x['score'] for x in items if x['arm']=='scaledclean']
    return dict(strata=out,per_case=comparisons,scaledclean=aggregate(controls),scaledclean_measurement_confounds=[x['id'] for x in items if x['arm']=='scaledclean' and (x['score']['word']['edit_distance'] or x['score']['character']['edit_distance'])])

def weight_path(value,locks):
    """Validate local path or real seekable local file without reading/seeking it."""
    import os
    if isinstance(value,(str,Path)):
        p=Path(value).resolve()
    else:
        require(hasattr(value,'seekable') and value.seekable(),'Unverified nonseekable weight stream')
        name=getattr(value,'name',None) or getattr(value,'path',None)
        require(isinstance(name,(str,Path)),'Unknown filelike weight source')
        p=Path(name).resolve();require(hasattr(value,'fileno'),'Weight stream lacks local fd')
        actual=os.fstat(value.fileno());expected=p.stat();require((actual.st_dev,actual.st_ino)==(expected.st_dev,expected.st_ino),'Weight stream differs from named source')
    require(str(p) in locks,'Unfrozen weight source '+str(p));return str(p)

def allow_local_build(executable,argv,cwd,config):
    """Audit only; never execute or rewrite original native compiler commands."""
    import shutil
    require(isinstance(argv,(tuple,list)) and argv,'Native build command must be argv list')
    command=str(argv[0]);resolved=shutil.which(command) if not Path(command).is_absolute() else command
    require(resolved is not None,'Unknown local native executable')
    resolved=str(Path(resolved).resolve());allowed=config['local_build_executables'];require(resolved in allowed,'Process is not approved native compiler '+resolved)
    name=Path(resolved).name;flags=[str(a) for a in argv[1:]]
    if name=='ninja':
        if flags!=['--version']:
            require(set(flags)<=set(['-v','-j','1','2']) and '-v' in flags,'Unexpected native Ninja flags')
            require(cwd is not None and str(Path(cwd).resolve()) in config['native_build_directories'],'Unexpected native Ninja cwd')
    else:
        require(flags in [['-v'],['--version'],['-dumpfullversion','-dumpversion']],'Only local compiler version probes allowed directly')
    return dict(executable=resolved,argv=list(argv),cwd=None if cwd is None else str(cwd))
