"""Explicitly authorized full-training/resume or bounded technical smoke. Never auto-run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import sys
import time
import traceback

BASE=Path(__file__).resolve().parent
PROJECT=BASE.parent.parent
CODE=('train_longrun.py','training_core.py','resume_io.py','editing.py','loss_helpers.py','gradient_budget.py')
DRAW=('step','utt','crop_start','normfactor','center','parent','t','aux_t')

def require(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def read(path):return json.loads(Path(path).read_text())

def write(path,obj):
    Path(path).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False)+'\n')

def gate(mode,arm,authorization):
    cfg=read(BASE/'config.yaml');authpath=Path(authorization)
    require(authpath.is_absolute() and authpath.resolve().parent==BASE and not authpath.is_symlink(),'Authorization must be local explicit file')
    auth=read(authpath)
    require(not (BASE/'authorizations_used'/(sha(authpath)+'.json')).exists(),'This authorization was already consumed; explicit new authorization required')
    expected=dict(config_sha256=sha(BASE/'config.yaml'),code_sha256={n:sha(BASE/n) for n in CODE},
        core_sha256=cfg['source']['core_sha256'],registry_sha256={k:v['sha256'] for k,v in cfg['registries'].items()},
        metadata_receipt_sha256=cfg['metadata_receipt']['sha256'],original_cv_metadata_sha256=cfg['evaluation']['original_cv_metadata']['sha256'],source_checkpoint_sha256=cfg['source']['checkpoint']['sha256'],
        analysis_plan_sha256=sha(BASE/'analysis_plan.md'),preparation_scope_sha256=sha(BASE/'PREPARATION_SCOPE.md'))
    require(auth.get('status')=='authorized_once' and auth.get('mode')==mode and auth.get('arm')==arm,'Explicit mode/arm authorization required')
    for k,v in expected.items():require(auth.get(k)==v,'Authorization identity mismatch: '+k)
    require(cfg['protocol_version']=='cpel_matched_fulltrain_100k_v1','Wrong protocol')
    require(len(expected['core_sha256'])==52,'Expected complete52 core locks')
    for p,h in expected['core_sha256'].items():require(sha(p)==h,'Core source changed: '+p)
    registries={}
    for k,item in cfg['registries'].items():
        require(sha(item['path'])==item['sha256'],'Registry bytes differ: '+k)
        rows=[json.loads(s) for s in Path(item['path']).read_text().splitlines()]
        require(len(rows)==item['records'] and len({r['utt'] for r in rows})==len(rows),'Registry coverage mismatch')
        registries[k]=rows
    require(sha(cfg['metadata_receipt']['path'])==cfg['metadata_receipt']['sha256'],'Receipt changed')
    require(sha(cfg['evaluation']['original_cv_metadata']['path'])==expected['original_cv_metadata_sha256'],'Original CV metadata changed')
    require(cfg['schedule']['max_steps_per_arm']==cfg['optimization']['max_steps_per_arm']==100000 and cfg['schedule']['checkpoint_every_updates']==5000
        and cfg['schedule']['immutable_EMA_steps']==[10000,25000,50000,75000,100000],'Fixed schedule differs')
    old=read(PROJECT/'experiments/p8_cpel_native_pair_ablation_20261020/config.yaml')
    for k,v in old['optimization'].items():
        if k not in ('max_steps_per_arm','selection'):require(cfg['optimization'].get(k)==v,'Original optimization changed: '+k)
    require(cfg['editing']==old['editing'] and cfg['disable']==old['disable'],'Scientific editing/disable fields changed')
    require(len(registries['train'])==10797 and len(registries['cv'])==1186 and len(registries['validation'])==256,'Full data registry required')
    selected=[];bins=set()
    for row in registries['validation']:
        if row['priority_bin'] not in bins:bins.add(row['priority_bin']);selected.append(row)
    require(len(selected)==4,'Expected four fixed bins')
    if mode=='technical_smoke':
        require(arm=='both' and auth.get('smoke_public_utts')==[r['utt'] for r in selected],'Smoke fixed public IDs differ')
        require(auth.get('allowed_run_roots')==['smoke/N/reference','smoke/N/shadow','smoke/D/reference','smoke/D/shadow'],'Smoke path scope differs')
        require(not (BASE/'smoke').exists(),'Smoke previously started; no repeat')
    else:
        smoke_path=Path(auth.get('technical_smoke_result_path',''))
        require(smoke_path.is_absolute() and smoke_path.resolve().parent==BASE and not smoke_path.is_symlink(),'Explicit local technical-smoke result required')
        require(sha(smoke_path)==auth.get('technical_smoke_result_sha256'),'Technical-smoke receipt SHA differs')
        smoke=read(smoke_path)
        require(smoke.get('status')=='completed_technical_smoke_only' and smoke.get('identity')==expected and len(smoke.get('public',[]))==4,'Technical smoke did not pass same frozen identity')
        require(set(smoke.get('arms',{}))=={'N','D'} and all(a.get('status')=='passed_exact' and a.get('retained_updates')==4 and a.get('actual_updates')==6 for a in smoke['arms'].values()),'Both exact-resume smoke arms required')
        require(arm in ('N','D'),'Training needs explicit arm')
        root=BASE/'runs'/arm
        if mode=='full_training':require(not root.exists(),'Existing arm cannot be started again')
        else:
            require(mode=='resume_training' and root.exists(),'Explicit existing resume arm required')
            latest=root/'latest_resume.pt';r=auth.get('resume',{})
            require(r.get('sha256')==sha(latest) and type(r.get('completed_step')) is int and type(r.get('log_commit')) is dict,'Resume SHA/step/log authorization missing')
        if arm=='D':
            prior=read(BASE/'runs/N/status.json')
            require(prior['status']=='completed' and prior['completed_step']==100000,'Ordinary arm must complete before CPEL arm')
            require(prior['identity']['config_sha256']==expected['config_sha256'] and prior['identity']['code_sha256']=={**expected['code_sha256'],**expected['core_sha256']},'Ordinary arm identity differs')
            verify_milestones(BASE/'runs/N',cfg,prior['identity'])
    return cfg,auth,expected,registries,selected


def environment(cfg):
    # No model or CUDA import precedes mode/authorization/metadata gates.
    dirs=cfg['runtime_directories']
    require(Path(dirs['output_root']).resolve()==BASE and dirs['only_data'],'Output root differs')
    for k in ('TMPDIR','XDG_CACHE_HOME','CUDA_CACHE_PATH','MPLCONFIGDIR'):
        p=Path(dirs[k]);require(p.is_absolute() and p.resolve().is_relative_to('/data'),'Cache outside/data')
        p.mkdir(parents=True,exist_ok=True);os.environ[k]=str(p)
    os.environ.update(PATH=cfg['native_path'],PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',
        TOKENIZERS_PARALLELISM='false',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    sys.path.insert(0,str(PROJECT))


def resource_guard(cfg,stage=None,temporary_bytes=0):
    size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file())
    # Conservatively retain complete audio+checkpoint budgets even as artifacts accumulate.
    remaining=cfg['limits']['planned_final_audio_max_bytes']+cfg['limits']['checkpoint_and_metadata_reserve_bytes']
    extra=max(temporary_bytes or 0,cfg['checkpointing']['temporary_reserve_bytes'])
    require(size+remaining+extra<=cfg['limits']['max_directory_bytes'],'Directory/pending output reservation exceeded')
    require(shutil.disk_usage(BASE).free-remaining-extra>=cfg['limits']['minimum_free_disk_bytes'],'10GiB free floor/pending reservations violated')


def identity(core,cfg,expected):
    return dict(arm=core.arm,source_sha256=expected['source_checkpoint_sha256'],config_sha256=expected['config_sha256'],
        code_sha256={**expected['code_sha256'],**expected['core_sha256']},train_registry_sha256=cfg['registries']['train']['sha256'],
        dev_registry_sha256=hashlib.sha256(json.dumps(expected['registry_sha256'],sort_keys=True).encode()).hexdigest(),
        parameter_signature=core.parameter_digest(),frozen_digest=hashlib.sha256(json.dumps(core.frozen,sort_keys=True).encode()).hexdigest())


def physical(root,core):
    with (root/'physical_events.jsonl').open('a') as f:
        f.write(json.dumps(dict(pid=os.getpid(),attempt_id=core.physical_attempt,physical_attempt=core.physical_attempt,retained_step=core.step,calls=core.calls,wall_time=time.time()),sort_keys=True)+'\n');f.flush();os.fsync(f.fileno())


def verify_milestone(root,step,ident):
    import torch
    from training_core import canonical
    path=root/('ema_'+str(step)+'.pt');rp=root/('ema_'+str(step)+'.receipt.json')
    require(path.exists() and rp.exists(),'Milestone missing or partial; explicit repair required')
    receipt=read(rp)
    require(receipt['status']=='complete' and receipt['identity']==ident and receipt['completed_step']==step
        and receipt['sha256']==sha(path) and receipt['bytes']==path.stat().st_size,'Immutable receipt identity/hash differs')
    loaded=torch.load(path,map_location='cpu',weights_only=True)
    require(loaded['schema']=='cpel-longrun-ema-v1' and loaded['identity']==ident and loaded['completed_step']==step
        and canonical(loaded['dnn'])==receipt['dnn_sha256'],'Immutable content differs')
    return loaded


def verify_milestones(root,cfg,ident):
    for step in cfg['schedule']['immutable_EMA_steps']:verify_milestone(root,step,ident)


def milestone(root,core,ident,cfg):
    import torch
    import resume_io as rio
    from training_core import cpu_copy,canonical
    core.check_frozen();torch.cuda.synchronize(core.device)
    state=cpu_copy(core.model.dnn.state_dict())
    for n,t in core.ema.items():state[n]=t.detach().cpu().clone()
    p=root/('ema_'+str(core.step)+'.pt');rp=root/('ema_'+str(core.step)+'.receipt.json')
    packet=dict(schema='cpel-longrun-ema-v1',identity=ident,completed_step=core.step,dnn=state)
    if p.exists() or rp.exists():
        loaded=verify_milestone(root,core.step,ident)
        require(rio.tree_equal(packet,loaded),'Existing current milestone differs from exact resume boundary')
        return
    require(not p.is_symlink() and not rp.is_symlink(),'Immutable symlink forbidden')
    resource_guard(cfg)
    with p.open('xb') as f:torch.save(packet,f);f.flush();os.fsync(f.fileno())
    loaded=torch.load(p,map_location='cpu',weights_only=True);require(rio.tree_equal(packet,loaded),'Immutable serialization mismatch')
    with rp.open('x') as f:
        json.dump(dict(status='complete',identity=ident,completed_step=core.step,path=p.name,sha256=sha(p),bytes=p.stat().st_size,dnn_sha256=canonical(state)),f);f.flush();os.fsync(f.fileno())
    fd=os.open(str(root),os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    resource_guard(cfg)


def smoke_public(core,rows,cfg,result):
    import numpy as np
    import torch
    from sgmse.model import ScoreModel
    from training_core import canonical
    m=core.model;before=canonical(m.dnn.state_dict());old_sampler=m.get_pc_sampler
    require(sys.getprofile() is None,'Existing profiler cannot be overwritten')
    for row in rows:
        s=core.read(row['clean_path_loader']);n=core.read(row['wav_path'])
        require(s.shape==n.shape==(1,480000),'Complete30s source length required; no crop/chunk fallback')
        r=float(n.abs().max().clamp_min(1e-12));N=core.spec(n/r);A=core.front(N)
        front=(m.frontend.spec_to_waveform(A,length=n.shape[-1])*r).squeeze(1)
        require(front.shape==n.shape,'Full frontend length mismatch')
        active={}
        def observe(predictor,corrector,y,**kwargs):
            require(predictor=='reverse_diffusion' and corrector=='none' and kwargs['N']==30,'Unexpected public sampler')
            run=old_sampler(predictor,corrector,y,**kwargs)
            def sampled():
                out,nfe=run();active['nfe']=int(nfe);return out,nfe
            return sampled
        def profile(frame,event,retval):
            if event=='return' and frame.f_code is ScoreModel.enhance.__code__:
                active.update(norm=float(frame.f_locals['norm_factor']),length=int(frame.f_locals['T_orig']))
        seed=cfg['seed']+100000+row['original_cv_index'];torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        count=core.calls['score'];start=time.monotonic();m.get_pc_sampler=observe;sys.setprofile(profile)
        core.calls['public']+=1
        try:
            with torch.no_grad():wave=m.enhance(front,y_cond=n.to(core.device),N=30,sampler_type='pc',predictor='reverse_diffusion',corrector='none',corrector_steps=1,
                snr=.5,timeit=False,rrsd=False,anchor_calib=False,anchor_v2=False,regen_strength=None)
        finally:m.get_pc_sampler=old_sampler;sys.setprofile(None)
        a=np.asarray(wave)
        require(a.size==480000 and np.isfinite(a).all() and active['length']==480000 and active['norm']==r and active['nfe']==30 and core.calls['score']-count==30,'Complete public invariant failed')
        result.append(dict(utt=row['utt'],seed=seed,length=a.size,norm=r,nfe=active['nfe'],score_calls=core.calls['score']-count,
            output_sha256=hashlib.sha256(a.tobytes()).hexdigest(),seconds=time.monotonic()-start,cuda_peak_bytes=torch.cuda.max_memory_allocated(core.device)))
        require(canonical(m.dnn.state_dict())==before,'Public mutated source dnn');core.check_frozen()


def run_smoke(cfg,expected,registries,selected,result):
    import torch
    import resume_io as rio
    from training_core import Core
    root=BASE/'smoke';root.mkdir();result['public']=[];result['arms']={};matched_draws=None
    for arm in ('N','D'):
        ref=root/arm/'reference';shadow=root/arm/'shadow';ref.mkdir(parents=True);shadow.mkdir()
        result['arms'][arm]=dict(status='running')
        core=Core(cfg,arm,registries['train'],call_sink=result['arms'][arm]);ident=identity(core,cfg,expected)
        result['arms'][arm].update(calls=core.calls,physical_attempt=core.physical_attempt)
        try:
            if arm=='N':
                smoke_public(core,selected,cfg,result['public'])
                # Public is initialization-only; reset full training RNG, identical for both arms.
                import random
                import numpy as np
                random.seed(cfg['seed']);np.random.seed(cfg['seed']);torch.manual_seed(cfg['seed']);torch.cuda.manual_seed_all(cfg['seed'])
            j=rio.Journal(ref);rows=[]
            for _ in range(2):
                row=core.update();rows.append(row);commit=j.append(row);physical(ref,core)
            packet=core.snapshot(ident,commit)
            rio.save_latest(ref,packet,ident,lambda **kw:resource_guard(cfg,**kw),temporary_reserve_bytes=cfg['checkpointing']['temporary_reserve_bytes'])
            for _ in range(2):row=core.update();rows.append(row);commit=j.append(row);physical(ref,core)
            reference=core.snapshot(ident,commit)
            actual_draws=[{k:row[k] for k in DRAW} for row in rows]
            if matched_draws is None:matched_draws=actual_draws
            else:require(actual_draws==matched_draws,'Smoke N/D draw mismatch')
            # Only copy our own committed2-step prefix into separately authorized shadow root.
            with (ref/'train_steps.jsonl').open('rb') as f:data=f.read(packet['log_commit']['offset'])
            with (shadow/'train_steps.jsonl').open('xb') as f:f.write(data);f.flush();os.fsync(f.fileno())
            rio.save_latest(shadow,packet,ident,lambda **kw:resource_guard(cfg,**kw),temporary_reserve_bytes=cfg['checkpointing']['temporary_reserve_bytes'])
            loaded=rio.load_latest(shadow,ident);require(loaded['identity']==identity(core,cfg,expected),'Actual model/signature identity changed')
            j2=rio.Journal(shadow,expected=loaded['log_commit']);core.restore(loaded)
            for _ in range(2):row=core.update();commit=j2.append(row);physical(shadow,core)
            restored=core.snapshot(ident,commit)
            require(rio.tree_equal(reference,restored),'Real source exact resume differs; no tolerance fallback')
            require((ref/'train_steps.jsonl').read_bytes()==(shadow/'train_steps.jsonl').read_bytes(),'Scientific journal differs')
            require(core.calls['optimizer']==6 and core.calls['reverse']==12,'Smoke physical update counts differ')
            result['arms'][arm]=dict(status='passed_exact',retained_updates=4,actual_updates=6,calls_this_process=core.calls,physical_attempt=core.physical_attempt,identity=ident,log_commit=commit)
        finally:
            result['arms'][arm].update(actual_calls=core.calls,retained_step_at_exit=core.step)
            core.close();del core;torch.cuda.empty_cache()
    require(len(result['public'])==4 and sum(a['actual_calls']['source_loads'] for a in result['arms'].values())==2,'Incomplete public feasibility/source loads')
    result['actual_totals']={k:sum(a['actual_calls'][k] for a in result['arms'].values()) for k in result['arms']['N']['actual_calls']}
    result['retained_updates_total']=8;result['actual_updates_total']=12;result['matched_draws']=matched_draws
    result['status']='completed_technical_smoke_only';result['full_training_started']=False


def run_training(mode,arm,cfg,auth,expected,registries,result):
    import resume_io as rio
    from training_core import Core
    root=BASE/'runs'/arm
    if mode=='full_training':root.mkdir(parents=True)
    core=Core(cfg,arm,registries['train'],call_sink=result);ident=identity(core,cfg,expected);expected_rows=None
    try:
        if mode=='resume_training':
            packet=rio.load_latest(root,ident);r=auth['resume']
            require(packet['completed_step']==r['completed_step'] and packet['log_commit']==r['log_commit'],'Resume committed boundary differs')
            journal=rio.Journal(root,expected=packet['log_commit']);commit=packet['log_commit'];core.restore(packet)
            if core.step in cfg['schedule']['immutable_EMA_steps']:milestone(root,core,ident,cfg)
        else:journal=rio.Journal(root)
        if arm=='D':
            expected_rows=(BASE/'runs/N/train_steps.jsonl').open()
            for _ in range(core.step):require(bool(expected_rows.readline()),'N matching prefix too short')
        while core.step<100000:
            resource_guard(cfg);row=core.update()
            if expected_rows is not None:
                other=json.loads(expected_rows.readline());require(all(row[k]==other[k] for k in DRAW),'N/D100% draw mismatch')
            commit=journal.append(row);physical(root,core)
            if core.step%5000==0:
                packet=core.snapshot(ident,commit)
                rio.save_latest(root,packet,ident,lambda **kw:resource_guard(cfg,**kw),temporary_reserve_bytes=cfg['checkpointing']['temporary_reserve_bytes'])
                if core.step in cfg['schedule']['immutable_EMA_steps']:milestone(root,core,ident,cfg)
                write(root/'status.json',dict(status='running',completed_step=core.step,calls_this_process=core.calls,physical_attempt=core.physical_attempt,identity=ident,log_commit=commit))
        core.check_frozen();verify_milestones(root,cfg,ident);require(journal.step==100000,'Retained scientific coverage incomplete')
        result.update(status='completed',completed_step=core.step,calls_this_process=core.calls,physical_attempt=core.physical_attempt,identity=ident,log_commit=commit)
    finally:
        if expected_rows is not None:expected_rows.close()
        result.update(actual_retained_step=core.step,physical_calls_this_process=core.calls)
        core.close()


def execution_seconds(cfg,mode,arm,auth):
    seconds=cfg['technical_smoke']['maximum_seconds'] if mode=='technical_smoke' else cfg['limits']['training_seconds_per_arm']
    if mode!='technical_smoke':
        prior_arm=prior_total=0.
        for old in BASE.glob('attempt_*.json'):
            prior=read(old)
            if prior.get('mode') not in ('full_training','resume_training'):continue
            if prior.get('status')=='running' or 'seconds' not in prior:
                # Root must bind an external terminal audit, never treat unresolved time as zero.
                audit=auth.get('prior_attempts_audit',{})
                require(str(old) in audit,'Unresolved prior attempt needs Root terminal/elapsed audit')
                item=audit[str(old)];proof=read(item['path'])
                require(sha(old)==item['attempt_sha256'] and sha(item['path'])==item['sha256'] and proof.get('terminal') is True and proof.get('elapsed_seconds',-1)>=0,'Prior-attempt terminal audit differs')
                prior=dict(prior,status='audited_terminal',seconds=proof['elapsed_seconds'])
            prior_total+=prior['seconds']
            if prior.get('arm')==arm:prior_arm+=prior['seconds']
        seconds=min(seconds-prior_arm,cfg['limits']['total_training_seconds']-prior_total)
        require(seconds>=1,'Cumulative frozen training budget exhausted')
    return int(seconds)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--mode',required=True,choices=['technical_smoke','full_training','resume_training'])
    parser.add_argument('--arm',required=True,choices=['N','D','both']);parser.add_argument('--authorization',required=True);args=parser.parse_args()
    cfg,auth,expected,registries,selected=gate(args.mode,args.arm,args.authorization)
    attempt=BASE/('attempt_'+str(time.time_ns())+'.json');result=dict(status='running',mode=args.mode,arm=args.arm,identity=expected,failure=None,network_refusals=[])
    def deadline(*_):raise TimeoutError('Frozen execution wall budget reached')
    seconds=execution_seconds(cfg,args.mode,args.arm,auth)
    start=time.monotonic();result.update(pid=os.getpid(),started_wall=time.time());write(attempt,result)
    used=BASE/'authorizations_used';used.mkdir(exist_ok=True)
    with (used/(sha(args.authorization)+'.json')).open('x') as f:json.dump(dict(mode=args.mode,arm=args.arm,attempt=attempt.name),f);f.flush();os.fsync(f.fileno())
    oldsignal=signal.signal(signal.SIGALRM,deadline);signal.alarm(seconds)
    try:
        environment(cfg)
        import socket
        def deny(*a,**kw):result['network_refusals'].append(repr(a)[:300]);raise RuntimeError('Network forbidden')
        socket.socket.connect=deny;socket.socket.connect_ex=deny;socket.create_connection=deny;socket.getaddrinfo=deny
        resource_guard(cfg)
        # FIRST source storage read only after explicit mode/auth/metadata gates.
        require(sha(cfg['source']['checkpoint']['path'])==expected['source_checkpoint_sha256'],'Source checkpoint differs')
        if args.mode=='technical_smoke':run_smoke(cfg,expected,registries,selected,result)
        else:run_training(args.mode,args.arm,cfg,auth,expected,registries,result)
        require(not result['network_refusals'],'Blocked network attempts occurred')
    except BaseException as exc:
        result.update(status='failed',failure=repr(exc),traceback=traceback.format_exc())
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,oldsignal);result['seconds']=time.monotonic()-start
        try:
            observed={p:sha(p) for p in expected['core_sha256']}
            require(observed==expected['core_sha256'],'Core source protection changed')
            require(all(sha(BASE/n)==h for n,h in expected['code_sha256'].items()),'Driver source protection changed')
            require(sha(BASE/'config.yaml')==expected['config_sha256'],'Config protection changed')
            require(sha(cfg['source']['checkpoint']['path'])==expected['source_checkpoint_sha256'],'Source checkpoint post lock changed')
            require(sha(cfg['metadata_receipt']['path'])==expected['metadata_receipt_sha256'],'Metadata receipt post lock changed')
            require(sha(cfg['evaluation']['original_cv_metadata']['path'])==expected['original_cv_metadata_sha256'],'Original CV metadata post lock changed')
            require(sha(BASE/'analysis_plan.md')==expected['analysis_plan_sha256'] and sha(BASE/'PREPARATION_SCOPE.md')==expected['preparation_scope_sha256'],'Protocol documents changed')
            for k,item in cfg['registries'].items():require(sha(item['path'])==item['sha256'],'Registry post lock changed:'+k)
            result['source_locks_post_matched']=True
        except BaseException as exc:result.update(status='failed',source_protection_failure=repr(exc))
        write(attempt,result)
        if args.mode!='technical_smoke' and (BASE/'runs'/args.arm).exists():write(BASE/'runs'/args.arm/'status.json',result)
    if result['status']=='failed':raise SystemExit(1)

if __name__=='__main__':main()
