"""CPEL boundary-only resume I/O. No training, auto-resume, deletion or CUDA calls."""
from pathlib import Path
import hashlib
import json
import math
import os
import random
import time
import numpy as np
import torch

BASE=Path(__file__).resolve().parent
SCHEMA='cpel-exact-resume-v1'
IDENTITY_KEYS={'arm','source_sha256','config_sha256','code_sha256','train_registry_sha256',
               'dev_registry_sha256','parameter_signature','frozen_digest'}
FIELDS={'schema','identity','completed_step','boundary','raw','ema','ema_decay','ema_updates',
        'optimizer','rng','remaining_order','registry_count','log_commit'}

def require(ok,message):
    if not ok:raise ValueError(message)


def target(root,name):
    root=Path(root)
    require(root.is_absolute() and not root.is_symlink(),'Explicit nonsymlink absolute root required')
    root=root.resolve()
    require(root.is_relative_to(BASE) and root.is_relative_to('/data'),'Only dedicated preparation subtree on /data')
    require(name in {'train_steps.jsonl','latest_resume.pt'},'Only journal or mutable latest; milestones forbidden')
    p=root/name
    require(not p.is_symlink(),'Symlink output forbidden')
    return p


def safe_tree(value):
    if torch.is_tensor(value):
        require(value.device.type=='cpu' and value.layout==torch.strided,'Checkpoint tensors must be CPU strided')
        return
    if type(value) is float:require(math.isfinite(value),'Nonfinite primitive');return
    if value is None or type(value) in (str,int,bool):return
    if type(value) in (list,tuple):
        for item in value:safe_tree(item)
        return
    if type(value) is dict:
        require(all(type(k) in (str,int) for k in value),'Unsafe mapping keys')
        for item in value.values():safe_tree(item)
        return
    raise ValueError('Unsafe checkpoint type: '+str(type(value)))


def tree_equal(a,b):
    if torch.is_tensor(a):return torch.is_tensor(b) and a.dtype==b.dtype and a.shape==b.shape and torch.equal(a,b)
    if type(a)!=type(b):return False
    if type(a) is dict:return a.keys()==b.keys() and all(tree_equal(a[k],b[k]) for k in a)
    if type(a) in (list,tuple):return len(a)==len(b) and all(tree_equal(x,y) for x,y in zip(a,b))
    return a==b


def hash_prefix(path,offset):
    h=hashlib.sha256();remaining=offset
    with open(path,'rb') as f:
        while remaining:
            data=f.read(min(1048576,remaining));require(bool(data),'Short log prefix')
            h.update(data);remaining-=len(data)
    return h.hexdigest()


def verify_log(root,commit,allow_tail=False):
    require(set(commit)=={'offset','sha256'},'Invalid log commit')
    p=target(root,'train_steps.jsonl')
    require(type(commit['offset']) is int and commit['offset']>=0,'Invalid log offset')
    require((p.stat().st_size>=commit['offset'] if allow_tail else p.stat().st_size==commit['offset']),'Uncommitted/missing log tail; explicit recovery required')
    require(hash_prefix(p,commit['offset'])==commit['sha256'],'Log prefix SHA mismatch')


class Journal:
    """One prefix scan on explicit open; linear append thereafter, no tail repair."""
    def __init__(self,root,expected=None):
        self.path=target(root,'train_steps.jsonl');self.hash=hashlib.sha256();self.offset=0;self.step=0
        if self.path.exists():
            require(expected is not None,'Existing journal needs explicit expected commit')
            verify_log(root,expected)
            with self.path.open('rb') as f:
                for line in f:
                    require(line.endswith(b'\n'),'Partial journal line')
                    row=json.loads(line);require(row['step']==self.step+1,'Nonconsecutive journal')
                    self.step=row['step'];self.hash.update(line);self.offset+=len(line)
        else:
            require(expected is None,'Missing committed journal')
            with self.path.open('xb') as f:f.flush();os.fsync(f.fileno())
    def append(self,row):
        require(type(row['step']) is int and row['step']==self.step+1,'Expected next completed optimizer step')
        require(self.path.stat().st_size==self.offset,'Concurrent journal change')
        line=(json.dumps(row,sort_keys=True,allow_nan=False,separators=(',',':'))+'\n').encode()
        with self.path.open('ab') as f:f.write(line);f.flush();os.fsync(f.fileno())
        self.hash.update(line);self.offset+=len(line);self.step=row['step']
        return dict(offset=self.offset,sha256=self.hash.hexdigest())


def capture_rng(generators,cuda_states=None):
    """CUDA states supplied explicitly by driver; never initializes CUDA here."""
    n=np.random.get_state()
    return dict(python=random.getstate(),numpy=dict(name=n[0],keys=n[1].tolist(),pos=int(n[2]),
        has_gauss=int(n[3]),cached_gaussian=float(n[4])),torch_cpu=torch.get_rng_state().clone(),
        torch_cuda={} if cuda_states is None else {str(k):v.cpu().clone() for k,v in cuda_states.items()},
        generators={k:g.get_state().cpu().clone() for k,g in generators.items()})


def restore_rng(state,generators,restore_cuda=None):
    """Call LAST, after model/optimizer/state construction and injection."""
    require(set(generators)==set(state['generators']),'Generator identity mismatch')
    if state['torch_cuda']:
        require(restore_cuda is not None,'Explicit CUDA RNG restoration required')
        restore_cuda(state['torch_cuda'])
    n=state['numpy'];np.random.set_state((n['name'],np.asarray(n['keys'],dtype=np.uint32),n['pos'],n['has_gauss'],n['cached_gaussian']))
    random.setstate(state['python']);torch.set_rng_state(state['torch_cpu'])
    for k,g in generators.items():g.set_state(state['generators'][k])


def validate(packet,identity,root,allow_tail=False):
    safe_tree(packet)
    require(type(packet) is dict and set(packet)==FIELDS,'Exact resume schema fields required')
    require(packet['schema']==SCHEMA and packet['identity']==identity and set(identity)==IDENTITY_KEYS,'Resume identity mismatch')
    def digest(value):return type(value) is str and len(value)==64 and all(c in '0123456789abcdef' for c in value)
    require(identity['arm'] in ('N','D'),'Unknown training arm')
    for k in IDENTITY_KEYS-{'arm','code_sha256'}:require(digest(identity[k]),'Invalid identity digest: '+k)
    require(type(identity['code_sha256']) is dict and bool(identity['code_sha256']) and all(type(k) is str and digest(v) for k,v in identity['code_sha256'].items()),'Invalid code identity')
    require(packet['boundary']=='optimizer_ema_log_committed','Only complete update boundary can be saved')
    step=packet['completed_step'];require(type(step) is int and step>=1,'Invalid completed step')
    require(packet['ema_updates']==step and packet['ema_decay']==.999,'EMA update/decay mismatch')
    require(type(packet['raw']) is dict and bool(packet['raw']) and type(packet['ema']) is dict and bool(packet['ema']),'Missing raw/EMA')
    for name,t in packet['raw'].items():require(type(name) is str and torch.is_tensor(t),'Invalid raw state')
    for name,t in packet['ema'].items():
        require(name in packet['raw'] and torch.is_tensor(t) and t.shape==packet['raw'][name].shape and t.dtype==packet['raw'][name].dtype,'EMA topology mismatch')
    for t in [*packet['raw'].values(),*packet['ema'].values()]:
        require(bool(torch.isfinite(t).all()),'Nonfinite model state')
    opt=packet['optimizer'];require(set(opt)=={'state','param_groups'} and bool(opt['param_groups']),'Missing optimizer state')
    require(type(opt['state']) is dict and bool(opt['state']),'Empty optimizer state')
    ids=[]
    for group in opt['param_groups']:
        require(type(group) is dict and type(group.get('params')) is list and not group.get('amsgrad',False),'Expected plain Adam param groups')
        ids.extend(group['params'])
    require(all(type(i) is int for i in ids) and len(ids)==len(set(ids)) and set(opt['state']).issubset(ids),'Optimizer parameter identity mismatch')
    for state in opt['state'].values():
        require(set(state)=={'step','exp_avg','exp_avg_sq'},'Expected unmodified Adam state')
        require(float(state['step'])>=1 and float(state['step'])<=step,'Invalid Adam step')
        a,b=state['exp_avg'],state['exp_avg_sq']
        require(torch.is_tensor(a) and torch.is_tensor(b) and a.shape==b.shape and a.dtype==b.dtype,'Missing/mismatched Adam moments')
        require(bool(torch.isfinite(a).all()) and bool(torch.isfinite(b).all()),'Nonfinite Adam moments')
    require(type(packet['registry_count']) is int and packet['registry_count']>0,'Invalid registry count')
    order=packet['remaining_order'];require(type(order) is list and len(set(order))==len(order) and all(type(x) is int and 0<=x<packet['registry_count'] for x in order),'Invalid shuffle order')
    rng=packet['rng'];require(set(rng)=={'python','numpy','torch_cpu','torch_cuda','generators'},'Missing RNG states')
    require(set(rng['numpy'])=={'name','keys','pos','has_gauss','cached_gaussian'},'Unsafe NumPy RNG encoding')
    n=rng['numpy'];require(n['name']=='MT19937' and type(n['keys']) is list and len(n['keys'])==624 and all(type(v) is int and 0<=v<2**32 for v in n['keys']) and type(n['pos']) is int and 0<=n['pos']<=624 and n['has_gauss'] in (0,1),'Invalid NumPy legacy RNG')
    py=rng['python'];require(type(py) is tuple and len(py)==3 and py[0]==3 and type(py[1]) is tuple and len(py[1])==625,'Invalid Python RNG')
    require(bool(rng['generators']) and type(rng['torch_cuda']) is dict,'Independent/CUDA generators absent or malformed')
    for t in [rng['torch_cpu'],*rng['torch_cuda'].values(),*rng['generators'].values()]:
        require(torch.is_tensor(t) and t.dtype==torch.uint8 and t.ndim==1,'Invalid Torch RNG state')
    verify_log(root,packet['log_commit'],allow_tail=allow_tail)
    p=target(root,'train_steps.jsonl')
    with p.open('rb') as f:
        last=None
        while f.tell()<packet['log_commit']['offset']:
            line=f.readline();require(bool(line),'Short committed log');last=json.loads(line)
    require(last is not None and last['step']==step,'Committed step/log mismatch')
    return packet


def load_latest(root,identity):
    p=target(root,'latest_resume.pt')
    return validate(torch.load(p,map_location='cpu',weights_only=True),identity,root)


def save_latest(root,packet,identity,disk_reserve_guard,*,temporary_reserve_bytes):
    """Guard callback reserves pending outputs + one full temporary + free floor.

    Any pre-replace failure preserves old latest and new temporary/receipt.
    A post-replace directory-fsync failure is reported as commit uncertainty.
    """
    p=target(root,'latest_resume.pt');validate(packet,identity,root)
    if p.exists():validate(torch.load(p,map_location='cpu',weights_only=True),identity,root,allow_tail=True)
    require(callable(disk_reserve_guard),'External disk reservation guard mandatory')
    require(type(temporary_reserve_bytes) is int and temporary_reserve_bytes>0,'Explicit temporary upper bound required')
    tmp=p.with_name('latest_resume.tmp.'+str(os.getpid())+'.'+str(time.time_ns()))
    replaced=False
    try:
        disk_reserve_guard(stage='before_write',temporary_bytes=temporary_reserve_bytes)
        with tmp.open('xb') as f:torch.save(packet,f);f.flush();os.fsync(f.fileno())
        require(tmp.stat().st_size<=temporary_reserve_bytes,'Temporary exceeded reserved upper bound')
        disk_reserve_guard(stage='temporary_written',temporary_bytes=temporary_reserve_bytes)
        candidate=torch.load(tmp,map_location='cpu',weights_only=True);validate(candidate,identity,root)
        require(tree_equal(packet,candidate),'Serialized state differs from submitted snapshot')
        os.replace(tmp,p);replaced=True
        fd=os.open(str(p.parent),os.O_RDONLY)
        try:os.fsync(fd)
        finally:os.close(fd)
        return dict(path=str(p),bytes=p.stat().st_size,completed_step=packet['completed_step'])
    except BaseException as exc:
        receipt=dict(status='failed',error=repr(exc)[:500],temporary=str(tmp),replaced=replaced,
                     old_latest_preserved=not replaced,automatic_retry=False)
        receipt_path=p.with_name(tmp.name+'.failed.json')
        with receipt_path.open('x') as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
        raise
