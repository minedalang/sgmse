"""Explicit one-shot CPU toy only; creates new small files in this directory."""
from pathlib import Path
import copy
import hashlib
import json
import random
import numpy as np
import torch
import resume_io as io

BASE=Path(__file__).resolve().parent
IDENTITY=dict(arm='N',source_sha256='0'*64,config_sha256='1'*64,code_sha256={'toy':'2'*64},
 train_registry_sha256='3'*64,dev_registry_sha256='4'*64,parameter_signature='6'*64,frozen_digest='5'*64)

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def fresh():
    random.seed(17);np.random.seed(19);torch.manual_seed(23)
    model=torch.nn.Linear(2,1);opt=torch.optim.Adam(model.parameters(),lr=.001)
    ema={n:p.detach().clone() for n,p in model.named_parameters()}
    generators={'shuffle_crop':torch.Generator().manual_seed(29),'noise_time':torch.Generator().manual_seed(31)}
    return model,opt,ema,generators,[]

def train(model,opt,ema,gens,order,journal,start,end):
    for step in range(start,end):
        if not order:order.extend(torch.randperm(7,generator=gens['shuffle_crop']).tolist())
        item=order.pop();x=torch.randn(1,2,generator=gens['noise_time'])
        x=x+torch.rand(1,2)+float(np.random.rand())+random.random()+item*.01
        opt.zero_grad(set_to_none=True);loss=model(x).square().mean();loss.backward();opt.step()
        with torch.no_grad():
            for n,p in model.named_parameters():ema[n].mul_(.999).add_(p,alpha=.001)
        commit=journal.append(dict(step=step+1,item=item,loss=float(loss.detach())))
    return commit

def snapshot(model,opt,ema,gens,order,step,commit):
    return dict(schema=io.SCHEMA,identity=copy.deepcopy(IDENTITY),completed_step=step,boundary='optimizer_ema_log_committed',
      raw={n:t.detach().cpu().clone() for n,t in model.state_dict().items()},ema={n:t.clone() for n,t in ema.items()},
      ema_decay=.999,ema_updates=step,optimizer=copy.deepcopy(opt.state_dict()),rng=io.capture_rng(gens),
      remaining_order=list(order),registry_count=7,log_commit=commit)

def guard(**kw):
    # Toy only: real driver must reserve a full largest temporary + pending files +1GiB floor.
    assert kw['stage'] in ('before_write','temporary_written')
    size=sum(p.stat().st_size for p in TEST.rglob('*') if p.is_file())
    if size>1048576:raise RuntimeError('Toy output exceeds1MiB')

def main():
    global TEST
    TEST=BASE/'tiny_resume_artifacts_v3'
    TEST.mkdir(exist_ok=False)  # Explicit one-shot; never reuse or erase a previous run.
    direct=TEST/'direct';split=TEST/'split';direct.mkdir();split.mkdir()
    m,o,e,g,r=fresh();j=io.Journal(direct);commit=train(m,o,e,g,r,j,0,4);expected=snapshot(m,o,e,g,r,4,commit)
    m,o,e,g,r=fresh();j=io.Journal(split);commit=train(m,o,e,g,r,j,0,2);packet=snapshot(m,o,e,g,r,2,commit)
    io.save_latest(split,packet,IDENTITY,guard,temporary_reserve_bytes=65536)
    # Deliberately consume RNG in fresh construction; restore only after all injection.
    loaded=io.load_latest(split,IDENTITY);m,o,e,g,r=fresh()
    m.load_state_dict(loaded['raw'],strict=True);o.load_state_dict(loaded['optimizer'])
    e={n:t.clone() for n,t in loaded['ema'].items()};r=list(loaded['remaining_order'])
    j=io.Journal(split,expected=loaded['log_commit']);io.restore_rng(loaded['rng'],g)
    commit=train(m,o,e,g,r,j,2,4);actual=snapshot(m,o,e,g,r,4,commit)
    assert io.tree_equal(expected,actual),'CPU exact-resume mismatch'
    latest=split/'latest_resume.pt';old_sha=sha(latest)
    def fail_after_write(**kw):
        guard(**kw)
        if kw['stage']=='temporary_written':raise RuntimeError('Synthetic pre-replace failure')
    try:io.save_latest(split,actual,IDENTITY,fail_after_write,temporary_reserve_bytes=65536)
    except RuntimeError:pass
    else:raise AssertionError('Expected injected failure')
    assert sha(latest)==old_sha,'Old latest altered on failed replacement'
    assert len(list(split.glob('latest_resume.tmp.*.failed.json')))==1
    try:io.load_latest(split,IDENTITY)
    except ValueError:pass  # journal extends beyond old latest: no automatic truncation.
    else:raise AssertionError('Uncommitted tail accepted')
    io.save_latest(split,actual,IDENTITY,guard,temporary_reserve_bytes=65536);final=io.load_latest(split,IDENTITY)
    assert io.tree_equal(final,actual)
    for root,name in [(Path('/tmp'),'latest_resume.pt'),(split,'10000_ema.pt'),(split,'../source.pt')]:
        try:io.target(root,name)
        except ValueError:pass
        else:raise AssertionError('Forbidden target accepted')
    broken=copy.deepcopy(actual);del broken['rng']
    try:io.validate(broken,IDENTITY,split)
    except ValueError:pass
    else:raise AssertionError('Incomplete schema accepted')
    out=dict(status='passed_cpu_toy_only',exact_raw_ema_optimizer_rng_order_log=True,
      atomic_failure_preserved_old_latest=True,temporary_and_failure_receipt_retained=True,
      successful_atomic_replacement=True,uncommitted_tail_refused=True,unsafe_targets_refused=True,
      incomplete_schema_refused=True,cuda_or_real_source_verified=False,
      artifact_bytes=sum(p.stat().st_size for p in TEST.rglob('*') if p.is_file()),
      module_sha256=sha(BASE/'resume_io.py'),test_sha256=sha(Path(__file__)))
    assert out['artifact_bytes']<1048576
    (BASE/'tiny_resume_checks.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))

if __name__=='__main__':main()
