"""Raw-gradient budget composition, not a static-lambda objective derivative."""
import math
import torch


def compose_gradients(gm, gp, active, apply_pair):
    if len(gm)!=len(gp):raise ValueError('Gradient collections differ')
    m2=p2=dot=0.
    for m,p in zip(gm,gp):
        if m is None and p is not None:
            raise RuntimeError('Pair gradient without M graph support would change Adam None semantics')
        for value in (m,p):
            if value is not None and not bool(torch.isfinite(value).all()):
                raise RuntimeError('Nonfinite raw gradient; no silent skip')
        if m is not None:m2+=float(m.detach().double().square().sum())
        if p is not None:p2+=float(p.detach().double().square().sum())
        if m is not None and p is not None:dot+=float((m.detach().double()*p.detach().double()).sum())
    if not all(math.isfinite(x) for x in [m2,p2,dot]):raise RuntimeError('Nonfinite gradient geometry')
    mn,pn=math.sqrt(m2),math.sqrt(p2)
    candidate=min(100.,.01*mn/pn) if active and mn>0 and pn>0 else 0.
    applied=candidate if apply_pair else 0.
    combined=[]
    for m,p in zip(gm,gp):
        if m is None:combined.append(None)
        elif p is None or applied==0:combined.append(m.detach())
        else:combined.append(m.detach()+applied*p.detach())
    return combined,dict(M_norm=mn,pair_norm=pn,
        cosine=None if mn==0 or pn==0 else dot/(mn*pn),
        eta_candidate=candidate,eta_applied=applied,
        effective_candidate_budget=None if mn==0 else candidate*pn/mn,
        effective_applied_budget=None if mn==0 else applied*pn/mn,
        active=bool(active),cap=100.,target_raw_gradient_ratio=.01,
        meaning='raw-gradient composition before clip/Adam; no eta derivative or Adam update bound')
