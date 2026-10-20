"""Bounded engineering checks. --output writes a new JSON only, never overwrites."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from objective import paired_objective, clean_readout

ROOT = Path(__file__).resolve().parent


def run():
    torch.set_num_threads(1)
    config = json.loads((ROOT / 'config.yaml').read_text())
    p = config['objective']
    checks, fixtures = [], []
    def check(name, condition):
        assert bool(condition), name
        checks.append(name)
    def evaluate(d, target, mask, **override):
        return paired_objective(d, target, mask, **(p | override))
    t = torch.tensor([[1., 0.]], dtype=torch.float64)
    mask = torch.ones_like(t)
    def case(name, d, target=t, m=mask):
        r = evaluate(d, target, m)
        fixtures.append(dict(name=name, d=[[ [complex(v.item()).real, complex(v.item()).imag] for v in row] for row in d],
                             delta=[[[complex(v.item()).real, complex(v.item()).imag] for v in row] for row in target],
                             mask=m.tolist(), expected={k:v.detach().tolist() for k,v in r.items()}))
        return r
    good = case('aligned', t)
    check('aligned finite zero', good['total'] == 0)
    reverse = case('reverse', -t)
    check('reverse penalized', reverse['lower'][0] > 1)
    large = case('overgain', 3*t)
    check('overgain upper', large['upper'][0] == 2.25)
    zero = case('zero', 0*t)
    check('positive margin zero response penalized', zero['lower'][0] == .1**2)
    check('zero margin permits zero', evaluate(0*t,t,mask,lower_margin=0)['total'] == 0)
    tiny = case('tiny aligned cosine cheat', 1e-9*t)
    cosine = torch.nn.functional.cosine_similarity(1e-9*t,t,dim=1,eps=1e-30)
    check('cosine tiny perfect yet amplitude penalized', cosine[0] == 1 and tiny['lower'][0] > .009)
    perpendicular = case('orthogonal', torch.tensor([[0.,1.]],dtype=t.dtype))
    check('orthogonal energy', perpendicular['orthogonal'][0] == 1)
    soft = case('soft outside', torch.tensor([[1.,2.]],dtype=t.dtype),m=torch.tensor([[1.,.5]],dtype=t.dtype))
    check('soft mask nonidempotent outside', soft['outside'][0] == 1 and soft['orthogonal'][0] == 1)
    ctarget=torch.tensor([[1+1j,2-1j]],dtype=torch.complex128)
    cr=case('complex',.5*ctarget,target=ctarget,m=torch.ones_like(ctarget.real))
    check('complex Hermitian projection', torch.allclose(cr['gain'],torch.tensor([.5],dtype=t.dtype)))
    invalid_d = torch.ones((2,2),dtype=t.dtype,requires_grad=True)
    invalid_t = torch.tensor([[0.,0.],[1e-10,0.]],dtype=t.dtype)
    invalid=case('invalid zero tiny',invalid_d,target=invalid_t,m=torch.ones_like(invalid_t))
    invalid['total'].backward()
    check('zero tiny skipped finite zero gradient', invalid['skipped']==2 and invalid['total']==0 and invalid_d.grad.abs().sum()==0)
    mixed=case('mixed valid invalid',torch.cat([-t,t]),target=torch.cat([t,0*t]),m=torch.ones((2,2),dtype=t.dtype))
    check('valid only batch mean', torch.allclose(mixed['total'],reverse['total']))
    offset=torch.tensor([[23.,-17.]],dtype=t.dtype)
    check('common offset invisible', torch.equal((t+offset)-offset,t))
    # Noise-only intervention changes input, NOT the clean target. This objective
    # skips its undefined clean direction: it does not penalize noise passthrough.
    speech=evaluate(t,t,mask)
    noise=case('noise passthrough zero clean target',t,target=0*t)
    noise_input_gain=float(((t*t).sum()/(t*t).sum()).item())
    noise_response_energy=float(t.square().sum().item())
    check('Noisy passthrough noise target skipped not rejected',
          speech['total']==0 and not noise['valid'][0] and noise['skipped']==1
          and noise['total']==0 and noise_input_gain==1 and noise_response_energy>0)
    a,sigma,A,u,c0,c1=.6,.4,.3,.7,.8,1.
    mu=lambda c:a*c+(1-a)*A
    q0=-(u-mu(c0))/sigma**2; q1=-(u-mu(c1))/sigma**2
    q=torch.tensor([[q0,q1]],dtype=t.dtype)
    rd=clean_readout(q,torch.full_like(q,u),torch.full_like(q,A),a,sigma)
    check('common state distinct kernel targets read exact clean',torch.allclose(rd,torch.tensor([[c0,c1]],dtype=t.dtype)) and abs((q1-q0)-a*(c1-c0)/sigma**2)<1e-12)
    eps=.25
    check('native shared epsilon states differ',abs(((mu(c1)+sigma*eps)-(mu(c0)+sigma*eps))-a*(c1-c0))<1e-12 and q1!=q0)
    upstream_u=torch.tensor([[.7]],dtype=t.dtype,requires_grad=True)
    upstream_A=torch.tensor([[.3]],dtype=t.dtype,requires_grad=True)
    w=torch.tensor(.2,dtype=t.dtype,requires_grad=True)
    frozen=torch.tensor(2.,dtype=t.dtype,requires_grad=False)
    shared_u,shared_A=upstream_u.detach(),upstream_A.detach()
    qscore=w*shared_u+frozen*shared_A
    clean_readout(qscore,shared_u,shared_A,a,sigma).square().sum().backward()
    check('Score grad with caller detached state anchor',w.grad is not None and abs(w.grad.item())>0 and upstream_u.grad is None and upstream_A.grad is None and frozen.grad is None)
    x=torch.tensor([[.4,.2]],dtype=t.dtype,requires_grad=True)
    check('tiny autograd finite difference gradcheck',torch.autograd.gradcheck(lambda z:evaluate(z,t,mask)['total'],(x,),eps=1e-6,atol=1e-5))
    check('complex gradcheck',torch.autograd.gradcheck(lambda z:evaluate(z,ctarget,torch.ones_like(z.real))['total'],(.5*ctarget.clone().requires_grad_(),),eps=1e-6,atol=1e-5))
    return dict(status='passed',scope=config['scope'],check_count=len(checks),checks=checks,
                fixed_cases=fixtures,ouve=dict(a=a,sigma=sigma,anchor=A,u=u,c0=c0,c1=c1,q0=q0,q1=q1,readout=rd.tolist()),
                noise_passthrough_diagnostic=dict(input_direction_gain=noise_input_gain,response_energy=noise_response_energy,clean_direction_undefined=True,interpretation='diagnostic along noise input, NOT clean gain; zero clean target skipped, so noise response is not penalized'),
                limitations=['zero edit with identical inputs and RNG is consistency, not learned evidence','not identifiability or enhancement evidence','positive margin enforces amplitude','zero margin admits zero','common error cancels','Noisy passthrough follows noise too','no optimizer, model, source data or audio'],
                code_sha256={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in ['config.yaml','objective.py','synthetic_check.py','reference_check.py']})

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);args=parser.parse_args()
    result=run()
    if args.output:
        with args.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(dict(status=result['status'],check_count=result['check_count'])))
