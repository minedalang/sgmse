"""Independent scalar/Python-complex arithmetic; never imports objective/torch.

Consumes the main synthetic JSON rather than re-executing it. Reconstructs all
fixed-case components and the OUVE identities. Not an independent model test.
"""
import argparse
import json
import math
from pathlib import Path

ROOT=Path(__file__).resolve().parent


def reference(case,p):
    fields={k:[] for k in ['per_sample','gain','lower','upper','orthogonal','outside','delta_energy','valid']}
    for dr,tr,m in zip(case['d'],case['delta'],case['mask']):
        d=[complex(*z) for z in dr];t=[complex(*z) for z in tr]
        pd=[x*y for x,y in zip(d,m)];pt=[x*y for x,y in zip(t,m)]
        energy=sum(abs(x)**2 for x in pt);valid=energy>p['min_delta_energy']
        gain=sum((x.conjugate()*y).real for x,y in zip(pd,pt))/energy if valid else 0.
        lo=max(p['lower_margin']-gain,0)**2 if valid else 0.
        hi=max(gain-p['upper_margin'],0)**2 if valid else 0.
        orth=sum(abs(x-gain*y)**2 for x,y in zip(pd,pt))/energy if valid else 0.
        out=sum(abs((1-y)*x)**2 for x,y in zip(d,m)) if valid else 0.
        vals=[lo+p['upper_weight']*hi+p['orthogonal_weight']*orth+p['outside_weight']*out,gain,lo,hi,orth,out,energy,valid]
        for key,val in zip(fields,vals):fields[key].append(val)
    fields['skipped']=len(fields['valid'])-sum(fields['valid'])
    fields['total']=sum(fields['per_sample'])/max(1,sum(fields['valid']))
    return fields


def run(path):
    main=json.loads(path.read_text());p=json.loads((ROOT/'config.yaml').read_text())['objective']
    assert main['status']=='passed'
    count=0;maxerr=0.
    for case in main['fixed_cases']:
        expected=reference(case,p)
        for key,want in expected.items():
            got=case['expected'][key]
            pairs=zip(got,want) if isinstance(want,list) else [(got,want)]
            for g,w in pairs:
                assert math.isfinite(float(g))
                err=abs(float(g)-float(w)); maxerr=max(maxerr,err)
                assert math.isclose(float(g),float(w),rel_tol=1e-11,abs_tol=1e-13),(case['name'],key,g,w)
                count+=1
    z=main['ouve'];a,s=z['a'],z['sigma'];u,A=z['u'],z['anchor']
    for h in [0,1]:
        c=z['c'+str(h)];q=-(u-a*c-(1-a)*A)/s**2
        assert abs(q-z['q'+str(h)])<1e-12
        assert abs((u+s*s*q-(1-a)*A)/a-c)<1e-12
    assert abs(z['q1']-z['q0']-a*(z['c1']-z['c0'])/s**2)<1e-12
    return dict(status='passed',fixed_cases=len(main['fixed_cases']),scalar_values_checked=count,max_abs_error=maxerr,
                scope='independent scalar arithmetic for saved fixed cases and OUVE; does not independently establish gradients or acoustic validity')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('metrics',type=Path);p.add_argument('--output',type=Path);args=p.parse_args()
    result=run(args.metrics)
    if args.output:
        with args.output.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    print(json.dumps(result))
