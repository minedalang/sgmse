"""Only arithmetic on saved JSON. No model, audio, bootstrap or resampling."""
from pathlib import Path
import json,math,statistics,hashlib,collections
D=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def average(values):return math.fsum(values)/len(values) if values else None
def finite(v):return isinstance(v,(float,int)) and math.isfinite(v)
def dist(v):
    v=[x for x in v if x is not None]
    return dict(n=len(v),min=min(v) if v else None,median=statistics.median(v) if v else None,max=max(v) if v else None,mean=average(v))
M=json.loads((D/'metrics_M.json').read_text());P=json.loads((D/'metrics_P.json').read_text());base=json.loads((D/'initial_base_metrics.json').read_text())
inputs={n:sha(D/n) for n in ['metrics_M.json','metrics_P.json','initial_base_metrics.json']}
sets={'initial':base['rows'],'M':M['evaluations']['final_M'],'P':P['evaluations']['final_P']}
assert M['status']==P['status']=='completed'
assert all(len(v)==64 and all(r['controls_complete'] and len(r['controls'])==4 for r in v) for v in sets.values())
assert P['initial_base_reused_sha256']==inputs['initial_base_metrics.json']
assert M['evaluations']['initial_base']==P['evaluations']['initial_base']==base['rows']
assert M['frozen_state_unchanged'] and P['frozen_state_unchanged']
assert M['frozen_state_pre']==P['frozen_state_pre']
regs=[json.loads(s) for s in (D/'dev_registry.jsonl').read_text().splitlines()]
bybin={r['utt']:r['priority_bin'] for r in regs}
assert all([r['utt'] for r in v]==[r['utt'] for r in regs] for v in sets.values())
normal_keys=['normalized_wave_mse','si_sdr_db','stoi','pesq']
def normal(rows):
    return {key:dict(total=len(rows),defined=sum(finite(r.get(key)) for r in rows),missing=sum(not finite(r.get(key)) for r in rows),mean=average([r[key] for r in rows if finite(r.get(key))])) for key in normal_keys}
normal_summary={label:normal(rows) for label,rows in sets.items()}
normal_pairs=[]
for initial,m,p in zip(sets['initial'],sets['M'],sets['P']):
    normal_pairs.append(dict(utt=m['utt'],bin=bybin[m['utt']],P_minus_M={k:p[k]-m[k] if finite(p.get(k)) and finite(m.get(k)) else None for k in normal_keys},M_minus_initial={k:m[k]-initial[k] if finite(m.get(k)) and finite(initial.get(k)) else None for k in normal_keys},P_minus_initial={k:p[k]-initial[k] if finite(p.get(k)) and finite(initial.get(k)) else None for k in normal_keys}))
paired_normal_summary={name:{k:dist([r[name][k] for r in normal_pairs]) for k in normal_keys} for name in ['P_minus_M','M_minus_initial','P_minus_initial']}
components=['lower','upper','orth','outside']
def packet_value(packet):
    if packet['status']!='evaluated':return None
    parents=packet['parents'];assert [x['parent'] for x in parents]==[0,1]
    if packet['kind']=='noise_only':
        return dict(gain=average([x['noise_direction_response'] for x in parents]) if all(finite(x.get('noise_direction_response')) for x in parents) else None,energy=average([x['clean_output_change_energy'] for x in parents]),strict=packet['codec_diagnostics']['strict_pass'])
    c={k:average([x[k] for x in parents]) for k in components}
    return dict(**c,pair=c['lower']+c['upper']+.25*c['orth']+.2*c['outside'],gain=average([x['projection_gain'] for x in parents]),response_energy=average([x['response_energy'] for x in parents]),outside_change=average([x['outside_change'] for x in parents]),valid_parents=sum(x['valid'] for x in parents),zero_parents=sum(x['zero_response'] for x in parents),reverse_parents=sum(x['reverse_direction'] for x in parents),overgain_parents=sum(x['overgain'] for x in parents))
packets={label:{row['utt']:{p['kind']:p for p in row['controls']} for row in rows} for label,rows in sets.items()}
kinds=['speech_hann','speech_tukey','noise_only','speech_tukey_pcm16']
coverage={};control_summary={};perfile=[]
for kind in kinds:
    coverage[kind]={};control_summary[kind]={}
    for label in sets:
        pp=[packets[label][row['utt']][kind] for row in regs]
        values=[packet_value(p) for p in pp];evaluated=[v for v in values if v is not None]
        coverage[kind][label]=dict(records=64,status_counts=dict(collections.Counter(p['status'] for p in pp)),evaluated=len(evaluated),parents=2*len(evaluated))
        if kind=='noise_only':
            coverage[kind][label].update(codec_eligible=sum(p.get('codec_diagnostics',{}).get('status')=='codec_compatible' for p in pp),strict_pass=sum(p.get('codec_diagnostics',{}).get('strict_pass') is True for p in pp),strict_fail=sum(p.get('codec_diagnostics',{}).get('strict_pass') is False for p in pp),evaluated_strict=sum(v['strict'] for v in evaluated),undefined_direction_files=sum(v['gain'] is None for v in evaluated))
            control_summary[kind][label]={'engineering':dict(files=len(evaluated),gain=dist([v['gain'] for v in evaluated]),energy=dist([v['energy'] for v in evaluated])),'strict_intersection':dict(files=sum(v['strict'] for v in evaluated),gain=dist([v['gain'] for v in evaluated if v['strict']]),energy=dist([v['energy'] for v in evaluated if v['strict']]))}
        else:
            control_summary[kind][label]={k:dist([v[k] for v in evaluated]) for k in ['pair',*components,'gain','response_energy','outside_change']}
            coverage[kind][label].update(valid_parents=sum(v['valid_parents'] for v in evaluated),zero_parents=sum(v['zero_parents'] for v in evaluated),reverse_parents=sum(v['reverse_parents'] for v in evaluated),overgain_parents=sum(v['overgain_parents'] for v in evaluated))
    for row in regs:
        pp={label:packets[label][row['utt']][kind] for label in sets};vv={label:packet_value(p) for label,p in pp.items()}
        perfile.append(dict(utt=row['utt'],bin=row['priority_bin'],kind=kind,status={label:p['status'] for label,p in pp.items()},values=vv))
paired_controls={}
for kind in kinds:
    matching=[r for r in perfile if r['kind']==kind and r['values']['M'] is not None and r['values']['P'] is not None]
    assert all(r['status']['M']==r['status']['P'] for r in perfile if r['kind']==kind)
    if kind=='noise_only':
        paired_controls[kind]={}
        for group,rows in [('engineering',matching),('strict_intersection',[r for r in matching if r['values']['M']['strict'] and r['values']['P']['strict']])]:
            paired_controls[kind][group]=dict(files=len(rows),gain_delta=dist([r['values']['P']['gain']-r['values']['M']['gain'] for r in rows if r['values']['P']['gain'] is not None and r['values']['M']['gain'] is not None]),energy_delta=dist([r['values']['P']['energy']-r['values']['M']['energy'] for r in rows]))
    else:
        mm=average([r['values']['M']['pair'] for r in matching]);pm=average([r['values']['P']['pair'] for r in matching])
        paired_controls[kind]=dict(files=len(matching),M_pair_mean=mm,P_pair_mean=pm,ratio_P_over_M=None if mm is None or mm<=0 else pm/mm,pair_difference=dist([r['values']['P']['pair']-r['values']['M']['pair'] for r in matching]),components_P_minus_M={k:average([r['values']['P'][k]-r['values']['M'][k] for r in matching]) for k in components})
recipe_keys=['step','utt','crop_start','normfactor','center','parent','t','aux_t']
assert len(M['steps'])==len(P['steps'])==500
assert all(all(m[k]==p[k] for k in recipe_keys) for m,p in zip(M['steps'],P['steps']))
budget={};max_error=0.
for label,result in [('M',M),('P',P)]:
    steps=result['steps'];assert result['calls']['optimizer']==500 and result['calls']['backward']==1000 and result['calls']['score_train']==2000
    for s in steps:
        b=s['gradient_budget'];candidate=min(100.,.01*b['M_norm']/b['pair_norm']) if b['active'] and b['M_norm']>0 and b['pair_norm']>0 else 0.
        applied=candidate if label=='P' else 0.
        assert math.isclose(candidate,b['eta_candidate'],rel_tol=1e-12,abs_tol=1e-12) and math.isclose(applied,b['eta_applied'],rel_tol=1e-12,abs_tol=1e-12)
        if b['M_norm']>0:
            expected=applied*b['pair_norm']/b['M_norm'];max_error=max(max_error,abs(expected-b['effective_applied_budget']));assert expected<=.01+1e-12
    budget[label]=dict(steps=500,cap_candidate_steps=sum(s['gradient_budget']['eta_candidate']==100 for s in steps),inactive_steps=sum(not s['gradient_budget']['active'] for s in steps),zero_M_steps=sum(s['gradient_budget']['M_norm']==0 for s in steps),zero_pair_steps=sum(s['gradient_budget']['pair_norm']==0 for s in steps),grad_clipped_steps=sum(s['grad_preclip']>1 for s in steps),eta_candidate=dist([s['gradient_budget']['eta_candidate'] for s in steps]),eta_applied=dist([s['gradient_budget']['eta_applied'] for s in steps]),effective_applied=dist([s['gradient_budget']['effective_applied_budget'] for s in steps]),M_norm=dist([s['gradient_budget']['M_norm'] for s in steps]),pair_norm=dist([s['gradient_budget']['pair_norm'] for s in steps]),cosine=dist([s['gradient_budget']['cosine'] for s in steps]))
bins={}
for b in sorted(set(bybin.values())):
    nn=[r for r in normal_pairs if r['bin']==b];bins[b]={'normal_P_minus_M':{k:dist([r['P_minus_M'][k] for r in nn]) for k in normal_keys},'controls':{}}
    for kind in ['speech_tukey','speech_tukey_pcm16','noise_only']:
        rows=[r for r in perfile if r['bin']==b and r['kind']==kind and r['values']['M'] is not None and r['values']['P'] is not None]
        key='gain' if kind=='noise_only' else 'pair';mm=average([r['values']['M'][key] for r in rows if r['values']['M'][key] is not None]);pm=average([r['values']['P'][key] for r in rows if r['values']['P'][key] is not None]);bins[b]['controls'][kind]=dict(files=len(rows),M_mean=mm,P_mean=pm,P_minus_M=None if mm is None or pm is None else pm-mm)
summary=dict(status='M_P_execution_completed_scientific_screen_not_met',protocol_version='v2_dynamic_gradient_budget_and_precision_control',input_json_sha256=inputs,normal=normal_summary,paired_normal=paired_normal_summary,normal_per_file=normal_pairs,coverage=coverage,control_summary=control_summary,paired_controls=paired_controls,control_per_file=perfile,gradient_budget=budget,recipe_matching=dict(matched_steps=500,fields=recipe_keys),SNR_descriptive_groups=bins,arm_execution={a:dict(seconds=v['seconds'],calls=v['calls'],frozen_state_pre=v['frozen_state_pre'],frozen_state_post=v['frozen_state_post'],frozen_state_unchanged=v['frozen_state_unchanged'],cuda_peak_allocated_bytes=v['cuda_peak_allocated_bytes'],backend_ema=v['backend_ema'],network_attempts=v['network_attempts']) for a,v in [('M',M),('P',P)]},scientific_screen=dict(raw_P_over_M=paired_controls['speech_tukey']['ratio_P_over_M'],precision_P_over_M=paired_controls['speech_tukey_pcm16']['ratio_P_over_M'],required_max_ratio=.9,met=False,Root_decision='close this v2 configuration; no A/B, added steps, weights or window rescue; does not reject all counterfactual training concepts'),limitations=['existing CV development, not new independent confirmation','noise codec eligibility is not pure noise/FPR; 27 energy-insufficient rows not counted as zero response','pair outside is whitened and distinct from unwhitened outside_change','dynamic preclip raw-gradient norm bound is not Adam update budget/static scalar-lambda objective','no human content/phoneme/identity claims'])
(D/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
check=dict(status='passed',scope='saved JSON scalar bookkeeping only; no independent gradients/model/audio',protected_input_json_sha256=inputs,paired_steps=500,validated_gradient_steps=1000,max_effective_budget_reconstruction_error=max_error,baseline_reuse_sha_verified=True,raw_and_precision_pair_means='parent-average within each file, then equal files; weighted lower+upper+.25orth+.2outside',no_resampling=True)
for p,h in inputs.items():assert sha(D/p)==h
(D/'training_scalar_validation.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps({'normal_P_minus_M':{k:v['mean'] for k,v in paired_normal_summary['P_minus_M'].items()},'paired_controls':paired_controls,'budget':budget,'seconds':M['seconds']+P['seconds'],'coverage':coverage['noise_only']},indent=2))
