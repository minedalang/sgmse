"""Saved JSON arithmetic only; no model/audio/gradient or statistical rerun."""
import json,math,statistics,hashlib,collections
from pathlib import Path
D=Path(__file__).resolve().parent
O=D.parent/'p8_cpel_mp_pilot_20261019'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def mean(v):return math.fsum(v)/len(v) if v else None
def good(v):return isinstance(v,(int,float)) and math.isfinite(v)
def dist(v):
    v=[x for x in v if good(x)]
    return dict(n=len(v),min=min(v) if v else None,median=statistics.median(v) if v else None,max=max(v) if v else None,mean=mean(v))
N=json.loads((D/'metrics_N.json').read_text());B=json.loads((D/'metrics_D.json').read_text())
old={a:json.loads((O/f'metrics_{a}.json').read_text()) for a in ['M','P']}
baseline=json.loads((O/'initial_base_metrics.json').read_text())
rows={'source_initial':baseline['rows'],'N':N['evaluations']['final_N'],'D':B['evaluations']['final_D'],'historical_M':old['M']['evaluations']['final_M'],'historical_P':old['P']['evaluations']['final_P']}
inputs={str(p):sha(p) for p in [D/'metrics_N.json',D/'metrics_D.json',O/'metrics_M.json',O/'metrics_P.json',O/'initial_base_metrics.json']}
assert all(len(rr)==64 and all(r['controls_complete'] and len(r['controls'])==4 for r in rr) for rr in rows.values())
ids=[r['utt'] for r in rows['source_initial']]
assert all([r['utt'] for r in rr]==ids for rr in rows.values())
assert N['evaluations']['initial_base']==B['evaluations']['initial_base']==baseline['rows']
assert N['initial_base_reused_sha256']==B['initial_base_reused_sha256']==sha(O/'initial_base_metrics.json')
assert N['frozen_state_unchanged'] and B['frozen_state_unchanged'] and N['frozen_state_pre']==B['frozen_state_pre']
cfg=json.loads((D/'config.yaml').read_text());regpath=D/cfg['registries']['dev']['path'];registry=[json.loads(x) for x in regpath.read_text().splitlines()];bins={r['utt']:r['priority_bin'] for r in registry}
keys=['normalized_wave_mse','si_sdr_db','stoi','pesq']
normal={a:{k:dict(total=64,defined=sum(good(r.get(k)) for r in rr),unknown=sum(not good(r.get(k)) for r in rr),mean=mean([r[k] for r in rr if good(r.get(k))])) for k in keys} for a,rr in rows.items()}
differences={}
for a,b in [('D','N'),('N','source_initial'),('D','source_initial'),('historical_P','historical_M')]:
    per=[]
    for x,y in zip(rows[a],rows[b]):per.append(dict(utt=x['utt'],bin=bins[x['utt']],delta={k:x[k]-y[k] if good(x.get(k)) and good(y.get(k)) else None for k in keys}))
    differences[a+'_minus_'+b]=dict(per_file=per,summary={k:dist([r['delta'][k] for r in per]) for k in keys})
components=['lower','upper','orth','outside']
def value(p):
    if p['status']!='evaluated':return None
    pp=p['parents'];assert [v['parent'] for v in pp]==[0,1]
    if p['kind']=='noise_only':return dict(strict=p['codec_diagnostics']['strict_pass'],gain=mean([v['noise_direction_response'] for v in pp]) if all(good(v.get('noise_direction_response')) for v in pp) else None,energy=mean([v['clean_output_change_energy'] for v in pp]))
    c={k:mean([v[k] for v in pp]) for k in components}
    return dict(**c,pair=c['lower']+c['upper']+.25*c['orth']+.2*c['outside'],gain=mean([v['projection_gain'] for v in pp]),energy=mean([v['response_energy'] for v in pp]),outside_change=mean([v['outside_change'] for v in pp]),valid_parents=sum(v['valid'] for v in pp),zero_parents=sum(v['zero_response'] for v in pp),reverse_parents=sum(v['reverse_direction'] for v in pp),overgain_parents=sum(v['overgain'] for v in pp))
kinds=['speech_hann','speech_tukey','noise_only','speech_tukey_pcm16']
pack={a:{r['utt']:{p['kind']:p for p in r['controls']} for r in rr} for a,rr in rows.items()}
coverage={};control={};filecontrols=[]
for kind in kinds:
    coverage[kind]={};control[kind]={}
    for a in rows:
        pp=[pack[a][u][kind] for u in ids];vv=[value(p) for p in pp];ev=[v for v in vv if v is not None]
        coverage[kind][a]=dict(total_files=64,status_counts=dict(collections.Counter(p['status'] for p in pp)),evaluated_files=len(ev),parents=2*len(ev))
        if kind=='noise_only':
            coverage[kind][a].update(codec_eligible=sum(p.get('codec_diagnostics',{}).get('status')=='codec_compatible' for p in pp),strict_pass=sum(p.get('codec_diagnostics',{}).get('strict_pass') is True for p in pp),strict_fail=sum(p.get('codec_diagnostics',{}).get('strict_pass') is False for p in pp),evaluated_strict=sum(v['strict'] for v in ev),undefined_direction=sum(v['gain'] is None for v in ev))
            control[kind][a]={group:dict(files=len(vs),gain=dist([v['gain'] for v in vs]),energy=dist([v['energy'] for v in vs])) for group,vs in [('engineering',ev),('strict_intersection',[v for v in ev if v['strict']])]}
        else:
            control[kind][a]={k:dist([v[k] for v in ev]) for k in ['pair',*components,'gain','energy','outside_change']}
            coverage[kind][a].update({k:sum(v[k] for v in ev) for k in ['valid_parents','zero_parents','reverse_parents','overgain_parents']})
    for u in ids:filecontrols.append(dict(utt=u,bin=bins[u],kind=kind,status={a:pack[a][u][kind]['status'] for a in rows},values={a:value(pack[a][u][kind]) for a in rows}))
paired={}
for kind in kinds:
    rr=[r for r in filecontrols if r['kind']==kind and r['values']['N'] is not None and r['values']['D'] is not None]
    assert all(r['status']['N']==r['status']['D'] for r in filecontrols if r['kind']==kind)
    if kind=='noise_only':
        paired[kind]={}
        for group,gg in [('engineering',rr),('strict_intersection',[r for r in rr if r['values']['N']['strict'] and r['values']['D']['strict']])]:
            paired[kind][group]=dict(files=len(gg),gain_delta=dist([r['values']['D']['gain']-r['values']['N']['gain'] for r in gg if good(r['values']['D']['gain']) and good(r['values']['N']['gain'])]),energy_delta=dist([r['values']['D']['energy']-r['values']['N']['energy'] for r in gg]))
    else:
        nn=mean([r['values']['N']['pair'] for r in rr]);dd=mean([r['values']['D']['pair'] for r in rr]);paired[kind]=dict(files=len(rr),N_pair_mean=nn,D_pair_mean=dd,D_over_N=None if nn is None or nn<=0 else dd/nn,component_delta={k:mean([r['values']['D'][k]-r['values']['N'][k] for r in rr]) for k in components},pair_delta=dist([r['values']['D']['pair']-r['values']['N']['pair'] for r in rr]))
recipe=['step','utt','crop_start','normfactor','center','parent','t','aux_t'];assert len(N['steps'])==len(B['steps'])==500
assert all(all(n[k]==d[k] for k in recipe) for n,d in zip(N['steps'],B['steps']))
budgets={};maxerr=0.
for a,v in [('N',N),('D',B)]:
    assert v['status']=='completed' and v['calls']['optimizer']==500 and v['calls']['backward']==1000 and v['calls']['score_train']==2000
    assert all(x['common_clean_aux_computed'] is False and 'aux' not in x for x in v['steps'])
    for x in v['steps']:
        b=x['gradient_budget'];candidate=min(100.,.01*b['M_norm']/b['pair_norm']) if b['active'] and b['M_norm']>0 and b['pair_norm']>0 else 0.;applied=candidate if a=='D' else 0.
        assert math.isclose(candidate,b['eta_candidate'],rel_tol=1e-12,abs_tol=1e-12) and math.isclose(applied,b['eta_applied'],rel_tol=1e-12,abs_tol=1e-12)
        if b['M_norm']>0:
            effective=applied*b['pair_norm']/b['M_norm'];maxerr=max(maxerr,abs(effective-b['effective_applied_budget']));assert effective<=.01+1e-12
    ss=v['steps'];budgets[a]=dict(steps=500,gradient_denominator='native DSM only; legacy field M_norm retains its name',inactive=sum(not x['gradient_budget']['active'] for x in ss),zero_native=sum(x['gradient_budget']['M_norm']==0 for x in ss),zero_pair=sum(x['gradient_budget']['pair_norm']==0 for x in ss),candidate_cap100=sum(x['gradient_budget']['eta_candidate']==100 for x in ss),clipped_steps=sum(x['grad_preclip']>1 for x in ss),native_norm=dist([x['gradient_budget']['M_norm'] for x in ss]),pair_norm=dist([x['gradient_budget']['pair_norm'] for x in ss]),cosine=dist([x['gradient_budget']['cosine'] for x in ss]),eta_candidate=dist([x['gradient_budget']['eta_candidate'] for x in ss]),eta_applied=dist([x['gradient_budget']['eta_applied'] for x in ss]),effective_applied=dist([x['gradient_budget']['effective_applied_budget'] for x in ss]))
snr={}
for b in sorted(set(bins.values())):
    rr=[x for x in differences['D_minus_N']['per_file'] if x['bin']==b];snr[b]={'normal_D_minus_N':{k:dist([x['delta'][k] for x in rr]) for k in keys},'controls':{}}
    for kind in kinds:
        cc=[x for x in filecontrols if x['bin']==b and x['kind']==kind and x['values']['N'] is not None and x['values']['D'] is not None];key='gain' if kind=='noise_only' else 'pair'
        snr[b]['controls'][kind]=dict(files=len(cc),N_mean=mean([x['values']['N'][key] for x in cc if good(x['values']['N'][key])]),D_mean=mean([x['values']['D'][key] for x in cc if good(x['values']['D'][key])]))
q=paired;normaldiff=differences['D_minus_N']['summary'];noise=q['noise_only']['engineering']
gates={'raw_pair':q['speech_tukey']['files']==64 and q['speech_tukey']['N_pair_mean']>1e-12 and q['speech_tukey']['D_over_N']<=.9,'precision_pair':q['speech_tukey_pcm16']['files']==64 and q['speech_tukey_pcm16']['N_pair_mean']>1e-12 and q['speech_tukey_pcm16']['D_over_N']<=.9,'normal_PESQ':normaldiff['pesq']['n']==64 and normaldiff['pesq']['mean']>=-.03,'normal_STOI':normaldiff['stoi']['n']==64 and normaldiff['stoi']['mean']>=-.005,'normal_SI_SDR':normaldiff['si_sdr_db']['n']==64 and normaldiff['si_sdr_db']['mean']>=-.2,'noise_gain':noise['gain_delta']['n']==noise['files'] and noise['gain_delta']['mean']<=.02,'no_zero_response_degeneracy':coverage['speech_tukey']['D']['zero_parents']<=coverage['speech_tukey']['N']['zero_parents'] and coverage['speech_tukey_pcm16']['D']['zero_parents']<=coverage['speech_tukey_pcm16']['N']['zero_parents']}
summary={'status':'N_D_execution_completed','scientific_screen_met':all(gates.values()),'gates':gates,'input_json_sha256':inputs,'normal':normal,'paired_normal':differences,'controls':control,'coverage':coverage,'paired_controls':paired,'control_per_file':filecontrols,'gradient_budget':budgets,'recipe_matching':{'steps':500,'fields':recipe,'all_common_clean_aux_computed_false':True},'SNR_descriptive':snr,'execution':{a:{k:v[k] for k in ['seconds','calls','frozen_state_pre','frozen_state_post','frozen_state_unchanged','cuda_peak_allocated_bytes','backend_ema']} for a,v in [('N',N),('D',B)]},'historical_comparison_limits':'Same exploratory64 dev reused. N/D norm denominator is native only versus old M/P native+common-point; differences cannot be attributed exclusively to redundancy or treated as an independent four-arm causal interaction. No A-original-native continuing-training arm here.','baseline_reuse':'Old source initial64 identity retained; no baseline inference or old terminal weights reused.'}
(D/'metrics.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
check={'status':'passed','scope':'JSON scalar bookkeeping only, no independent GPU gradient/model or audio recomputation','recipe_paired_steps':500,'native_only_budget_steps_checked':1000,'max_effective_reconstruction_error':maxerr,'no_common_aux_in_steps':True,'baseline_same64_hash_verified':True,'input_json_sha256':inputs,'parent_averaging':'lower+upper+.25orth+.2outside perparent; parents averaged perfile then equal files; ratio of means'}
for path,h in inputs.items():assert sha(Path(path))==h
(D/'independent_validation.json').write_text(json.dumps(check,indent=2)+'\n')
print(json.dumps({'gates':gates,'screen_met':all(gates.values()),'normal_D_minus_N':{k:v['mean'] for k,v in normaldiff.items()},'paired_controls':paired,'budget':budgets,'seconds':N['seconds']+B['seconds']},indent=2))
