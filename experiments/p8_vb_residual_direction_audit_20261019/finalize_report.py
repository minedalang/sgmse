from pathlib import Path
import json,hashlib,math,subprocess,datetime
D=Path(__file__).resolve().parent;R=D.parents[1]
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();write=lambda n,v:(D/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
m=read(D/'metrics.json');v=read(D/'independent_validation.json');assert m['status']=='completed' and v['status']=='passed' and sha(D/'metrics.json')==v['metrics_sha256'];pre=read(D/'source_preservation_pre.json');assert pre==read(D/'source_preservation_post.json')
for p,h in pre.items():assert sha(p)==h,p
st=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=R).decode();old=read(D/'worker_preflight.json')['git_porcelain'];prefix='experiments/'+D.name+'/'
f=lambda s:[l for l in s.splitlines() if prefix not in l];assert f(st)==f(old)
write('execution_provenance.json',{'main_command':'python3 -B '+str(D/'run_audit.py'),'main_session':95349,'main_exit':0,'main_pid':read(D/'STARTED.json')['pid'],'main_pid_evidence':'exclusive STARTED created by actual process; no independent ps observation claimed','main_seconds':m['seconds'],'validator_command':'python3 -B '+str(D/'validate_audit.py'),'validator_session':76201,'validator_exit':0,'validator_seconds':v['seconds'],'main_decodes':72,'validator_additional_decodes':72,'total_pcm_decodes':144,'unique_source_WAVs':72,'one_main_one_validation':True,'no_retries':True,'models':0,'ASR':0,'quality_metric_recomputation':0,'audio_outputs':0,'network':0,'training':0,'git_outside_newdir_unchanged':True,'actual_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'source32generation_closed':True})
a=m['strata']['all'];report='''# Fixed residual direction versus step length: existing full PCM audit

## Scope and execution

Research Lead-approved new CPU-only scientific diagnosis, not reopening generation, ASR, quality measurement, selector or training. All32 prescribed existing F/OFF pairs and8 scaled-clean PCM controls,72 unique WAVs. Full16k mono PCM16, original lengths and amplitudes, no centering/resampling/lag/frequency/crop/gainfit. Original noisy/noise/raw.npy not decoded; raw and all other old artifacts only hashed for preservation. No new waveform written, including no oracle blend.

Actual main command `python3 -B '''+str(D/'run_audit.py')+'''`, unique session95349 exit0, elapsed1.984868038s; exclusive STARTED UTC2026-10-19T07:16:36.325637+00:00 records actual PID21031 (not claimed independently observed). Unique independent validation `python3 -B '''+str(D/'validate_audit.py')+'''`, session76201 exit0, elapsed11.701257314s. Main decodes72, validator additional72, total **144 real PCM decodes**. Cached8 clean reused across conditions. No external array/audio/model libraries; standard library wave/struct/math.fsum. Main60s/2MiB cap, validator separate60s. Counters models/ASR/newquality/network/training/audiooutputs/resampling/retries all0.

Current AGENTS/lowercase research entry/frozen analysis/config read; full attachment4551B SHAa235e23451a88273189fb9d649904940a868725db05f3dd38983e49290b73a6b. Self readonly SQLite provider codex_tools_compat; actual bridge39452/listening16022 alive, no model process. Current dirty tree retained; no push.

## Core quantities

S scaled-clean PCM, F frontend PCM, B OFF PCM; e=F−S,d=B−F, P=mean(S²), E0=mean(e²), A=mean(e*d), D=mean(d²), E1=mean((B−S)²). Full-line quadratic E(alpha)=E0+2alphaA+alpha²D. Clean-aware alpha=clip(−A/D,0,1), no output synthesis. Common alpha minimizes equal-case P-normalized MSE. This oracle is not deployable and not selected by ASR or quality.

All32 optimal alpha values are **strictly interior**, range0.0181861678–0.7615874866. Thus every saved full direction has a negative initial MSE slope yet its MSE-minimum on this segment lies before full OFF. This describes PCM pointwise squared error only; it is not actual phone restoration, a tested control algorithm, or user-perceived benefit.

All-case equalcase normalizedMSE: F **0.263909585992**, OFF **0.317828917582**, common-alpha oracle **0.256215524323** at alpha **0.261108468195**, mean per-case oracle **0.253907507159**. Common-alpha reduction relative F is '''+str((a['mean_frontend_normalized_MSE']-a['mean_common_alpha_normalized_MSE'])/a['mean_frontend_normalized_MSE']*100)+'''%; this is an in-sample clean-aware MSE bound, not an evaluated blend or new quality/content endpoint. No threshold-based pass or upgrade to training.

## Gain versus non-gain

Projection uses beta=<d,F>/<F,F>, d_gain=betaF, d_rest=d−betaF, **not fitting any gain to clean**. Gain/rest contributions are each 2mean(e*dpart)+mean(dpart²); cross orthogonality and sum identity independently checked. Non-gain energy fraction arithmetic case mean **0.983988491194** (not pooled energy ratio). Equalcase normalized MSE contribution: gain **−0.007727041569**, rest **+0.061646373160**, sum **+0.053919331590**. Most modification energy is not a global amplitude scale; this alone cannot identify residual noise, phase/time effects, detail or phone correctness.

**Do not equate MSE decomposition with SI-SDR explanation.** Pure nonzero global scaling leaves SI-SDR invariant yet changes amplitude-sensitive MSE. Original saved OFF−front SI-SDR mean−1.326094626dB is merely referenced, never recomputed or optimized over alpha. No STOI/PESQ/ASR calls; no invented human listening.

## All21 strata: equalcase clean-power normalized MSE

| Stratum | n | common alpha | F | OFF | common oracle | mean case oracle | alpha zero/interior/one | mean rest-energy fraction | gain contribution/P | rest contribution/P | saved SI-SDR OFF−F dB |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|
'''
for name,x in m['strata'].items():
 cs=x['alpha_classes'];vals=[x[k] for k in ['common_alpha_oracle','mean_frontend_normalized_MSE','mean_OFF_normalized_MSE','mean_common_alpha_normalized_MSE','mean_case_oracle_normalized_MSE']];report+='| '+name+' | '+str(x['cases'])+' | '+' | '.join(f'{v:.10g}' for v in vals)+f" | {cs['zero']}/{cs['interior']}/{cs['one']} | {x['mean_rest_energy_fraction']:.10g} | {x['mean_gain_MSE_contribution_normalized']:.10g} | {x['mean_rest_MSE_contribution_normalized']:.10g} | {x['mean_saved_OFF_minus_frontend_si_sdr_db']:.10g} |\n"
report+='\n## Full32 per-case audit\n\n| Case | alpha* | E0/P | E1/P | Eoracle/P | A/P | D/P | beta existing-delta/F | rest-energy fraction | gain contribution/P | rest contribution/P |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for r in m['cases']:
 vals=[r['alpha_oracle']]+[r['normalized_by_P'][k] for k in ['E0','E1','E_oracle','A','D']]+[r['gain_beta'],r['rest_energy_fraction'],r['gain_MSE_contribution']/r['P'],r['rest_MSE_contribution']/r['P']];report+='| '+r['basename']+' | '+' | '.join(f'{v:.11g}' if v is not None else 'null' for v in vals)+' |\n'
report+='''\n## Independent verification and preservation

Independent validator does not import original analysis/grouping functions. It decodes original72PCM again, directly constructs scalar residual squares from F+alpha(B−F)−S in memory at endpoints/per-case/common alpha, and directly computes residual-energy differences for gain/rest. It independently rebuilds21 groups and all32/21 summaries, verifies exact metadata/size/hashes/headers, P-normalization, alpha cases, quadratic identity, part contributions/orthogonality and the original saved SI-SDR references. Mathematical specification shared, implementation separate; no new optimization grid/bootstrap or saved audio. Numerical tolerance is floating identity checking, not a scientific selection threshold.

497 original/source/config/plan files hash-identical pre/post/current. Source72 header/size/hash locks rechecked after main; validator verifies them anew. Original metrics SHA577b9567aaad1a7ffc7f5d3fdae0b9c2dcf0da0bd89b86c0f5bbe14818d73f04 and input registry cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998 preserved. Main audit metrics SHA '''+sha(D/'metrics.json')+'''. All old reports/metadata/input/model outputs preserved. Current git outside newdir unchanged. No old artifact rewritten or newmodel_started.

## Limits and disposition

Eight exploratory train sentences/four speakers, two historical noises, repeated conditions; no independent confirmation/global uncertainty claim. Normalized MSE is amplitude-sensitive and not WER, SI-SDR or naturalness. Pointwise non-gain residual is not a phone classifier. Quantized saved PCM is the fixed observation domain, not preclip raw. Clean-aware optimal alpha does not establish an observable selector, generalization or innovation; historical oracle/anchor routes remain closed. Worker stops after this audit and independent validation, without outputblend/listening/ASR/quality/seed/window extension or training. Lead decides the next scientific question separately.
''';(D/'experiment_report.md').write_text(report)
write('worker_final_validation.json',{'status':'passed','preserved_source_files':len(pre),'source_pcm_unique':72,'decodes_main':72,'decodes_validator':72,'cases':32,'strata':21,'git_outside_newdir_unchanged':True,'metrics_sha256':sha(D/'metrics.json'),'independent_validation_sha256':sha(D/'independent_validation.json'),'report_sha256':sha(D/'experiment_report.md'),'max_quadratic_identity_residual':max(abs(r['identity_residual']) for r in m['cases']),'max_part_contribution_residual':max(abs(r['part_contribution_residual']) for r in m['cases']),'max_gain_rest_inner':max(abs(r['gain_rest_inner']) for r in m['cases']),'no_follow_on':True})
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in D.iterdir() if p.is_file() and p.name!='final_freeze.json'};write('final_freeze.json',{'files':files,'bytes_excluding_freeze_receipt':sum(x['bytes'] for x in files.values()),'budget':2097152,'scope':'completed CPU audit, no new audio/model/selector','original32generation_closed':True});total=sum(p.stat().st_size for p in D.iterdir() if p.is_file());assert total<2097152;print(json.dumps({'total_bytes':total,'metrics_sha256':sha(D/'metrics.json'),'report_sha256':sha(D/'experiment_report.md'),'freeze_sha256':sha(D/'final_freeze.json')}))
