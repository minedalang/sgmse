from pathlib import Path
import json,hashlib,subprocess,datetime,wave,math
D=Path(__file__).resolve().parent;R=D.parents[1]
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();wr=lambda n,x:(D/n).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')
m=read(D/'metrics.json');v=read(D/'independent_validation.json');assert m['status']=='completed' and v['status']=='passed' and sha(D/'metrics.json')==v['metrics_sha256'];locks=read(D/'source_PCM_locks.json');before=read(D/'source_metadata_pre.json');assert before==read(D/'source_metadata_post.json')
for p,h in before.items():assert sha(p)==h,p
for p,x in locks.items():
 assert sha(p)==x['sha256'] and Path(p).stat().st_size==x['bytes']
 with wave.open(p,'rb') as w:assert x['header']==dict(rate=w.getframerate(),channels=w.getnchannels(),width=w.getsampwidth(),frames=w.getnframes(),compression=w.getcomptype())
status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=R).decode();old=read(D/'worker_preflight.json')['git_porcelain'];prefix='experiments/'+D.name+'/'
f=lambda s:[l for l in s.splitlines() if prefix not in l];assert f(status)==f(old)
wr('execution_provenance.json',{'main_command':'python3 -B '+str(D/'run_audit.py'),'main_exec_session':66911,'main_pid':22326,'pid_evidence':'exclusive STARTED records actual process PID; not independently ps-observed','started':read(D/'STARTED.json'),'main_exit':0,'main_seconds':m['seconds'],'validation_command':'python3 -B '+str(D/'validate_audit.py'),'validator_exec_session':89363,'validator_exit':0,'validator_seconds':v['seconds'],'main_decodes':80,'additional_validator_decodes':80,'total_actual_decodes':160,'unique_WAVs':80,'retries':0,'new_inference':0,'new_audio_outputs':0,'ASR':0,'quality_recomputations':0,'training':0,'network':0,'metrics_sha256':sha(D/'metrics.json'),'no_old_file_modified':True,'final_report_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
a=m['strata']['all'];report='''# Cached four-seed finite-ensemble decomposition — CPU only

## Provenance and critical historical limitation

Existing64 current cache WAVs for all16 filename×seeds2026–2029, plus16 original clean paths from manifest rows16:32 (0-based),80 unique WAVs. Original metrics SHA89bbe3200c9d658115086d6f7400abb9cde696710dbae623a50b64ae6dc20a89; manifest SHA4349228558395bcdabf9a1e4ab4b23ac6256834169c4338dfd4daca03746558e. Each filename exactly four prescribed seeds/NFE30/path listed in old metadata, no replacement or seed selection. Current PCM16/mono16k/NONE, full equal lengths, no added centering/gain/lag/crop/resample.

**Original per_trajectory contains no WAV-byte SHA. Current80 byte/header/size locks certify present cache and pre/post stability only, NOT byte identity to generation time.** Source config/report/code/path and old metric metadata are provenance clues, weaker than historical audio hashes. No hashes invented, recovery/retrieval/re-generation performed. Results describe the current frozen cache; any historical mutation remains unresolved.

Old configuration is80k EMA **Router active**, manual prepare_specs/build_rrsd_score_calibrator, DNS-CV exploratory16 files/8speakers. It is NOT current VoiceBank32 native Router-off. Historical heldout meant outside old selector development half, not final-test/pretraining-independent. Cache PCM already includes historical clipping and quantization; no raw/preclip uncertainty recovered. No new source noisy/frontend/checkpoint/audio decoded.

## Execution and validation

Read actual AGENTS/lowercase entry/analysis/config and full4551B attachment SHAa235e23451a88273189fb9d649904940a868725db05f3dd38983e49290b73a6b; self readonly SQLite codex_tools_compat, bridge39452/listening16022 live. No existing marker/model process; dirty tree preserved.

Unique main `python3 -B '''+str(D/'run_audit.py')+'''`, session66911 exit0, STARTED2026-10-19T07:27:08.529883+00:00, PID22326 recorded by actual STARTED (not claimed independently ps-observed), **11.891850348s**,80PCM decodes. Unique validator `python3 -B '''+str(D/'validate_audit.py')+'''`, session89363 exit0, **12.374187083s**,additional80PCM decodes. Total160 actual decodes,80unique sources. Main/validator each120s, newdir2MiB. Zero model/ASR/quality/network/training/newwaveform/resampling/retry. Mean signal exists only temporary RAM, no audio or samplearray exported. Source hashes/headers/lengths checked pre/post/final; git outside newdir unchanged; old metadata/code/recentVB32/residual metrics unchanged.

Validator independent implementation imports no business functions. V4 independently from six pairwise distances: sum_k<l mean((Bk−Bl)²)/16, direct temporary scalar four-way-mean residual vs S. It independently checks16 files/4 endpoint MSEs/9 strata/equalfile normalization/ratios/raw IID estimates, exact original current hashes. Mathematical agreement does not close missing historical hash gap. No bootstrap/optimization grid/newseed or newmetric.

## Definitions and principal descriptive result

K=4, M=sample mean (not true conditional mean), P=mean S², E_average=mean_k MSE(Bk,S), E_mean=MSE(M,S), V4=mean_k MSE(Bk,M). Exact finite-set identity **E_average=E_mean+V4**. The lower sample-mean squared error is algebra, not a newly learned method, tested audio system or real content gain.

Equal-file/P-normalized means: E_average **0.249055739459**, E_mean **0.217425239969**, V4 **0.031630499490**. Pooled normalized variance fraction **12.700168869%** = sum_i(V4_i/P_i)/sum_i(E_average_i/P_i). In this current historical Router cache, the complementary sample-mean residual accounts for **87.299831131%** of pooled normalized error; seed dispersion is not the majority. This does not identify the cause of remaining error or prove irreducible bias.

Different statistic, mean per-file V4/E_average **13.567428415%** (range5.954019840–19.300258058%). These two ratios have different weights and must not be conflated. Per-speaker pooled normalized fractions appear below; no generalized model-wide variance estimate.

**Only under an additional IID-same-condition seed-draw assumption**, V_unbiased=(4/3)V4 and raw finite4_bias2_est=E_mean−V_unbiased/4. Across equalfile/P means these are0.042173999321 and0.206881740139; negative raw bias-est count0/16 (negative values would be kept, not clipped). Neither is calibrated posterior uncertainty, true conditional bias, phone uncertainty or hallucination rate. Independence/IID and clean truth not proved by using four seeds. E_mean remains noisy finite-four mean residual. Router/noise/content/PCM confounds prevent transfer to currentVB-OFF.

## All9 strata, equal-file clean-power normalized MSE

| Stratum | files | Eavg/P mean | Emean/P mean | V4/P mean | IID unbiased V/P | raw IID bias2est/P | mean file V4/Eavg | pooled normalized V4/Eavg | negative bias count |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
'''
for name,x in m['strata'].items():
 vals=[x['mean_normalized'][k] for k in ['E_average','E_mean','V_population4','V_unbiased_under_IID','finite4_bias2_est_raw_under_IID']]+[x['mean_file_variance_fraction'],x['pooled_normalized_variance_fraction']];report+='| '+name+' | '+str(x['files'])+' | '+' | '.join(f'{a:.12g}' if a is not None else 'null' for a in vals)+' | '+str(x['negative_bias2_est_count'])+' |\n'
report+='\n## All16 files and four seed endpoint MSEs (raw amplitude MSE)\n\n| File | samples | clean P | seed2026 | seed2027 | seed2028 | seed2029 | Eavg | Emean | V4 | V4/Eavg | raw IID bias2est | identity residual |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n'
for x in m['files']:
 vals=[x['P']]+[x['seed_endpoint_MSEs'][str(s)] for s in [2026,2027,2028,2029]]+[x[k] for k in ['E_average','E_mean','V_population4','population_variance_fraction','finite4_bias2_est_raw_under_IID','identity_residual']];report+='| '+x['filename']+' | '+str(x['samples'])+' | '+' | '.join(f'{v:.12g}' if v is not None else 'null' for v in vals)+' |\n'
report+='\n## Full16 normalized quantities\n\n| File | Eavg/P | Emean/P | V4/P | IID unbiasedV/P | raw IID bias2est/P |\n|---|---:|---:|---:|---:|---:|\n'
for x in m['files']:report+='| '+x['filename']+' | '+' | '.join(f'{v:.12g}' for v in x['normalized_by_P'].values())+' |\n'
report+='''\n## Preservation and stop

Source80 current PCM SHA/header/size in source_PCM_locks.json, old per_trajectory/path evidence in source_registry.json, oldmetadata/code and recentVB32/residual metric protection in source_metadata_pre/post.json. All checked again by independent validator and final reporting, zero old file writes. Main metrics SHA '''+sha(D/'metrics.json')+'''. No metrics were backfilled into historical records.

This completed cached-only decomposition weakens a majority-seed-dispersion explanation **for this cache and this full-PCM squared-error endpoint**. It is not a currentVB32 matched multiseed experiment, conditional-posterior calibration, phoneme correctness label, selection method, or quality/perception guarantee. Remaining error could reflect bias, reference/conditioning issues, residual noise, phase/timing or deterministic processing; this algebra does not separate them. No averaged audio/quality/ASR evaluation or learnedmethod run. Old multi-trajectory selector/anchor/closedVB32 stay closed. Worker stops without extra experiments, outputs, recruitment, training or follow-on inference.
'''
(D/'experiment_report.md').write_text(report)
wr('worker_final_validation.json',{'status':'passed','current_PCM_files_preserved':80,'metadata_code_protected':len(before),'git_outside_newdir_unchanged':True,'main_decodes':80,'validator_decodes':80,'files':16,'strata':9,'max_identity_residual':max(abs(x['identity_residual']) for x in m['files']),'historical_WAV_hash_gap':'unresolved; current locks cannot retroactively establish generation identity','metrics_sha256':sha(D/'metrics.json'),'report_sha256':sha(D/'experiment_report.md'),'no_follow_on':True})
files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in D.iterdir() if p.is_file() and p.name!='final_freeze.json'};wr('final_freeze.json',{'files':files,'bytes_excluding_this_receipt':sum(x['bytes'] for x in files.values()),'budget':2097152,'models':0,'ASR':0,'audio_outputs':0,'newgeneration':0});total=sum(p.stat().st_size for p in D.iterdir() if p.is_file());assert total<2097152
print(json.dumps({'total_bytes':total,'metrics_sha256':sha(D/'metrics.json'),'report_sha256':sha(D/'experiment_report.md'),'freeze_sha256':sha(D/'final_freeze.json')}))
