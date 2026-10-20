from pathlib import Path
import json,hashlib,subprocess,datetime,math
D=Path(__file__).resolve().parent;B=D.parent;R=D.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(n,v):(D/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n')
m=read(D/'metrics.json');reg=read(B/'input_stage/input_registry.json');val=read(D/'model_independent_validation.json');man=read(D/'execution_manifest.json');pre=read(D/'worker_preflight.json')
assert m['status']=='completed' and val['status']=='passed' and sha(D/'metrics.json')==val['metrics_sha256']
for p,h in man['locks'].items():assert sha(p)==h,p
for p,a in m['artifact_manifest'].items():assert sha(D/p)==a['sha256'] and (D/p).stat().st_size==a['bytes'],p
status=subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all'],cwd=R).decode()
prefix='experiments/'+B.name+'/'
f=lambda s:[l for l in s.splitlines() if prefix not in l]
assert f(pre['git_porcelain'])==f(status)
write('execution_provenance.json',{'command':'/data/coding/sgmse/env/bin/python -B '+str(D/'run_pilot.py'),'exec_session':33718,'supervisor_pid':14679,'child_pid':14680,'pid_evidence':'actually observed both by Worker ps; also Root independently observed and supervisor metrics agree','started':read(D/'STARTED.json'),'supervisor_seconds':m['supervisor_seconds'],'exit_code':0,'unique_run':True,'smokes':0,'retries':0,'validator_command':'/data/coding/sgmse/env/bin/python -B '+str(D/'validate_model_run.py'),'validator_session':50900,'validator_exit':0,'unique_validator':True,'validator_elapsed':'not separately timed; no fabricated duration','validator_scope':val,'metrics_original_sha256':sha(D/'metrics.json'),'input_registry_sha256':sha(B/'input_stage/input_registry.json'),'root_authorization_sha256':sha(B/'model_stage_root_authorization.json'),'utc_report':datetime.datetime.now(datetime.timezone.utc).isoformat()})
quality_errors=[{'basename':r['basename'],'arm':a,'metric':k,**t} for r in m['quality']['per_case'] for a,d in r['arms'].items() for k,t in d.items() if t['error'] or t['warnings']]
write('worker_preservation_validation.json',{'status':'passed','source_locks':len(man['locks']),'pre_post_supervisor_current_all_equal':m['source_pre']==m['source_post']==m['supervisor_post_locks']==man['locks'],'runtime_original_artifacts':len(m['artifact_manifest']),'all_runtime_artifact_hashes_and_bytes_match':True,'git_other_entries_unchanged':True,'git_after':status,'original_metrics_sha256':sha(D/'metrics.json'),'quality_errors_or_warnings':quality_errors,'model_calls_after_unique_run':0,'ASR_calls_after_unique_run':0,'no_follow_on':True})
fmt=lambda x:'null' if x is None else f'{x:.6f}'
word=lambda a:f"{100*a['word']['micro_rate']:.3f}%; {a['word']['S']}/{a['word']['D']}/{a['word']['I']}"
report='''# Prospective fixed-candidate headroom: completed unique scientific run

## Execution and integrity

One native run (no additional smoke/retry), session **33718**, observed supervisor PID **14679**, child **14680**, STARTED **2026-10-19T06:47:45.930133+00:00**, supervisor elapsed **97.644696386 s**, exit 0. Frozen180s total/170s execution reserve10s/64MiB parent respected. Exact command:

`/data/coding/sgmse/env/bin/python -B '''+str(D/'run_pilot.py')+'''`

One prepared independent CPU validator, session **50900**, exit0/passed. Exact command: `/data/coding/sgmse/env/bin/python -B '''+str(D/'validate_model_run.py')+'''`. It verified 64 serialization outputs,104 scores,32 native cases,21 headroom strata and10000 exploratory bootstrap draws. SI-SDR independently recomputed; STOI/PESQ reuse the same frozen official metric implementations with independently implemented bookkeeping, not independent published algorithms. Validator elapsed was not separately measured. No neural model/ASR loaded by validator.

All413 frozen locks pre/post/supervisor/current and '''+str(len(m['artifact_manifest']))+''' original runtime artifacts hash/bytes matched. Source data, original metadata/CPU registry/metrics, old VB/B/failure records/checkpoints and dirty git outside current experiment unchanged. Original model metrics SHA256 `'''+sha(D/'metrics.json')+'''`; input registry `cfc2e1b4ac313037f7e324b6e548f0f56cb1ee32a80282475a0823c7a876d998`. Original metrics preserved, not rewritten for report. Bridge/listening processes checked alive after execution; no new listening or human judgments.

32 frontend and32 OFF calls, each NFE30; one enhancement-model load and one ASR-model load after enhancement weakrefs released. 104 transcribe API calls and104 internal decode API calls; **neural forward count not measured, API count is not forward count**. One strictly identified refused local urllib3 IPv6 probe, zero unknown refusals, zero actual network bytes/training/test access. All socket creation remained refused. Project input/enhancement/ASR pipeline resampling0; STOI internally resamples to10k/removes silence, PESQ jointly peak-scales and performs standard internal processing. No added gain/lag fit, threshold or seed search.

## Primary results and essential confounds

All cases retained. Word endpoint single-tiny.en ASR relative to publisher sentence text, **not human content truth or actual phoneme recovery**. Uncapped WER is required, including values above100%. Fixed front/OFF pool only; oracle chooses entire candidate by lower word edits (ties front), not word splicing or deployable selector. 32 conditions comprise8 sentences/4 speakers/2 reused noise recordings; not32 independent utterances. Local16k waveform/publisher-ID association does not prove realized phones, spoken alignment or conversion equivalence; prior2LSB source gate remains failed. Pretraining/external/anonymized exposure unknown.

Primary all: frontend E378, OFF E229, offline oracle E159 over N288; best fixed OFF, headroom70/288=**24.305556 percentage points**. However frontend wins23 cases, OFF wins4, ties5. **One frontend ASR repetition dominates error sums**: p227_030_N1_snr_m18 has222 edits,216 insertions; its literal repetitive hypothesis remains below and in raw JSON. Thus pooled OFF advantage/oracle headroom must not be described as broad real-content restoration. Most cases favor frontend, and three of four speaker totals favor frontend. No posthoc exclusion, WER clipping, replacement, or recalculated favorable endpoint performed.

Scaled-clean controls:72 reference words,2 substitutions/0 deletions/0 insertions (WER2.777778%); character3 edits/312 (0.961538%). Nonzero controls p230_027 and p231_032 are kept as ASR/reference adaptation confounds, not used to change IDs or reference. These controls do not certify all other labels as spoken truth.

Exploratory speaker-cluster bootstrap seed20261019,10000 draws,4 clusters: OFF−front microWER interval **[-253.278689,35.563380]pp**, headroom **[0.373134,29.347826]pp**. Four clusters and outlier sensitivity do not support reliable nominal coverage or confirmatory significance; intervals are descriptive, not an upgrade/train gate.

## All21 strata: WER and S/D/I

| Stratum | N | noisy WER;S/D/I | frontend WER;S/D/I | OFF WER;S/D/I | oracle WER;S/D/I | headroom pp | front/OFF/tie case wins |
|---|---:|---|---|---|---|---:|---|
'''
for name,d in m['headroom']['strata'].items():report+=f"| {name} | {d['N']} | {word(d['arms']['noisy'])} | {word(d['arms']['frontend'])} | {word(d['arms']['off'])} | {word(d['WER_oracle'])} | {100*d['headroom']:.6f} | {d['frontend_wins']}/{d['off_wins']}/{d['ties']} |\n"
report+='\n## CER and uncapped sentence means\n\n| Stratum | Arm | CER micro | CER sentence mean | WER sentence mean | S/D/I characters |\n|---|---|---:|---:|---:|---|\n'
for name,d in m['headroom']['strata'].items():
 for arm,a in list(d['arms'].items())+[('WER_oracle',d['WER_oracle'])]:
  c=a['character'];report+=f"| {name} | {arm} | {fmt(c['micro_rate'])} | {fmt(c['sentence_mean_rate'])} | {fmt(a['word']['sentence_mean_rate'])} | {c['S']}/{c['D']}/{c['I']} |\n"
report+='\n## Quality: actual full-length PCM, fixed same-scale clean reference\n\nAll96 arms/3 metrics valid: errors0,warnings0, noSTOI sentinel observed. OFF−front paired means all32: SI-SDR **−1.326095dB**, STOI **+0.009964728**, PESQ **+0.062593758**. Direction tradeoff is explicit; these are not A-style human naturalness scores or phone identities.\n\n| Stratum | Metric | noisy mean/n | frontend mean/n | OFF mean/n | OFF−front paired mean/n |\n|---|---|---|---|---|---|\n'
for name,ss in m['quality']['strata'].items():
 for key,d in ss.items():
  cells=[fmt(d[a]['mean'])+'/'+str(d[a]['valid_n']) for a in ['noisy','frontend','off']];p=d['off_minus_frontend_paired'];report+=f"| {name} | {key} | "+' | '.join(cells)+f" | {fmt(p['mean'])}/{p['paired_n']} |\n"
report+='\n## Per-case oracle and actual native observations\n\nEvery32 pair has NFE30 and identity length match. All64 outputs have zero preclip beyond±1, zero endpoint equality and zero serialization saturation/overrange. Real backend norm equals actual native noisy peak for each case. Raw arrays/PCM reencoding proof retained.\n\n| Case | seed | norm | front peak | OFF peak | E front/OFF | fixed oracle arm |\n|---|---:|---:|---:|---:|---|---|\n'
score={i['key']:i['score'] for i in m['items']};choices={x['basename']:x['arm'] for x in m['headroom']['strata']['all']['choices']}
for x in m['native_cases']:
 name=x['basename'];ef=score[name+'__frontend']['word']['edit_distance'];eo=score[name+'__off']['word']['edit_distance'];report+=f"| {name} | {x['actual_torch_seed']} | {x['backend_actual_norm_factor']:.12g} | {x['frontend_preclip']['peak']:.12g} | {x['off_preclip']['peak']:.12g} | {ef}/{eo} | {choices[name]} |\n"
report+='\n## Full104 literal reference/hypotheses and scores\n\nText below is JSON-escaped exact original strings (not normalized paraphrases). Detailed token/character DP alignments and all decoder option records reside in outputs/asr/*_raw.json and *_score.json.\n'
for i in m['items']:
 s=i['score'];report+='\n### '+i['key']+'\n\nReference: '+json.dumps(s['original_reference'],ensure_ascii=False)+'\n\nHypothesis: '+json.dumps(s['original_hypothesis'],ensure_ascii=False)+'\n\nWord S/D/I/N='+ '/'.join(str(s['word'][k]) for k in ['S','D','I','N'])+'; WER='+fmt(s['word']['rate'])+'; character S/D/I/N='+ '/'.join(str(s['character'][k]) for k in ['S','D','I','N'])+'; CER='+fmt(s['character']['rate'])+'; decode API calls='+str(i['decode_api_calls'])+'.\n'
report+='''\n## Disposition

Completed fixed measurement, not a learned selector or validated P8-V. Nonzero proxy pool headroom is a necessary-condition diagnostic only; observed repetition/clean-ASR discrepancies and4 clusters require cautious Lead interpretation. No true hallucination/deletion/recovery labels or new human feedback established. Do not train, extend seeds, rerun old/new samples, alter SNR/windows or select a favorable secondary endpoint based on this report. Worker stops after this report/verification; any new experiment requires new Lead design. Final accounting includes metadata/inputs/model artifacts and excludes symlink target duplication, recorded separately.\n'''
(D/'experiment_report.md').write_text(report)
assert all(r['nfe']==30 and r['native_length_match']['operation']=='identity' for r in m['native_cases'])
for r in m['native_cases']:
 for arm in ['frontend_preclip','off_preclip']:
  assert all(r[arm][k]==0 for k in ['above_one','below_minus_one','at_positive_endpoint','at_negative_endpoint','serialization_above_max_pcm16','serialization_below_min_pcm16'])
count=lambda root:sum(p.stat().st_size for p in root.rglob('*') if p.is_file() and not p.is_symlink())
write('worker_final_accounting.json',{'parent_bytes_before_this_receipt':count(B),'model_stage_bytes_before_this_receipt':count(D),'parent_cap':67108864,'exclude_symlink_target_duplication':True,'receipt_not_in_its_own_prior_total':True,'metrics_sha256':sha(D/'metrics.json'),'report_sha256':sha(D/'experiment_report.md'),'no_model_follow_on':True});assert count(B)<67108864
print(json.dumps({'parent_bytes_current':count(B),'model_stage_bytes_current':count(D),'report_sha256':sha(D/'experiment_report.md'),'metrics_sha256':sha(D/'metrics.json'),'verified_original_artifacts':len(m['artifact_manifest'])}))
