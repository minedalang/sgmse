"""Static record consolidation; no model imports, no audio decoding."""
from pathlib import Path
import json,hashlib,subprocess
H=Path(__file__).resolve().parent;R=H.parent.parent
D=json.loads((H/'metrics.json').read_text());refs=json.loads((H/'source_references.json').read_text());matched=json.loads((H/'matched_references.json').read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
base=D['original_runtime']['source_basename'];inp=R/'experiments/p8_frontend_backend_audit_20261020/inputs/noisy'/base
D['original_runtime'].update(input_symlink_path=str(inp),input_symlink_exists=inp.exists(),input_symlink_resolves_to=str(inp.resolve()),rrsd=False);D['original_runtime'].pop('rrsd_off',None)
assert inp.resolve()==Path(D['source_record']['source_paths']['noisy'])
replay=R/'experiments/p8_10720_pairing_forensics_20261018/replay_off/attempt02/metrics.json';prior=json.loads(replay.read_text())
D['current_code_matches_successful_10720_replay']={p:sha(R/p)==h for p,h in prior['script_sha256'].items()};assert all(D['current_code_matches_successful_10720_replay'].values())
D['historical_literal_command_and_code_hashes_available']=False
D['prior_interventions']=[{'experiment':'p8_y_pre_mask_counterfactual_20261020','fileid':5243,'target_s':[10.,10.5],'negative_s':[16.5,17.],'branch':'Y_pre only; noisy unchanged','seeds':[2026,2027,2028],'same_original_prefixed_seed':False,'completed':True},{'experiment':'p8_10720_tail_causal_pilot_20261018','fileid':10720,'interval_s':[15.05,15.75],'input_variant':'noise_minus10dB','branch':'condition clean replacement / anchor clean replacement / both','completed':True},{'experiment':'p8_10720_tail_condition20_20261018','fileid':10720,'interval_s':[15.05,15.75],'input_variant':'noise_minus20dB','branch':'condition clean replacement','completed':True},{'experiment':'p8_3321_tail_condition_20261018','fileid':3321,'interval_s':[22.62,23.66],'branch':'condition clean replacement','completed':True}]
scan=[]
for p in sorted((R/'experiments').glob('p8*/config.yaml')):
 if p.parent==H:continue
 text=p.read_text(errors='replace')
 if '5243' in text or any(w in p.parent.name for w in ['tail_causal','tail_condition','y_pre_mask']):
  scan.append({'path':str(p),'sha256':sha(p),'content':text})
D['existing_config_scan_matches']=len(scan);(H/'prior_config_scan.json').write_text(json.dumps(scan,indent=2)+'\n')
D['same_location_direct_noisy_speech_removal_found']=False;D['prior_search_scope']='All experiments/p8*/config.yaml plus relevant P8-M/tail reports and driver references; no completed direct-noisy speech-removal at5243 19.45s found. This is a bounded project-record search, not a claim about undocumented runs.'
D['unmet_criteria']=['Natural5243 no-op exact replay not run or achieved.','No cached full-precision natural5243 frontend found in original export/forensics; recomputation required, PCM anchor is insufficient for an exact native gate.','Original literal historical launch command and contemporaneous source hashes unavailable; reconstruct from config/logs/symlink and current code.','19.35–19.55s is a diagnostic neighborhood, not manually verified isolated /d/ boundaries.','No independent noisy-only human intelligibility judgment or Router-specific human score.']
D['static_eligibility']='Paired components/provenance/seed permit a guarded future replay preparation. Runtime gate must reproduce original full30s and clip PCM before any approved intervention. No runtime permission or scientific direction is implied by this audit.'
for p in [R/'sgmse/model.py',R/'sgmse/joint_data_module.py',R/'sgmse/frontends/storm.py',R/'experiments/p8_q_a_ctc_fixed_time_v2_20261018/metrics.json']:
 refs.append({'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)})
 for i,line in enumerate(p.read_text(errors='replace').splitlines(),1):
  if p.name=='model.py' and any(w in line for w in ['norm_factor =','y_cond /','y / norm_factor','def enhance']):matched.append({'path':str(p),'line':i,'text':line})
(H/'source_references.json').write_text(json.dumps(refs,indent=2)+'\n');(H/'matched_references.json').write_text(json.dumps(matched,indent=2)+'\n')
assert all(sha(Path(r['path']))==r['sha256'] for r in refs)
text='''# Natural5243 evidence feasibility audit

Completed CPU-only; zero model initialization, inference, training, downloads or audio output. Source WAVs were decoded once each in float64; final source-byte hashes were rechecked. The audit reads the original reports/configs/logs and current code without importing model modules.

All clean/noisy/Y_pre/OFF/Router sources are mono WAV PCM16,16kHz,480000 samples (30s). All five source SHA256 values match the registered corrected aligned-package metrics. Each saved 2s clip exactly equals original samples[300000:332000],18.75–20.75s, with zero PCM difference. Full paths, headers, lengths and recomputed hashes are in metrics.json. The actual paired noise_fileid_5243.wav exists, with the same header/length; SHA2562528de81b56c9f1ea11081e5bca604f80c27bb8d34bd36617148b7c2e31ca6a2.

|Range|Component SNR dB|Mixture max error LSB|Mixture RMS error LSB|LS clean gain|LS noise gain|
|---|---:|---:|---:|---:|---:|
'''
for key in ['global','context','target_neighborhood']:
 m=D['mixture'][key];text+=f"|{key} {m['seconds']}|{m['paired_component_snr_db']:.6f}|{m['mixture_error_max_lsb']:.1f}|{m['mixture_error_rms_lsb']:.6f}|{m['least_squares_clean_gain']:.9f}|{m['least_squares_noise_gain']:.9f}|\n"
text+='''
Native mixture decomposition passes the conservative3LSB bound everywhere; actual max is1LSB (3.0517578125e−5). The bound follows three independently quantized PCM16 operands, each <1LSB error, and does not assume exact original float mixtures are retained. There is no evidence that arbitrary component gains are needed: global LS gains are essentially1. Fits are diagnostic only; no replacement noise or variant was generated. The200ms19.35–19.55s neighborhood is samples[309600:312800]. It brackets recognizer frames35:36 (about19.45–19.47s) and is not a manually isolated /d/. Its SNR is a paired-component neighborhood average, not /d/ SNR or proof of noisy recoverability. Exact PCM nonzero support spans the30s file; clean RMS, peak, energy and nonzero counts for all ranges are recorded. Neighborhood clean energy is0.57405% of full clean energy; nonzero support is not phonetic segmentation.

The original saved basename is00_book_02324_chp_0015_reader_10658_7_CGPxaXQVd_U-fan_Freesound_validated_350631_0-oHvYwexeY6k_snr-19_fileid_5243.wav. Original inputs/noisy contains a symlink with exactly that basename to the actual dataset noisy source. Current joint_enhance_by_dir.py290–292 uses base2026 plus little-endian BLAKE2b(digest_size4) of that basename modulo2^32: effective seed369124093. Using the unprefixed dataset basename gives613242433, also wrong for original replay. Original config and off_retry.log establish EMA,PC,N30,none corrector,snr0.5,RRSD false; log reports successful return and EMA enabled. Seeding occurs before full-file frontend computation. Exact literal historical command/source hashes are unavailable; this is reconstruction supported by actual artifacts, not claimed verbatim command recovery.

Static pipeline: joint_enhance_by_dir.py302–312 loads full30s noisy, normalizes by full noisy peak, recomputes full-precision frontend, matches length, and sends Y_pre as anchor/noisy as condition in mode2. Lines316–343 call backend and serialize clipped PCM16. Reusing the saved PCM16 Y_pre as anchor changes the original full-precision frontend. No raw5243 frontend was found in the original audit/10720 replay directories; no exact5243 replay has been performed. The successful10720 replay wrapper supplies a precedent for recomputing frontend and comparing PCM after serialization, with original prefixed basename, seed and full30s context. Current hashes of its five relevant code files exactly match that successful replay's recorded hashes. This supports static compatibility only, not guaranteed5243 runtime equivalence.

P8-M on5243 was Y_pre attenuation at10–10.5s (negative16.5–17s), original noisy unchanged; its seeds2026/2027/2028 are direct seeds, distinct from original prefixed-file seed369124093. It was not direct-noisy speech removal at19.45s. Tail causal work used10720 at15.05–15.75s with noise−10/−20dB variants, and3321 at22.62–23.66s, replacing near-quiet condition/anchor with known clean. Those are distinct samples/locations and interventions. No completed direct-noisy speech-removal at natural5243's target neighborhood was found in all P8 config files plus matched historical driver/report references. prior_config_scan.json and matched_references.json preserve the bounded search evidence. Undocumented runs cannot be excluded.

The existing tail helper contributes a deterministic internal raised-cosine intervention, exact outside-support identity, peak check, frontend float/RNG guard and no-op PCM gate. It is not a ready5243 entry: its source variant, seed, samples and baseline hashes are10720-specific. The2s VCV fixed_norm helper also hardcodes32000 samples and cannot simply replace a full480000-sample natural replay. A future implementation must preserve original full-file frontend context/normalization, paired seed/RNG/initial state and full-precision anchor. Public enhance may derive normalization from condition peak, so any intervention must prove the peak remains identical or use an explicitly frozen reference and first pass an exact no-op gate. No helper was executed here.

Unmet criteria: exact native5243 runtime replay; recovered full-precision frontend; exact original launch provenance; manually verified /d/ boundaries; independent noisy intelligibility and separate Router/off human judgments. Source/component/seed feasibility passes. A guarded replay preparation is statically feasible, conditional on a later approved full30s no-op reproduction of saved frontend and OFF full/clip PCM before any intervention. This audit does not authorize or choose a new experiment.

Evidence levels remain separate: original user observation confirms one prompted, nonblind frontend/backend audible /d/ difference; the fixed-time CTC report locates a proxy frame and does not give manual boundaries. The user did not separately score OFF and Router. The source/mixture audit does not establish noisy independent recovery or hallucination. All report/config/code references are rehashed in source_references.json. Checkpoint hash is inherited from an existing completed immutable lock, explicitly not rehashed/loaded here. Existing dirty files and data are preserved; root research plan was read only.
'''
(H/'experiment_report.md').write_text(text)
D['artifact_bytes']=sum(p.stat().st_size for p in H.rglob('*') if p.is_file());assert D['artifact_bytes']<5000000
(H/'metrics.json').write_text(json.dumps(D,indent=2)+'\n')
print(json.dumps({'status':D['status'],'artifact_bytes':D['artifact_bytes'],'seed':D['original_runtime']['effective_file_seed'],'source_refs':len(refs),'unmet_criteria':D['unmet_criteria']},indent=2))
