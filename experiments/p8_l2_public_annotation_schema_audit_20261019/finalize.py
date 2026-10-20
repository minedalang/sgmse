from pathlib import Path
import hashlib,json,subprocess,datetime
B=Path(__file__).resolve().parent; R=Path('/data/coding/sgmse')
def h(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def put(n,x):(B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
s=json.loads((B/'structure.json').read_text()); f=json.loads((B/'fetch_provenance.json').read_text()); v=json.loads((B/'independent_validation.json').read_text()); locks=json.loads((B/'protected_hashes.json').read_text()); assert all(h(Path(p))==sha for p,sha in locks.items())
git=subprocess.check_output(['git','-C',str(R),'status','--porcelain=v1']); (B/'git_status_after.txt').write_bytes(git)
m={'experiment_id':B.name,'status':'completed_public_single_example_metadata_audit','fetch':{'requests':1,'http_status':f['response_metadata']['http_code'],'curl_exit_code':f['exit_code'],'elapsed_seconds':f['elapsed_seconds'],'body_bytes':f['body_bytes'],'body_sha256':f['body_sha256'],'redirects':0,'retries':0,'wire_bytes_measured':False},'structure':{'global_xmin':s['global_xmin'],'global_xmax':s['global_xmax'],'tier_counts':{t['name']:len(t['intervals']) for t in s['tiers']},'empty_counts':{t['name']:t['empty_label_count'] for t in s['tiers']},'phone_label_counts':{k:x['count'] for k,x in s['phone_label_inventory'].items()},'all_Decimal_times_ranges_continuity_counts_passed':True},'independent_validation':'passed','protected_hashes_post_match':True,'protected_count':len(locks),'git_porcelain_before_sha256':h(B/'git_status_before.txt'),'git_porcelain_after_sha256':h(B/'git_status_after.txt'),'git_porcelain_same':git==(B/'git_status_before.txt').read_bytes(),'calls':{'audio_download':0,'audio_decode':0,'resample':0,'models':0,'ASR':0,'training':0,'annotation_contract_on_actual_labels':0},'limitations':['public single annotation example only, not registered corpus access','publisher annotation assertions not verified against audio','no phonetic truth, candidate coverage, calibration or enhancement hypothesis validated','sp and empty labels preserve raw markers; no silence/speech semantic inference','a/d absent only in this sample; no corpus-wide inference','business and independent parser implemented by same Worker, separate algorithms and no shared parsing functions','prior Root web rendition exposed expected structure before fetch; not blinded independent sample selection']}
put('metrics.json',m)
report='''# Public L2-ARCTIC TextGrid metadata audit — completed

## Authorized scope and execution
One official public annotation example only, explicitly linked by the publisher Annotations paragraph. This is not a route around registration for the full corpus. No forms, login, mirror, full dataset or WAV was accessed. Config and analysis plan remain byte-identical.

- URL: https://psi.engr.tamu.edu/wp-content/uploads/2018/03/lxc_arctic_a0018.txt
- UTC: START to END; elapsed ELAPSED seconds. Actual exec session 24342; supervisor PID 7515 and curl PID 7518 captured by the launcher/Popen, not inferred from a later ps.
- Exactly one `/usr/bin/curl` GET with proxy7890, identity encoding, 30s/10s time limits, 65536-byte body limit, retry0, max-redirs0, HTTPS only and no `-L`. Exact argv, stdout, stderr, response headers and exit code are archived.
- curl exit0, HTTP/2 200, redirects0. Raw body **3579 bytes**, SHA256 **bb2e198e06fb6646f7f60f6548e5ed7d6d558987bad52c4d0e9a278190d88573**. Content-Type text/plain, UTF-8 decoding succeeded. Saved HTTP headers include proxy CONNECT200 followed by origin200. curl-reported body/header sizes are not a measurement of total wire bytes; wire bytes were not measured.

## Complete structure, not imported supervision
Global bounds: 0 to 1.6214965986394558 seconds. Three IntervalTiers, 30 total intervals:

| Tier | Intervals | Empty raw labels |
|---|---:|---:|
| words | 7 | 2 |
| phones | 16 | 1 |
| IPA | 7 | 4 |

Every original tier name/class/index, declared count and interval index/start/end/raw label is retained in structure.json. Decimal checks passed for nonnegative ordered bounds, tier/global bounds, sequential indices, declared counts and exact adjacency continuity. No timing repairs or dropped empty intervals.

Phones inventory: 12 nonempty no-comma labels, 3 substitution triples, 1 empty, 0 addition/deletion triples, 0 unknown comma forms. All three triples are `R,AH0,s`, `Z,S,s`, `JH,CH,s`. The raw `sp` marker is at [1.44,1.6); the empty phones marker is at [1.6,1.6214965986394558). No semantic assertion that `sp` means speech or silence, nor that an empty interval is silence truth. Absence of a/d markers in this one example says nothing about the corpus as a whole. IPA and word tiers are kept separate and not treated as phones.

## Compatibility finding — existing v0 remains untouched
Static source inspection only: the synthetic v0 contract's ASCII-symbol fallback would treat an ordinary `sp` symbol as speech, despite this audit not establishing its semantics. An empty symbol would be rejected; the IPA tier is outside that ASCII phone contract as well. Therefore the real example must not be fed directly into v0 or claimed supported based on synthetic tests. No actual label was passed to the contract and none of its six files or manifest was modified. Source-specific marker policy and approval remain Research Lead decisions.

## Validation and preservation
Independent standard-library validation used a line-state parser, separate from the business regex/block parser; no shared parser imports, functions or regular expressions. It compared all 30 interval records including every raw label and numeric lexeme, and separately rechecked Decimal bounds/continuity/counts and source SHA. Both implementations were written by the same Worker using the same raw source/schema, so this is implementation-level cross-checking, not independent human annotation or a blinded scientific confirmation. Root had already seen a web rendition of this public example before fetch; no independent sample-selection claim.

Protected pre/post: six contract files plus their manifest, old VoiceBank and B-opening metrics, frozen config and analysis plan (11 files) all match. Full git porcelain snapshots are preserved; SAMEGIT. Existing bridge39452 and listening16022 were present in actual preflight and retained, with no service changes. No audio download/decode, resampling, models, ASR, training, waveforms, retries or failure occurred. Only this audit directory was written.

Success means raw public example provenance and structure passed. It does not establish realized phonetic truth, candidate completeness, P8-V calibration, training permission or enhancement improvement. Full-corpus registration remains required separately. No follow-up fetch is authorized by this report.

## Reproduction and evidence
The historical network argv is in fetch_provenance.json and must not be replayed as routine reproduction. Recheck cached data offline with `python3 -B audit_structure.py` and `python3 -B validate_independent.py`; source raw bytes are immutable evidence. Original execution scripts and their hashes are in artifact_manifest.json. No software contract import is needed.
'''.replace('START',f['utc_start']).replace('END',f['utc_end']).replace('ELAPSED',str(f['elapsed_seconds'])).replace('SAMEGIT','before/after tracked+untracked path-status listing unchanged' if m['git_porcelain_same'] else 'before/after listing differs; see raw snapshots, not attributed to this Worker')
(B/'experiment_report.md').write_text(report)
put('artifact_manifest.json',{'files':[{'path':str(p.relative_to(B)),'bytes':p.stat().st_size,'sha256':h(p)} for p in sorted(B.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','final_receipt.json']]})
size=sum(p.stat().st_size for p in B.rglob('*') if p.is_file()); assert size+2000<2000000
put('final_receipt.json',{'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'measured_directory_bytes_before_this_receipt':size,'limit_bytes':2000000,'protected_hashes_current_match':all(h(Path(p))==sha for p,sha in locks.items()),'metrics_sha256':h(B/'metrics.json'),'report_sha256':h(B/'experiment_report.md')})
print(json.dumps({'status':m['status'],'directory_bytes_final':sum(p.stat().st_size for p in B.rglob('*') if p.is_file()),'metrics_sha256':h(B/'metrics.json'),'report_sha256':h(B/'experiment_report.md'),'git_same':m['git_porcelain_same']}))
