from pathlib import Path
import json,time,datetime,hashlib,concurrent.futures,subprocess
HERE=Path(__file__).resolve().parent
# Reuse only request/parser definitions; do not execute original orchestration.
scope={'__file__':str(HERE/'retrieve_sources.py')};exec((HERE/'retrieve_sources.py').read_text().split("cfg={'status'")[0],scope)
progress=json.loads((HERE/'retrieval_progress.json').read_text());scope['records']=progress['records'];scope['payload']=progress['saved_payload_bytes']
cfg=json.loads((HERE/'config.yaml').read_text());elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(cfg['started_utc'])).total_seconds();scope['START']=time.monotonic()-elapsed
commit=json.loads((HERE/'galdse/master_commit.json').read_text())['sha'];paths=['galdse/model.py','galdse/stochastic_process.py','galdse/backbones/TinyNCSNpp.py']
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(lambda p:scope['fallback']('galdse/code/'+p,f'https://raw.githubusercontent.com/wangchengzhong/GALDSE/{commit}/'+p),paths))
records=scope['records'];codebytes=sum(p.stat().st_size for p in (HERE/'galdse/code').rglob('*.py'));localsnap=json.loads((HERE/'local_source_locks.json').read_text());unchanged=all(hashlib.sha256(Path(r['path']).read_bytes()).hexdigest()==r['sha256'] for r in localsnap)
version=max(int(p.stem.split('v')[-1]) for p in (HERE/'papers').glob('galdse_2409.15101v*.html'))
after=subprocess.check_output(['git','status','--porcelain'],cwd=HERE.parent.parent).decode();(HERE/'git_status_after.txt').write_text(after)
before=(HERE/'git_status_before.txt').read_text();filtered=lambda s:'\n'.join(l for l in s.splitlines() if 'p8_evidence_prior_audit_20261019' not in l)
metrics={'status':'completed_with_recorded_transport_failures','retrieval_finished_utc':scope['now'](),'wall_seconds':time.monotonic()-scope['START'],'saved_payload_bytes':scope['payload'],'essential_code_bytes':codebytes,'galdse_commit':commit,'galdse_arxiv_version':version,'records':records,'local_sources_unchanged':unchanged,'git_status_outside_audit_unchanged':filtered(before)==filtered(after),'training_forwards':0,'inference_forwards':0,'note':'Broad utility iteration was intentionally interrupted after targeted ncsnpp/conformer files; no utility was saved. Completed only essential model/TinyNCSNpp/stochastic_process path. No successful payload downloaded twice.'}
assert codebytes<=500000 and metrics['saved_payload_bytes']<=5000000 and metrics['wall_seconds']<=300
scope['dump']('retrieval_metrics.json',metrics)
print(json.dumps({k:v for k,v in metrics.items() if k!='records'},indent=2))
