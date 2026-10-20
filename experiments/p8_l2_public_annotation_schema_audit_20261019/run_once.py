from pathlib import Path
import subprocess,hashlib,json,datetime,os,time,sqlite3
B=Path(__file__).resolve().parent; R=Path('/data/coding/sgmse'); C=R/'research_tools/p8_annotation_contract_20261019'
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(n,x): (B/n).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
locks={str(C/f['path']):f['sha256'] for f in json.loads((C/'file_hashes.json').read_text())['files']}; locks[str(C/'file_hashes.json')]=sha(C/'file_hashes.json')
locks[str(R/'experiments/p8_voicebank_native_content_pilot_20261019/attempt02/metrics.json')]='e0ed9b10c646cc19d28cc8c6c6cd0225fd87c1b19b77bac108b1bb19a17066e0'
locks[str(R/'experiments/p8_vb_caseB_opening_existing_audit_20261019/metrics.json')]='a99373256417f29d01ee6f06758586ca06d6806335c52a53c3a096655211bbd7'
for n in ['analysis_plan.md','config.yaml']: locks[str(B/n)]=sha(B/n)
assert all(sha(Path(p))==h for p,h in locks.items())
assert not (B/'metrics.json').exists() and not (B/'raw_textgrid.txt').exists()
conn=sqlite3.connect('file:/data/codex-state/state_5.sqlite?mode=ro',uri=True)
provider=conn.execute("select id,agent_path,model_provider from threads where agent_path='/root/experiment_worker_public_annotation' order by rowid desc limit 1").fetchone(); assert provider[2]=='codex_tools_compat'
assert b'bridge.py' in Path('/proc/39452/cmdline').read_bytes()
git=subprocess.check_output(['git','-C',str(R),'status','--porcelain=v1']); (B/'git_status_before.txt').write_bytes(git)
write('protected_hashes.json',locks)
with (B/'STARTED').open('x') as f: f.write(utc()+'\n')
cmd=['/usr/bin/curl','--proxy','http://127.0.0.1:7890','--header','Accept-Encoding: identity','--max-time','30','--connect-timeout','10','--max-filesize','65536','--retry','0','--max-redirs','0','--proto','=https','--request','GET','--dump-header',str(B/'response_headers.txt'),'--output',str(B/'raw_textgrid.txt'),'--write-out','%{json}\n','https://psi.engr.tamu.edu/wp-content/uploads/2018/03/lxc_arctic_a0018.txt']
start=utc(); t=time.monotonic(); p=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE); pid=p.pid
out,err=p.communicate(); elapsed=time.monotonic()-t
(B/'curl_stdout.txt').write_bytes(out); (B/'curl_stderr.txt').write_bytes(err)
body=B/'raw_textgrid.txt'; response=json.loads(out) if out else {}
receipt={'utc_start':start,'utc_end':utc(),'elapsed_seconds':elapsed,'command_argv':cmd,'supervisor_pid':os.getpid(),'curl_pid_observed_via_popen':pid,'exit_code':p.returncode,'provider':provider,'response_metadata':response,'body_bytes':body.stat().st_size if body.exists() else None,'body_sha256':sha(body) if body.exists() else None,'network_request_attempts':1,'redirects_followed':0,'automatic_retries':0,'wire_bytes_not_measured':True}
write('fetch_provenance.json',receipt)
assert all(sha(Path(p))==h for p,h in locks.items())
assert p.returncode==0 and response.get('http_code')==200 and body.stat().st_size<=65536, 'fetch failed; no retry'
assert body.read_bytes().lstrip().startswith(b'File type = "ooTextFile"'), 'unexpected body'
print(json.dumps({k:receipt[k] for k in ['exit_code','elapsed_seconds','supervisor_pid','curl_pid_observed_via_popen','body_bytes','body_sha256']}))
