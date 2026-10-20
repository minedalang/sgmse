from pathlib import Path
import urllib.request,urllib.error,json,hashlib,time,datetime,subprocess,concurrent.futures,threading
from html.parser import HTMLParser
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
START=time.monotonic();lock=threading.Lock();records=[];payload=0
CAP=5_000_000

def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(b):return hashlib.sha256(b).hexdigest()
def dump(name,obj):(HERE/name).write_text(json.dumps(obj,indent=2)+'\n')
class Parser(HTMLParser):
 def __init__(self):super().__init__();self.parts=[];self.skip=0
 def handle_starttag(self,t,a):
  if t in ('script','style'):self.skip+=1
  if t in ('p','h1','h2','h3','div','li','tr'):self.parts.append('\n')
 def handle_endtag(self,t):
  if t in ('script','style'):self.skip=max(0,self.skip-1)
 def handle_data(self,d):
  if not self.skip:self.parts.append(d)

def fetch(name,url,proxy=False,limit=1_500_000):
 global payload
 if time.monotonic()-START>280:raise RuntimeError('Wall cap before request')
 r={'name':name,'url':url,'retrieved_utc':now(),'proxy':proxy}
 try:
  opener=urllib.request.build_opener(urllib.request.ProxyHandler({'http':'http://127.0.0.1:7890','https':'http://127.0.0.1:7890'} if proxy else {}))
  req=urllib.request.Request(url,headers={'User-Agent':'SpeechEnhancementReadOnlyAudit/1.0','Accept':'application/vnd.github+json' if 'api.github.com' in url else '*/*'})
  with opener.open(req,timeout=18) as response:
   r.update(http_status=response.status,final_url=response.url,content_type=response.headers.get('Content-Type'))
   data=response.read(limit+1)
  if len(data)>limit:raise RuntimeError('Individual payload cap')
  with lock:
   if payload+len(data)>CAP:raise RuntimeError('Total payload cap')
   payload+=len(data)
  (HERE/name).parent.mkdir(parents=True,exist_ok=True);(HERE/name).write_bytes(data)
  r.update(bytes=len(data),sha256=sha(data),status='saved')
  if name.endswith('.html'):
   p=Parser();p.feed(data.decode('utf8',errors='replace'));(HERE/name.replace('.html','.txt')).write_text(''.join(p.parts))
 except Exception as exc:
  r.update(status='failed',error=str(exc),http_status=getattr(exc,'code',None))
 with lock:records.append(r);dump('retrieval_progress.json',{'records':records,'saved_payload_bytes':payload,'wall_seconds':time.monotonic()-START})
 return r

def fallback(name,url):
 r=fetch(name,url)
 if r['status']=='failed':r=fetch(name,url,True)
 return r

cfg={'status':'retrieving','budget_seconds':300,'saved_payload_cap_bytes':CAP,'essential_code_cap_bytes':500000,'training':False,'inference':False,'downloads':'public primary HTML/docs/targeted code only','source_identifiers':['2509.19495v1','2511.13300v1','2609.04245v1','2409.15101 latest','wangchengzhong/GALDSE master'],'started_utc':now()}
dump('config.yaml',cfg)
status=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT).decode();(HERE/'git_status_before.txt').write_text(status)
tracked=subprocess.check_output(['git','ls-files','-z'],cwd=ROOT).split(b'\0');existing={str(ROOT/p.decode()):sha((ROOT/p.decode()).read_bytes()) for p in tracked if p and (ROOT/p.decode()).is_file()}
localpaths=[ROOT/'experiments/p8_vcv_speech_component_ablation_20261019/experiment_report.md',ROOT/'experiments/p8_vcv_speech_component_ablation_20261019/metrics.json',ROOT/'experiments/p8_q_a_ctc_fixed_time_v2_20261018/experiment_report.md',ROOT/'experiments/p8_frontend_backend_audit_20261020/user_listening_observation.md']
localsnap=[{'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p.read_bytes())} for p in localpaths];dump('local_source_locks.json',localsnap)
jobs=[('papers/artifree_2509.19495v1.html','https://arxiv.org/html/2509.19495v1'),('papers/pase_2511.13300v1.html','https://arxiv.org/html/2511.13300v1'),('papers/grounded_2609.04245v1.html','https://arxiv.org/html/2609.04245v1'),('papers/galdse_abs.html','https://arxiv.org/abs/2409.15101'),('galdse/master_commit.json','https://api.github.com/repos/wangchengzhong/GALDSE/commits/master')]
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
 list(pool.map(lambda x:fallback(*x),jobs))
# Resolve latest arXiv version from abs page, then request that exact HTML.
import re
absfile=HERE/'papers/galdse_abs.html';versions=[]
if absfile.exists():versions=re.findall(r'2409\.15101v(\d+)',absfile.read_text())
version=max(map(int,versions)) if versions else None
fallback('papers/galdse_2409.15101'+(f'v{version}' if version else '_latest')+'.html','https://arxiv.org/html/2409.15101'+(f'v{version}' if version else ''))
commitfile=HERE/'galdse/master_commit.json';commit=None;codebytes=0
if commitfile.exists():
 commit=json.loads(commitfile.read_text())['sha']
 for name,url in [('galdse/tree.json',f'https://api.github.com/repos/wangchengzhong/GALDSE/git/trees/{commit}?recursive=1'),('galdse/README.md',f'https://raw.githubusercontent.com/wangchengzhong/GALDSE/{commit}/README.md')]:fallback(name,url)
 treefile=HERE/'galdse/tree.json'
 if treefile.exists():
  tree=json.loads(treefile.read_text());paths=[r for r in tree['tree'] if r['type']=='blob'];dump('galdse/tree_selection_candidates.json',[r for r in paths if any(w in r['path'].lower() for w in ['model','ncsn','backbone','sde'])])
  # Tree-driven precise implementation evidence; no generic package retrieval.
  wanted=[r for r in paths if r['path'].endswith('.py') and (r['path'].endswith('/model.py') or r['path'].endswith('/stochastic_process.py') or ('backbones/' in r['path'] and r['path'].split('/')[-1] in ['ncsnpp.py','conformer.py','TinyNCSNpp.py'])) ]
  dump('galdse/selected_code_paths.json',wanted)
  for r in wanted:
   if codebytes+r.get('size',500001)>500000:continue
   rr=fallback('galdse/code/'+r['path'],f'https://raw.githubusercontent.com/wangchengzhong/GALDSE/{commit}/'+r['path'])
   if rr['status']=='saved':codebytes+=rr['bytes']
unchanged=all(Path(p).exists() and sha(Path(p).read_bytes())==h for p,h in existing.items())
localunchanged=all(sha(Path(r['path']).read_bytes())==r['sha256'] for r in localsnap)
dump('retrieval_metrics.json',{'status':'completed_with_failures' if any(r['status']=='failed' for r in records) else 'completed','retrieval_finished_utc':now(),'wall_seconds':time.monotonic()-START,'saved_payload_bytes':payload,'essential_code_bytes':codebytes,'galdse_commit':commit,'galdse_arxiv_version':version,'records':records,'tracked_existing_files_unchanged':unchanged,'local_sources_unchanged':localunchanged,'training_forwards':0,'inference_forwards':0})
print(json.dumps({'payload':payload,'wall':time.monotonic()-START,'commit':commit,'version':version,'codebytes':codebytes,'failures':[r for r in records if r['status']=='failed']},indent=2))
