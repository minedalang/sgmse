from pathlib import Path,PurePosixPath
import json,time,os,datetime,hashlib,urllib.request,urllib.parse,zipfile,stat,zlib,signal,traceback
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def write(n,x):(D/n).write_text(json.dumps(x,indent=2)+'\n')
def main():
 assert not (D/'STARTED.json').exists();t=time.monotonic();signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError('90s total')));signal.alarm(90)
 with (D/'STARTED.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'pid':os.getpid()},f)
 cfg=json.loads((D/'config.yaml').read_text());requests=[];total=0
 class NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*a,**k):raise RuntimeError('redirect forbidden')
 opener=urllib.request.build_opener(urllib.request.ProxyHandler({'https':'http://127.0.0.1:7890'}),NoRedirect())
 def get(url,name):
  nonlocal total
  parsed=urllib.parse.urlsplit(url);assert parsed.scheme=='https' and parsed.netloc=='datashare.ed.ac.uk' and not parsed.username and not parsed.fragment;assert len(requests)<3 and time.monotonic()-t<90
  row={'url':url,'method':'GET','attempt':len(requests)+1,'status':None,'headers':{},'body_bytes':0};requests.append(row);write('http_requests.json',requests)
  try:
   with opener.open(urllib.request.Request(url,headers={'Accept-Encoding':'identity','User-Agent':'metadata-source-audit/1.0'}),timeout=20) as r:
    row['status']=r.status;row['headers']=dict(r.headers);assert r.status==200 and r.geturl()==url;assert r.headers.get('Content-Encoding','identity')=='identity';chunks=[]
    while True:
     b=r.read(min(65536,524288-total+1))
     if not b:break
     total+=len(b);row['body_bytes']+=len(b);chunks.append(b);assert total<=524288
    data=b''.join(chunks);(D/name).write_bytes(data);row['body_sha256']=sha(data);row['artifact']=name;write('http_requests.json',requests);return data
  except BaseException as e:row['error']=repr(e);write('http_requests.json',requests);raise
 try:
  raw=get(cfg['first_metadata_url'],'bundles_metadata.json');obj=json.loads(raw);bundles=obj['_embedded']['bundles'];assert obj['page']['totalPages']==1;original=[b for b in bundles if b['name']=='ORIGINAL'];assert len(original)==1;u=original[0]['_links']['bitstreams']['href'];parts=urllib.parse.urlsplit(u);q=urllib.parse.parse_qsl(parts.query);assert not any(k=='page' and v!='0' for k,v in q)
  if not any(k=='size' for k,v in q):q.append(('size','20'))
  u=urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(q)));raw=get(u,'bitstreams_metadata.json');bs=json.loads(raw);assert bs['page']['totalPages']==1;matches=[b for b in bs['_embedded']['bitstreams'] if b['name']=='logfiles.zip'];assert len(matches)==1;bit=matches[0];assert bit['sizeBytes']<=262144 and bit['checkSum']['checkSumAlgorithm'].upper()=='MD5' and bit['checkSum']['value']==cfg['expected_archive_md5_from_publisher_page'];data=get(bit['_links']['content']['href'],'logfiles.zip');assert len(data)==bit['sizeBytes'] and hashlib.md5(data).hexdigest()==cfg['expected_archive_md5_from_publisher_page']
  # Directory only inspection; decompress only exact train28 basename.
  with zipfile.ZipFile(D/'logfiles.zip') as z:
   inventory=[{'name':x.filename,'size':x.file_size,'compressed_size':x.compress_size,'CRC32':format(x.CRC,'08x'),'compression':x.compress_type} for x in z.infolist()];write('zip_directory_metadata.json',inventory);target=[x for x in z.infolist() if PurePosixPath(x.filename).name=='log_trainset_28spk.txt'];assert len(target)==1;x=target[0];p=PurePosixPath(x.filename);assert not p.is_absolute() and '..' not in p.parts and '\\' not in x.filename and not x.is_dir() and not stat.S_ISLNK(x.external_attr>>16) and not (x.flag_bits&1) and x.compress_type in (0,8) and x.file_size<=2097152
   train=z.read(x);assert len(train)==x.file_size and zlib.crc32(train)&0xffffffff==x.CRC;(D/'log_trainset_28spk.txt').write_bytes(train)
  result={'status':'fetched_training_entry_only','seconds':time.monotonic()-t,'requests':len(requests),'network_body_bytes':total,'archive_bytes':len(data),'archive_md5':hashlib.md5(data).hexdigest(),'archive_sha256':sha(data),'train_entry':x.filename,'train_bytes':len(train),'train_crc32':format(x.CRC,'08x'),'train_sha256':sha(train),'opened_zip_payloads':[x.filename],'test_or_56_payloads_opened':0,'audio_decodes':0,'models':0,'ASR':0,'training':0,'retries':0};write('fetch_result.json',result);print(json.dumps(result))
 except BaseException as e:write('failure.json',{'status':'failed','error':repr(e),'traceback':traceback.format_exc(),'requests':requests,'network_body_bytes':total,'seconds':time.monotonic()-t});print('FAILED',repr(e));raise
 finally:signal.alarm(0)
if __name__=='__main__':main()
