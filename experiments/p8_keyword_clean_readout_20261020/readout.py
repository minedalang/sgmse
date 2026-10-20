"""One authorized archive acquisition and fixed64 clean Whisper readout only."""
from pathlib import Path,PurePosixPath
import argparse,collections,hashlib,io,json,os,re,shutil,signal,ssl,sys,time,traceback,wave,zipfile,zlib
BASE=Path(__file__).resolve().parent
OLD=Path('/data/coding/sgmse/experiments/p8_voicebank_clean_text_calibration_20261019/config.yaml')
WORDS=['down','go','left','no','right','stop','up','yes']

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def write(path,data):Path(path).write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def require(ok,message):
    if not ok:raise RuntimeError(message)

def safe_member(name):
    p=PurePosixPath(name)
    require(not p.is_absolute() and '..' not in p.parts and '\\' not in name and '\x00' not in name,'Unsafe archive member')
    return p

def partition(group):
    pct=(int(hashlib.sha1(group.encode()).hexdigest(),16)%(2**27))*100/(2**27-1)
    return 'validation' if pct<10 else 'testing' if pct<20 else 'training'

def select(entries):
    groups=collections.defaultdict(lambda:collections.defaultdict(list))
    for row in entries:
        name=row['filename'];p=safe_member(name)
        if len(p.parts)!=3 or p.parts[0]!='mini_speech_commands' or p.parts[1] not in WORDS or p.suffix!='.wav':continue
        group=p.name.split('_nohash_')[0];groups[group][p.parts[1]].append(row)
    full={s:[g for g,v in groups.items() if set(v)==set(WORDS) and partition(g)==s] for s in ['validation','testing']}
    chosen=sorted(full['validation'],key=lambda g:hashlib.sha256(('p8-word-calibration-v1|'+g).encode()).hexdigest())[:8]
    require(len(chosen)==8,'Not enough complete validation groups')
    rows=[]
    for group in chosen:
        for word in WORDS:
            row=min(groups[group][word],key=lambda r:r['filename'])
            rows.append(dict(label=word,group=group,**row))
    return rows,full

def score(text,label):
    tokens=re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",text.replace('\u2019',"'").replace('\u2018',"'").lower())
    return dict(raw_text=text,tokens=tokens,normalized=' '.join(tokens),exact_label=tokens==[label],
                empty=not tokens,multiword=len(tokens)>1,out_of_vocabulary=any(t not in WORDS for t in tokens))

def gate():
    cfg=json.loads((BASE/'config.yaml').read_text());auth=json.loads((BASE/cfg['authorization_file']).read_text())
    require(auth['status']=='authorized','Root execution authorization absent')
    require(auth['config_sha256']==sha(BASE/'config.yaml') and auth['code_sha256']==sha(BASE/'readout.py'),'Code/config lock mismatch')
    require(auth['manifest_sha256']==sha(BASE/cfg['manifest_file']) and auth['registry_receipt_sha256']==sha(BASE/cfg['registry_receipt']),'Selection lock mismatch')
    for key in ['entries','central_directory']:
        path=BASE/cfg['source'][key+'_path'];expected=cfg['source'][key+'_sha256']
        require(sha(path)==expected and auth['source_sha256'][key]==expected,'Source metadata lock mismatch')
    manifest=json.loads((BASE/cfg['manifest_file']).read_text())
    rows,full=select(json.loads((BASE/cfg['source']['entries_path']).read_text()))
    require(rows==manifest['records'] and len(full['validation'])==9 and len(full['testing'])==8,'Fixed metadata selection mismatch')
    package=cfg['asr']['package_path'];locks={p:h for p,h in json.loads(OLD.read_text())['source_locks'].items() if p.startswith(package+'/') and p.endswith('.py')}
    require(bool(locks) and locks==auth['asr_package_sha256'],'Exact legacy Whisper source lock mismatch')
    for p,h in locks.items():require(sha(p)==h,'Installed Whisper source changed')
    require(auth['asr_model_sha256']==cfg['asr']['model_sha256'],'ASR weight authorization mismatch')
    require(not (BASE/'STARTED.json').exists() and not (BASE/'metrics.json').exists(),'Existing execution: no retry')
    return cfg,rows,locks

def execute(cfg,rows,locks):
    start=time.monotonic();out=BASE/'metrics.json';source=cfg['source'];limits=cfg['limits']
    result=dict(status='running',items=[],http=dict(requests=0,body_bytes=0),transcribe_calls=0,internal_decode_calls=0,
        asr_model_loads=0,frontend_calls=0,backend_calls=0,optimizer_steps=0,network_after_archive=[],failures=[],
        extracted_members=0,member_read_attempts=0,selected_audio_bytes=0,header_checked=0)
    def guard(reserve=0):
        size=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
        result['artifact_bytes']=size;result['disk_free_bytes']=shutil.disk_usage(BASE).free
        require(size+reserve<=209715200,'200MiB artifact cap')
        require(shutil.disk_usage(BASE).free-reserve>=1073741824,'1GiB disk reserve')
    def timeout(signum,frame):raise TimeoutError('Authorized download/total wall-clock cap')
    prior=signal.signal(signal.SIGALRM,timeout);signal.alarm(300)
    try:
        require(source['archive_bytes']==182082353 and limits['network_body_bytes']==182082353,'Fixed source budget changed')
        require(source['url']=='https://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip','Unexpected source URL')
        guard(182082353+3145728+8*1048576)
        with (BASE/'STARTED.json').open('x') as f:json.dump(dict(pid=os.getpid(),config_sha256=sha(BASE/'config.yaml'),code_sha256=sha(BASE/'readout.py')),f)
        import urllib.request
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self,*a,**k):raise RuntimeError('Redirect forbidden; no retry/mirror')
        opener=urllib.request.build_opener(NoRedirect(),urllib.request.HTTPSHandler(context=ssl.create_default_context()))
        archive=BASE/cfg['outputs']['source_archive'];download=time.monotonic();h=hashlib.sha256();signal.alarm(120)
        result['http']['requests']=1
        with opener.open(urllib.request.Request(source['url'],headers={'Accept-Encoding':'identity'}),timeout=20) as response:
            result['http'].update(status=response.status,headers=dict(response.headers),url=response.geturl())
            require(response.status==200 and response.geturl()==source['url'],'Unexpected HTTP source/status')
            require(response.headers.get('Content-Length')==str(182082353) and response.headers.get('ETag')==source['etag'],'Length/ETag mismatch')
            require(response.headers.get('Content-Encoding') in (None,'identity'),'Encoded HTTP body forbidden')
            with archive.open('xb') as f:
                while result['http']['body_bytes']<182082353:
                    block=response.read(min(1048576,182082353-result['http']['body_bytes']))
                    require(bool(block),'Truncated source archive');f.write(block);h.update(block);result['http']['body_bytes']+=len(block)
                    require(time.monotonic()-download<=120,'Download exceeded120s');guard()
        result['http'].update(seconds=time.monotonic()-download,sha256=h.hexdigest())
        signal.alarm(max(1,int(300-(time.monotonic()-start))))
        require(archive.stat().st_size==182082353,'Archive size mismatch')
        import socket
        def deny(sock,address,*a,**k):
            result['network_after_archive'].append(repr(address));raise RuntimeError('Network forbidden after archive')
        socket.socket.connect=deny;socket.socket.connect_ex=deny
        saved=[];total=0;folder=BASE/cfg['outputs']['selected_wav_directory'];folder.mkdir()
        with zipfile.ZipFile(archive) as z:
            require(len({i.filename for i in z.infolist()})==len(z.infolist()),'Duplicate archive member')
            for index,row in enumerate(rows):
                safe_member(row['filename']);info=z.getinfo(row['filename'])
                actual=dict(filename=info.filename,size=info.file_size,compressed_size=info.compress_size,CRC32=f'{info.CRC:08x}',
                    local_header_offset=info.header_offset,flags=info.flag_bits,compression=info.compress_type,external_attributes=info.external_attr)
                require(all(actual[k]==row[k] for k in actual),'Selected member differs from frozen central directory')
                require(info.file_size<=65536 and not info.flag_bits&1,'Member size/encryption disallowed')
                result['member_read_attempts']+=1
                raw=z.read(info);total+=len(raw);result['extracted_members']+=1;result['selected_audio_bytes']+=len(raw)
                require(total<=3145728 and len(raw)==info.file_size,'Selected audio byte cap')
                require(f'{zlib.crc32(raw)&0xffffffff:08x}'==row['CRC32'],'Selected raw CRC mismatch')
                with wave.open(io.BytesIO(raw)) as wav:
                    result['header_checked']+=1
                    require(wav.getnchannels()==1 and wav.getsampwidth()==2 and wav.getframerate()==16000 and wav.getcomptype()=='NONE','Expected mono PCM16 16k')
                    samples=wav.getnframes();require(1<=samples<=16000,'Unexpected sample count')
                path=folder/f'{index:02d}_{row["label"]}_{row["group"]}.wav'
                with path.open('xb') as f:f.write(raw)
                saved.append((row,path,samples));guard()
        result.update(reserved_member_audio_reads=0,
            archive_contains_reserved_compressed_bytes=True)
        sys.dont_write_bytecode=True
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',NUMBA_CACHE_DIR=str(BASE/'runtime_cache/numba'),
            XDG_CACHE_HOME=str(BASE/'runtime_cache'),TORCH_EXTENSIONS_DIR=str(BASE/'runtime_cache/torch'),TRITON_CACHE_DIR=str(BASE/'runtime_cache/triton'))
        require(sha(cfg['asr']['model_path'])==cfg['asr']['model_sha256'],'ASR weights differ')
        import random,numpy as np,torch,whisper,dataclasses
        import whisper.decoding as decoding
        require(Path(whisper.__file__).resolve()==Path(cfg['asr']['package_path'])/'__init__.py','Unexpected Whisper package')
        def no_download(*a,**k):raise RuntimeError('ASR weight fetch forbidden')
        whisper._download=no_download
        seed=cfg['asr']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        options=dict(cfg['asr']['transcribe_options']);options['temperature']=tuple(options['temperature'])
        require(options==dict(language='en',task='transcribe',temperature=(0.,),beam_size=1,best_of=1,condition_on_previous_text=False,
            initial_prompt=None,no_speech_threshold=None,verbose=False,word_timestamps=False,fp16=False),'Readout options changed')
        expected=dataclasses.asdict(decoding.DecodingOptions(language='en',task='transcribe',temperature=0.,beam_size=1,best_of=None,prompt=[],fp16=False))
        result['effective_decode_options']=expected
        model=whisper.load_model(cfg['asr']['model_path'],device=cfg['asr']['device']);model.eval();result['asr_model_loads']=1
        require(next(model.parameters()).dtype==torch.float32,'Expected float32 ASR')
        original=model.decode
        def decode(audio,decode_options=decoding.DecodingOptions(),**kwargs):
            result['internal_decode_calls']+=1
            require(dataclasses.asdict(decode_options)==expected,'Actual decoding options changed')
            return original(audio,decode_options,**kwargs)
        model.decode=decode
        for row,path,samples in saved:
            guard();require(result['transcribe_calls']<64,'Transcribe cap')
            with wave.open(str(path)) as wav:pcm=wav.readframes(samples)
            audio=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768.
            require(len(audio)==samples and np.isfinite(audio).all(),'Raw input mismatch')
            before=result['internal_decode_calls'];result['transcribe_calls']+=1
            with torch.inference_mode():prediction=model.transcribe(audio,**options)
            torch.cuda.synchronize();require(isinstance(prediction.get('text'),str),'Missing raw transcript: technical failure')
            result['items'].append(dict(**row,samples=samples,wav_sha256=sha(path),decode_calls=result['internal_decode_calls']-before,
                **score(prediction['text'],row['label'])))
            write(out,result)
        require(len(result['items'])==result['transcribe_calls']==64,'Incomplete readout')
        per={w:dict(correct=sum(r['exact_label'] for r in result['items'] if r['label']==w),total=8,
            accuracy=sum(r['exact_label'] for r in result['items'] if r['label']==w)/8) for w in WORDS}
        correct=sum(r['exact_label'] for r in result['items']);result['summary']=dict(correct=correct,total=64,accuracy=correct/64,per_word=per,
            applicability_passed=correct>=58 and all(v['correct']>=6 for v in per.values()))
        require(not result['network_after_archive'],'Blocked unexpected network attempt occurred')
        for p,h0 in locks.items():require(sha(p)==h0,'Whisper source changed during run')
        require(sha(cfg['asr']['model_path'])==cfg['asr']['model_sha256'],'Source ASR weight changed')
        result['status']='completed_fixed_clean_readout'
    except BaseException as error:
        result['status']='failed_technical';result['failures'].append(dict(error=repr(error),traceback=traceback.format_exc()))
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,prior);result['seconds']=time.monotonic()-start
        if result['seconds']>300:result['status']='failed_technical';result['failures'].append(dict(error='Total300s exceeded'))
        try:
            write(out,result);guard()
        except BaseException as error:
            result['status']='failed_technical'
            result['failures'].append(dict(error='Final resource/output failure: '+repr(error)))
            result['artifact_bytes']=sum(p.stat().st_size for p in BASE.rglob('*') if p.is_file() and not p.is_symlink())
            write(out,result)
    return 0 if result['status']=='completed_fixed_clean_readout' else 1

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute-once',action='store_true',required=True);p.parse_args()
    cfg,rows,locks=gate();raise SystemExit(execute(cfg,rows,locks))
