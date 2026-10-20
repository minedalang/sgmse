"""Bounded four-file official/local waveform gate. Safe metadata-only import."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, html, json, math, os, re, signal, struct, subprocess, sys, time, traceback, zlib
ROOT=Path('/data/coding/sgmse');OUT=Path(__file__).resolve().parent
NET_CAP=8_000_000;ART_CAP=12_000_000;TIME_CAP=180;REQUEST_CAP=16;MEMBER_CAP=2_000_000;CD_CAP=3_000_000

def require(ok,message):
    if not ok: raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def scalar(value):
    if hasattr(value,'item'):return value.item()
    if isinstance(value,Path):return str(value)
    raise TypeError(type(value).__name__)

def read(path):return json.loads(Path(path).read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,default=scalar,allow_nan=False)+'\n')
def artifact_size():return sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())
def check():
    require(time.monotonic()<float(os.environ['WAVE_GATE_DEADLINE']),'180 second deadline')
    require(artifact_size()<ART_CAP-100_000,'12MB artifact cap')

def publisher_metadata(text,guid):
    found=[]
    def scan(value):
        if isinstance(value,dict):
            if value.get('uuid')==guid and 'sizeBytes' in value:found.append(value)
            for child in value.values():scan(child)
        elif isinstance(value,list):
            for child in value:scan(child)
    for script in re.findall(r'<script[^>]*>(.*?)</script>',text,re.S):
        try:value=json.loads(html.unescape(script))
        except (ValueError,TypeError):continue
        scan(value)
    require(found and all(row==found[0] for row in found),'Official embedded bitstream metadata missing/conflicting')
    return found[0]

def safe_name(raw,flags):
    name=raw.decode('utf-8' if flags&0x800 else 'cp437')
    require(name and '\x00' not in name and '\\' not in name and ':' not in name,'Unsafe ZIP name')
    p=PurePosixPath(name);require(not p.is_absolute() and '..' not in p.parts and name.lstrip('/')==name,'Unsafe ZIP path')
    return name

def extra_check(extra):
    pos=0
    while pos<len(extra):
        require(pos+4<=len(extra),'Truncated ZIP extra')
        kind,n=struct.unpack_from('<HH',extra,pos);pos+=4
        require(pos+n<=len(extra),'Truncated ZIP extra value')
        require(kind!=1,'Unsupported ZIP64 extra field');pos+=n

def parse_eocd(suffix,total):
    at=suffix.rfind(b'PK\x05\x06');require(at>=0 and at+22<=len(suffix),'ZIP32 EOCD missing')
    sig,disk,cd_disk,on_disk,count,length,offset,comment=struct.unpack_from('<4s4H2LH',suffix,at)
    require(at+22+comment==len(suffix),'Unsupported EOCD trailing bytes/comment')
    require(disk==cd_disk==0 and on_disk==count,'Unsupported multi-disk ZIP')
    require(count!=0xffff and length!=0xffffffff and offset!=0xffffffff,'Unsupported ZIP64 EOCD')
    require(b'PK\x06\x07' not in suffix[max(0,at-20):at],'Unsupported ZIP64 locator')
    require(0<length<=CD_CAP,'Central directory exceeds 3MB')
    absolute_eocd=total-len(suffix)+at
    require(offset+length==absolute_eocd and offset>0,'Unsupported central layout')
    return {'entries':count,'size':length,'offset':offset,'eocd_offset':absolute_eocd,'comment_bytes':comment}

def parse_central(data,eocd,selected):
    pos=0;found={};entries=0
    while pos<len(data):
        require(pos+46<=len(data),'Truncated central header')
        f=struct.unpack_from('<4s6H3L5H2L',data,pos)
        require(f[0]==b'PK\x01\x02','Unsupported central record')
        made,needed,flags,method,crc,compressed,uncompressed,nlen,xlen,clen,disk,external,offset=f[1],f[2],f[3],f[4],f[7],f[8],f[9],f[10],f[11],f[12],f[13],f[15],f[16]
        require(needed<=20 and disk==0 and compressed!=0xffffffff and uncompressed!=0xffffffff and offset!=0xffffffff,'Unsupported ZIP version/ZIP64/disk')
        end=pos+46+nlen+xlen+clen;require(end<=len(data),'Truncated central variable fields')
        name=safe_name(data[pos+46:pos+46+nlen],flags);extra_check(data[pos+46+nlen:pos+46+nlen+xlen])
        if name in selected:
            require(name not in found,'Duplicate selected ZIP member')
            require(method in (0,8) and flags&~0x80e==0,'Unsupported compression/encryption/flags')
            require(0<compressed<=MEMBER_CAP and 0<uncompressed<=MEMBER_CAP,'Selected member exceeds 2MB')
            mode=(external>>16)&0xffff;require(mode&0o170000!=0o120000,'Selected member is symlink')
            require(offset+30+compressed<eocd['offset'],'Member overlaps directory')
            found[name]={'name':name,'name_bytes_hex':data[pos+46:pos+46+nlen].hex(),'version_needed':needed,'flags':flags,'method':method,'crc32':crc,'compressed_size':compressed,'uncompressed_size':uncompressed,'local_offset':offset}
        entries+=1;pos=end
    require(entries==eocd['entries'] and set(found)==set(selected),'Selected members/count mismatch')
    return found

def local_header(data,row):
    require(len(data)==30,'Local fixed header size')
    f=struct.unpack('<4s5H3L2H',data)
    require(f[0]==b'PK\x03\x04' and f[1]==row['version_needed'] and f[2]==row['flags'] and f[3]==row['method'],'Local/central header mismatch')
    if f[2]&8:
        require(f[6] in (0,row['crc32']) and f[7] in (0,row['compressed_size']) and f[8] in (0,row['uncompressed_size']),'Descriptor local size/CRC mismatch')
    else:require((f[6],f[7],f[8])==(row['crc32'],row['compressed_size'],row['uncompressed_size']),'Local size/CRC mismatch')
    require(0<f[9]<=4096 and f[10]<=65535,'Local variable header size cap')
    return f[9],f[10]

def decompress_member(payload,row):
    require(len(payload)==row['compressed_size'],'Compressed payload size mismatch')
    if row['method']==0:raw=payload
    else:
        dec=zlib.decompressobj(-15);raw=dec.decompress(payload,row['uncompressed_size']+1)
        require(dec.eof and not dec.unused_data and not dec.unconsumed_tail,'Invalid/bounded deflate stream')
    require(len(raw)==row['uncompressed_size']<=MEMBER_CAP,'Uncompressed member size mismatch')
    require((zlib.crc32(raw)&0xffffffff)==row['crc32'],'Member CRC32 mismatch')
    return raw

class RangeReader:
    def __init__(self,cfg,state,destination,checkpoint):
        self.cfg=cfg;self.state=state;self.destination=destination;self.checkpoint=checkpoint;self.validators={};self.session=None
    def open_response(self,headers):
        import requests
        if self.session is None:
            self.session=requests.Session();self.session.trust_env=False
            self.session.proxies={'http':self.cfg['proxy'],'https':self.cfg['proxy']}
            adapter=requests.adapters.HTTPAdapter(max_retries=0);self.session.mount('https://',adapter)
        remaining=float(os.environ['WAVE_GATE_DEADLINE'])-time.monotonic()
        return self.session.get(self.cfg['content_url'],headers=headers,stream=True,allow_redirects=False,timeout=(min(10,remaining),min(15,remaining)))
    def get(self,start,end,label):
        self.checkpoint();n=end-start+1
        require(0<=start<=end<self.cfg['archive_size'] and n<=CD_CAP,'Invalid/big Range request')
        require(len(self.state['requests'])<REQUEST_CAP,'16 request cap')
        require(self.state['network_bytes_accounted']+n+65536<=NET_CAP,'8MB network budget reservation')
        record={'index':len(self.state['requests']),'label':label,'requested_range':[start,end],'expected_bytes':n,'request_headers':{'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'},'payload_bytes':0,'status':None}
        self.state['requests'].append(record);self.state['network_bytes_reserved_next_response']=n+65536;self.checkpoint()
        path=self.destination/(f"request_{record['index']:02d}_{label}.bin");h=hashlib.sha256();response=None
        try:
            response=self.open_response(record['request_headers']);record['status']=response.status_code;record['response_headers']=dict(response.headers);record['response_headers_sha256']=hashlib.sha256(json.dumps(record['response_headers'],sort_keys=True).encode()).hexdigest()
            # Reserve 64KiB for each response's status/header protocol overhead, conservatively.
            serialized=sum(len(str(k).encode())+len(str(v).encode())+4 for k,v in response.headers.items())+128
            require(serialized<=65536,'HTTP response headers exceed reserved cap')
            record['header_bytes_estimated']=serialized;record['accounted_header_overhead']=65536
            self.state['network_bytes_accounted']+=65536
            require(response.status_code==206,'Range ignored/status not206: stop before reading response body')
            require(response.headers.get('Content-Range')==f'bytes {start}-{end}/{self.cfg["archive_size"]}','Content-Range mismatch')
            require(response.headers.get('Content-Encoding','identity').strip().lower()=='identity','Encoded Range response rejected')
            if response.headers.get('Content-Length') is not None:require(int(response.headers['Content-Length'])==n,'Content-Length mismatch')
            response_validators={key:response.headers.get(key) for key in ['ETag','Last-Modified']}
            require(any(isinstance(value,str) and value.strip() for value in response_validators.values()),'Archive version validator missing/empty: stop before reading response body')
            for key,value in response_validators.items():
                if key in self.validators:require(value==self.validators[key],'Archive validator changed/missing '+key)
                elif isinstance(value,str) and value.strip():self.validators[key]=value
            record['validators']=dict(self.validators)
            with path.open('xb') as f:
                while record['payload_bytes']<n:
                    self.checkpoint();chunk=response.raw.read(min(65536,n-record['payload_bytes']),decode_content=False)
                    require(chunk,'Truncated Range payload')
                    require(len(chunk)<=n-record['payload_bytes'],'Oversized Range body chunk')
                    f.write(chunk);h.update(chunk);record['payload_bytes']+=len(chunk);self.state['network_bytes_accounted']+=len(chunk)
                    require(self.state['network_bytes_accounted']<=NET_CAP,'8MB network cap')
            require(record['payload_bytes']==n,'Range length mismatch')
            record['success']=True;return path.read_bytes()
        except BaseException as exc:
            record['success']=False;record['error']=repr(exc);raise
        finally:
            if response is not None:response.close()
            record['payload_sha256']=h.hexdigest();record['partial_file']=str(path) if path.exists() else None
            self.state['network_bytes_reserved_next_response']=0;self.checkpoint()

def waveform_metrics(derived,local):
    import numpy as np
    a=np.asarray(derived,dtype=np.float64).reshape(-1);b=np.asarray(local,dtype=np.float64).reshape(-1)
    require(len(a)==len(b)>0 and np.isfinite(a).all() and np.isfinite(b).all(),'Waveform length/finite mismatch')
    delta=a-b;error=float(np.mean(delta**2));power=float(np.mean(b**2));aa=a-a.mean();bb=b-b.mean();den=float(np.sqrt((aa@aa)*(bb@bb)))
    return {'samples':len(a),'max_abs_difference':float(np.max(np.abs(delta))),'max_abs_difference_LSB':float(np.max(np.abs(delta))*32768),'rms_difference_LSB':float(np.sqrt(error)*32768),'correlation':float((aa@bb)/den) if den>0 else None,'correlation_status':'defined' if den>0 else 'undefined_zero_variance','snr_db':10*math.log10(power/error) if power>0 and error>0 else None,'snr_status':'finite' if power>0 and error>0 else 'infinite_exact' if power>0 else 'both_zero' if error==0 else 'zero_reference_nonzero_error','zero_alignment':True,'gate_passed':float(np.max(np.abs(delta)))<=2/32768}

def supervise():
    for name,h in read(OUT/'frozen_preparation_manifest.json').items():require(sha(OUT/name)==h,'Prepared file changed '+name)
    for name in ['STARTED.json','outputs','metrics.json','experiment_report.md','run.log','progress.json','child_result.json','central_selected_frozen.json']:
        require(not (OUT/name).exists() and not (OUT/name).is_symlink(),'Existing execution artifact '+name)
    started=time.monotonic()
    with (OUT/'STARTED.json').open('x') as f:json.dump({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':[sys.executable,str(Path(__file__).resolve())],'frozen_manifest_sha256':sha(OUT/'frozen_preparation_manifest.json')},f,indent=2)
    env=os.environ.copy();env['WAVE_GATE_DEADLINE']=str(started+TIME_CAP);env['WAVE_GATE_SUPERVISED']=str(os.getpid())
    reason=None
    with (OUT/'run.log').open('x') as log:
        child=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--child'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        while child.poll() is None:
            if time.monotonic()-started>=TIME_CAP or artifact_size()>=ART_CAP-200_000:
                reason='180s deadline' if time.monotonic()-started>=TIME_CAP else '12MB artifact cap';os.killpg(child.pid,signal.SIGKILL);child.wait();break
            time.sleep(.1)
    d=read(OUT/'child_result.json') if (OUT/'child_result.json').exists() else read(OUT/'progress.json') if (OUT/'progress.json').exists() else {'status':'failed','failures':[]}
    if reason or (child.returncode!=0 and d.get('status') not in ('failed','waveform_gate_failed')):
        d['status']='failed';d.setdefault('failures',[]).append({'supervisor_error':reason or 'child exit '+str(child.returncode)})
    d.update(supervisor_seconds=time.monotonic()-started,child_exit_code=child.returncode,artifact_bytes_before_final=artifact_size())
    d['artifact_manifest']={str(p.relative_to(OUT)):{'bytes':p.stat().st_size,'sha256':sha(p)} for p in OUT.rglob('*') if p.is_file() and not p.is_symlink() and p.name not in ('metrics.json','experiment_report.md')}
    write('metrics.json',d)
    (OUT/'experiment_report.md').write_text('Official/local VoiceBank four-file waveform gate: '+d['status']+'\n\nSelection is from existing valid p226/p287 metadata only. See metrics.json for all network transactions, fixed resampling comparisons, failures, and pre/post locks. No test content, models, training, gain/delay/trim/search or retry. Passing supports origin/conversion consistency for these four files only; it does not verify spoken text/phoneme timing or pretraining holdout. Existing preparation-script presence is not proof of past execution.\n')
    print(json.dumps({'status':d['status'],'seconds':d['supervisor_seconds'],'requests':len(d.get('requests',[])),'network_bytes_accounted':d.get('network_bytes_accounted'),'failures':d.get('failures')},default=scalar))
    return 0 if d['status']=='passed' else 1

def runtime():
    require(os.environ.get('WAVE_GATE_SUPERVISED')==str(os.getppid()),'Supervisor required')
    cfg=read(OUT/'config.yaml');sel=read(OUT/'frozen_selection.json');d={'status':'running','failures':[],'requests':[],'network_bytes_accounted':0,'network_bytes_reserved_next_response':0,'samples':{},'models':0,'training_calls':0,'audio_decodes':0,'resample_calls':0,'test_content_opened':False,'automatic_retries':0}
    out=OUT/'outputs'
    def checkpoint():
        write('progress.json',d);check()
    def verify(label):
        hashes={}
        for path,h in cfg['source_locks'].items():check();hashes[path]=sha(path);require(hashes[path]==h,label+' source hash '+path)
        for name,h in read(OUT/'frozen_preparation_manifest.json').items():require(sha(OUT/name)==h,label+' prepared '+name)
        d[label+'_hashes']=hashes
    try:
        check();verify('pre');d['environment']={'python':sys.version,'executable':sys.executable,'PATH':os.environ.get('PATH'),'cwd':str(ROOT),'proxy':cfg['proxy']}
        d['git_head']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();d['git_status_before']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        metadata=publisher_metadata(Path(cfg['publisher_html']).read_text(),cfg['bitstream_guid'])
        require(metadata['_name']==cfg['archive_name'] and metadata['sizeBytes']==cfg['archive_size'] and metadata['_links']['content']['href']==cfg['content_url'],'Official embedded name/size/URL mismatch')
        d['publisher_metadata']={'name':metadata['_name'],'size':metadata['sizeBytes'],'checksum_metadata_only_not_full_archive_verified':metadata['checkSum'],'source_html':cfg['publisher_html'],'license_path':cfg['license_file'],'license':'CC BY 4.0'}
        out.mkdir();checkpoint()
        reader=RangeReader(cfg,d,out,checkpoint)
        require(reader.get(0,0,'range_probe')==b'P','Unsupported ZIP prefix')
        suffix=reader.get(max(0,cfg['archive_size']-65557),cfg['archive_size']-1,'eocd_suffix');eocd=parse_eocd(suffix,cfg['archive_size']);d['eocd']=eocd
        central=reader.get(eocd['offset'],eocd['offset']+eocd['size']-1,'central_directory')
        rows=parse_central(central,eocd,[r['official_member'] for r in sel['samples']]);d['central_selected']=rows;d['central_directory_sha256']=hashlib.sha256(central).hexdigest();write('central_selected_frozen.json',rows);checkpoint()
        for sample in sel['samples']:
            row=rows[sample['official_member']];offset=row['local_offset'];header=reader.get(offset,offset+29,sample['id']+'_local_fixed');nlen,xlen=local_header(header,row)
            variable=reader.get(offset+30,offset+29+nlen+xlen,sample['id']+'_local_name_extra')
            require(variable[:nlen].hex()==row['name_bytes_hex'] and safe_name(variable[:nlen],row['flags'])==row['name'],'Local member name mismatch');extra_check(variable[nlen:])
            start=offset+30+nlen+xlen;require(start+row['compressed_size']<=eocd['offset'],'Selected payload overlaps central directory')
            compressed=reader.get(start,start+row['compressed_size']-1,sample['id']+'_payload');raw=decompress_member(compressed,row)
            target=out/(sample['id']+'_official48k.wav');require(target.parent==out,'Unsafe output destination')
            with target.open('xb') as f:f.write(raw)
            d['samples'][sample['id']]={'official_sha256':sha(target),'official_bytes':len(raw),'member':row,'local_source_sha256':sample['source_audio_SHA256']};checkpoint()
        require(len(d['requests'])==15,'Expected15 Range requests without retries')
        import numpy as np
        import soundfile as sf
        import torch
        import torchaudio
        d['environment'].update(numpy=np.__version__,soundfile=sf.__version__,torch=torch.__version__,torchaudio=torchaudio.__version__)
        resampler=torchaudio.transforms.Resample(orig_freq=48000,new_freq=16000).to('cpu')
        actual_defaults={key:getattr(resampler,key) for key in ['lowpass_filter_width','rolloff','resampling_method','beta']}
        require(actual_defaults==cfg['resample_defaults'],'Installed Resample defaults differ from frozen recipe')
        d['resample_recipe']={'class':'torchaudio.transforms.Resample','orig_freq':48000,'new_freq':16000,'dtype':'float32','device':'cpu','defaults':actual_defaults,'source_script':cfg['recipe_script'],'procedure_presence_is_not_execution_proof':True,'gain_delay_trim_search':False}
        all_pass=True
        for sample in sel['samples']:
            checkpoint();sid=sample['id'];official=out/(sid+'_official48k.wav');local=Path(sample['clean_path']);oi=sf.info(official);li=sf.info(local)
            require(oi.samplerate==48000 and li.samplerate==16000 and oi.channels==li.channels==1 and oi.subtype==li.subtype=='PCM_16','Required mono48k/16k PCM16 headers '+sid)
            require(li.frames==sample['clean_header']['frames'],'Frozen local frames changed')
            require((oi.frames+2)//3==li.frames,'Official/local length ratio '+sid)
            x,xsr=sf.read(official,dtype='float32');y,ysr=sf.read(local,dtype='float32');d['audio_decodes']+=2
            require(x.shape==(oi.frames,) and y.shape==(li.frames,) and xsr==48000 and ysr==16000,'Decode geometry')
            with torch.inference_mode():derived=resampler(torch.from_numpy(x).view(1,-1)).view(-1).numpy()
            d['resample_calls']+=1;require(derived.dtype==np.float32,'Float32 fixed resample')
            metrics=waveform_metrics(derived,y);all_pass=all_pass and metrics['gate_passed']
            np.save(out/(sid+'_derived16k_float32.npy'),derived,allow_pickle=False)
            quantized=np.clip(np.floor(derived.astype(np.float64)*32768),-32768,32767).astype(np.int16)
            sf.write(out/(sid+'_derived16k.wav'),quantized,16000,subtype='PCM_16')
            local_int=np.rint(y.astype(np.float64)*32768).astype(np.int16)
            metrics['exact_derived_PCM16_matches_local']=bool(np.array_equal(quantized,local_int));metrics['derived_PCM16_serialization']='floor/saturate32768 then SoundFile int16 PCM_16; separate from rawfloat gate';metrics['raw_float_exact_local']=bool(np.array_equal(derived,y));metrics['official_header']={'sample_rate':oi.samplerate,'channels':oi.channels,'frames':oi.frames,'subtype':oi.subtype};metrics['local_header']=sample['clean_header'];metrics['derived_array_sha256']=hashlib.sha256(derived.tobytes()).hexdigest();d['samples'][sid]['comparison']=metrics;checkpoint()
        require(d['audio_decodes']==8 and d['resample_calls']==4,'Four official+four local decode/call counts')
        d['status']='passed' if all_pass else 'waveform_gate_failed';d['all_four_waveforms_passed']=all_pass
    except BaseException as exc:
        d['status']='failed';d['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});traceback.print_exc()
    finally:
        try:verify('post');d['git_status_after']=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)
        except BaseException as exc:d['status']='failed';d['failures'].append({'postcheck_error':repr(exc)})
        write('child_result.json',d)
    return 0 if d['status']=='passed' else 1

if __name__=='__main__':
    if sys.argv[1:]==['--child']:sys.exit(runtime())
    require(not sys.argv[1:],'No parameter overrides allowed');sys.exit(supervise())
