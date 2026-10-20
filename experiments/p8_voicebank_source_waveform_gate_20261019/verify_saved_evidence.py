"""Offline independent evidence check; no gate import, network, models or resampling."""
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import hashlib,json,math,signal,socket,struct,time,zlib
from pathlib import Path,PurePosixPath
signal.alarm(60)
def deny(*args,**kwargs):raise RuntimeError('Network forbidden in independent verification')
socket.socket=deny;socket.create_connection=deny
import numpy as np
import soundfile as sf
P=Path(__file__).resolve().parent
start=time.monotonic()
def digest(data):return hashlib.sha256(data).hexdigest()
def sha(p):return digest(Path(p).read_bytes())
def load(name):return json.loads((P/name).read_text())
def size():return sum(x.stat().st_size for x in P.rglob('*') if x.is_file())
def require(ok,msg):
    if not ok:raise AssertionError(msg)
def safe(raw,flags):
    s=raw.decode('utf-8' if flags&2048 else 'cp437');q=PurePosixPath(s)
    require(s and not q.is_absolute() and '..' not in q.parts and not any(c in s for c in ('\x00','\\',':')),'Unsafe ZIP path')
    return s
r={'status':'running','network_requests':0,'models':0,'training_calls':0,'resample_calls':0,'checks':{},'samples':{},'failures':[],'verification_script_sha256':sha(__file__)}
try:
    cfg=load('config.yaml');sel=load('frozen_selection.json');m=load('metrics.json');r['metrics_sha256']=sha(P/'metrics.json')
    require(sha(P/'frozen_preparation_manifest.json')=='2aedc78cce659639f5164e3687356563cfd6e4bb78d395c2e77301dde064cab6','Preparation manifest lock')
    for name,h in load('frozen_preparation_manifest.json').items():require(sha(P/name)==h,'Frozen '+name)
    for name,h in cfg['source_locks'].items():require(sha(name)==h and m['pre_hashes'][name]==h and m['post_hashes'][name]==h,'Source '+name)
    for name,row in m['artifact_manifest'].items():require((P/name).stat().st_size==row['bytes'] and sha(P/name)==row['sha256'],'Run artifact '+name)
    r['checks']['frozen_preparation_files']=len(load('frozen_preparation_manifest.json'));r['checks']['source_locks_pre_post_now']=len(cfg['source_locks']);r['checks']['original_run_artifacts']=len(m['artifact_manifest'])
    payloads={};validators={};account=0;body=0
    for i,q in enumerate(m['requests']):
        a,b=q['requested_range'];n=b-a+1;h=q['response_headers'];lower={k.lower():v for k,v in h.items()}
        require(q['index']==i and q['status']==206 and q['success'],'HTTP status')
        require(q['request_headers']=={'Range':f'bytes={a}-{b}','Accept-Encoding':'identity'},'HTTP request headers')
        require(lower['content-range']==f'bytes {a}-{b}/{cfg["archive_size"]}' and lower.get('content-encoding','identity').strip().lower()=='identity','HTTP response range/encoding')
        require('content-length' not in lower or int(lower['content-length'])==n,'Content length')
        require(digest(json.dumps(h,sort_keys=True).encode())==q['response_headers_sha256'],'Header hash')
        require(any(lower.get(k,'').strip() for k in ('etag','last-modified')),'Version missing')
        for k,v in validators.items():require(lower.get(k)==v,'Version instability')
        for k in ('etag','last-modified'):
            if lower.get(k,'').strip():validators[k]=lower[k]
        raw=Path(q['partial_file']).read_bytes()
        require(len(raw)==n==q['expected_bytes']==q['payload_bytes'] and digest(raw)==q['payload_sha256'],'Body length/hash')
        require(q['accounted_header_overhead']==65536,'Header accounting')
        estimate=sum(len(str(k).encode())+len(str(v).encode())+4 for k,v in h.items())+128
        require(estimate==q['header_bytes_estimated'] and estimate<=65536,'Header estimate')
        body+=n;account+=n+65536;payloads[q['label']]=raw
    require(len(m['requests'])==15<=16 and account==m['network_bytes_accounted']<=8000000,'Request/network cap')
    r['checks']['HTTP_transactions']=len(m['requests']);r['network_body_bytes']=body;r['network_bytes_accounted']=account;r['stable_validators']=validators
    tail=payloads['eocd_suffix'];at=tail.rfind(b'PK\x05\x06');f=struct.unpack_from('<4s4H2LH',tail,at)
    require(f[1]==f[2]==0 and f[3]==f[4] and at+22+f[7]==len(tail),'EOCD')
    require(f[6]+f[5]==cfg['archive_size']-len(tail)+at,'Directory position')
    cd=payloads['central_directory'];require(len(cd)==f[5] and digest(cd)==m['central_directory_sha256'],'Directory body')
    selected={x['official_member'] for x in sel['samples']};rows={};pos=0;count=0
    while pos<len(cd):
        t=struct.unpack_from('<4s6H3L5H2L',cd,pos);require(t[0]==b'PK\x01\x02','Central signature')
        nl,xl,cl=t[10:13];namebytes=cd[pos+46:pos+46+nl];name=safe(namebytes,t[3]);count+=1
        if name in selected:
            require(name not in rows,'Duplicate member')
            rows[name]={'name':name,'name_bytes_hex':namebytes.hex(),'version_needed':t[2],'flags':t[3],'method':t[4],'crc32':t[7],'compressed_size':t[8],'uncompressed_size':t[9],'local_offset':t[16]}
        pos+=46+nl+xl+cl
    require(pos==len(cd) and count==f[4] and set(rows)==selected,'Central count/selection')
    require(rows==load('central_selected_frozen.json')==m['central_selected'],'Frozen selected central')
    for s in sel['samples']:
        sid=s['id'];row=rows[s['official_member']];h=struct.unpack('<4s5H3L2H',payloads[sid+'_local_fixed'])
        require(h[0]==b'PK\x03\x04' and h[1:4]==(row['version_needed'],row['flags'],row['method']),'Local fixed')
        variable=payloads[sid+'_local_name_extra'];require(len(variable)==h[9]+h[10] and variable[:h[9]].hex()==row['name_bytes_hex'],'Local variable')
        if h[2]&8:require(all(a in (0,b) for a,b in zip(h[6:9],(row['crc32'],row['compressed_size'],row['uncompressed_size']))),'Descriptor local values')
        else:require(h[6:9]==(row['crc32'],row['compressed_size'],row['uncompressed_size']),'Local CRC/sizes')
        data=payloads[sid+'_payload'];require(len(data)==row['compressed_size'],'Compressed size')
        if row['method']==8:
            dec=zlib.decompressobj(-15);raw=dec.decompress(data,row['uncompressed_size']+1);require(dec.eof and not dec.unused_data and not dec.unconsumed_tail,'Bounded deflate')
        else:require(row['method']==0,'Method');raw=data
        require(len(raw)==row['uncompressed_size'] and zlib.crc32(raw)&0xffffffff==row['crc32'],'CRC/size')
        official=P/'outputs'/(sid+'_official48k.wav');require(raw==official.read_bytes() and digest(raw)==m['samples'][sid]['official_sha256'],'Official WAV bytes')
        info=sf.info(official);require(info.samplerate==48000 and info.channels==1 and info.subtype=='PCM_16','Official header')
        a=np.load(P/'outputs'/(sid+'_derived16k_float32.npy'),allow_pickle=False);li=sf.info(s['clean_path']);ints,sr=sf.read(s['clean_path'],dtype='int16');b=ints.astype(np.float64)/32768
        require(a.dtype==np.float32 and a.shape==b.shape and len(a)==s['clean_header']['frames'] and (info.frames+2)//3==len(a) and sr==16000 and li.subtype=='PCM_16','Geometry/dtype')
        require(digest(a.tobytes())==m['samples'][sid]['comparison']['derived_array_sha256'],'Derived array hash')
        d=a.astype(np.float64)-b;maxerr=float(np.max(np.abs(d))*32768);rms=float(np.sqrt(np.mean(d*d))*32768);corr=float(np.corrcoef(a.astype(np.float64),b)[0,1]);snr=float(10*np.log10(np.mean(b*b)/np.mean(d*d)))
        prior=m['samples'][sid]['comparison']
        for k,v in [('max_abs_difference_LSB',maxerr),('rms_difference_LSB',rms),('correlation',corr),('snr_db',snr)]:require(math.isclose(v,prior[k],rel_tol=1e-11,abs_tol=1e-11),'Comparison '+k)
        quant=np.clip(np.floor(a.astype(np.float64)*32768),-32768,32767).astype(np.int16);saved,saved_sr=sf.read(P/'outputs'/(sid+'_derived16k.wav'),dtype='int16');require(saved_sr==16000 and np.array_equal(saved,quant),'Serialized PCM samples')
        require(bool(np.array_equal(quant,ints))==prior['exact_derived_PCM16_matches_local'] and (maxerr<=2)==prior['gate_passed'],'Gate/PCM outcome')
        r['samples'][sid]={'samples':len(a),'max_abs_difference_LSB':maxerr,'rms_difference_LSB':rms,'correlation':corr,'snr_db':snr,'gate_passed':maxerr<=2,'zero_alignment':True,'CRC_and_payload_verified':True,'serialized_PCM_verified':True}
    passed=all(x['gate_passed'] for x in r['samples'].values());require(m['all_four_waveforms_passed']==passed and m['status']==('passed' if passed else 'waveform_gate_failed'),'Overall gate outcome')
    require(m['models']==m['training_calls']==m['automatic_retries']==0 and not m['test_content_opened'],'Scope')
    require(m['supervisor_seconds']<180 and size()<12000000,'Time/artifact cap')
    r['checks']['gate_reached']=True;r['checks']['gate_passed']=passed;r['checks']['transport_complete']=True;r['checks']['serialized_waveforms']=4
    r['status']='passed'
except BaseException as e:
    r['status']='failed';r['failures'].append(repr(e))
r['seconds']=time.monotonic()-start;r['artifact_bytes_before_validation_json']=size()
(P/'independent_validation.json').write_text(json.dumps(r,indent=2,allow_nan=False)+'\n')
print(json.dumps(r,indent=2));raise SystemExit(0 if r['status']=='passed' else 1)
