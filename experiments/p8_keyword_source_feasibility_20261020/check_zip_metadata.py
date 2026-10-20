"""At most3 explicit HTTP GETs; no ZIP member reads or extraction."""
import pathlib,json,hashlib,urllib.request,urllib.error,struct,re,time,signal,shutil,collections,ast,os
D=pathlib.Path(__file__).resolve().parent
C=json.loads((D/'config.yaml').read_text());begin=time.monotonic()
if (D/'metrics.json').exists() or (D/'STARTED.json').exists():raise RuntimeError('No retries')
def timeout(*args):raise TimeoutError('Total60s cap')
signal.signal(signal.SIGALRM,timeout);signal.alarm(60)
class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*args,**kw):raise RuntimeError('Redirect forbidden, no follow-up request')
opener=urllib.request.build_opener(NoRedirect())
R={'status':'running','pid':os.getpid(),'requests':[],'response_body_bytes':0,'failures':[],'member_content_reads':0,'WAV_decodes':0,'archive_README_read':False}
def sha(b):return hashlib.sha256(b).hexdigest()
def get(url,range_header=None):
    if len(R['requests'])>=3:raise RuntimeError('HTTP3 cap')
    if shutil.disk_usage(D).free<1073741824:raise RuntimeError('Disk reserve')
    q={'url':url,'method':'GET','Range':range_header,'status':None,'body_bytes':0};R['requests'].append(q);start=time.monotonic()
    headers={'Accept-Encoding':'identity','User-Agent':'bounded-research-metadata-check/1'}
    if range_header:headers['Range']=range_header
    try:
        with opener.open(urllib.request.Request(url,headers=headers),timeout=20) as response:
            q.update(status=response.status,Content_Range=response.headers.get('Content-Range'),Content_Length=response.headers.get('Content-Length'),ETag=response.headers.get('ETag'),Last_Modified=response.headers.get('Last-Modified'))
            if range_header:
                if response.status!=206:raise RuntimeError('Range unsupported: status must206; body not read')
                m=re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)',q['Content_Range'] or '')
                if not m:raise RuntimeError('Unknown Content-Range; body not read')
                lo,hi,total=map(int,m.groups());expected=hi-lo+1
                if range_header.startswith('bytes=-'):
                    length=int(range_header[7:]);assert lo==max(0,total-length) and hi==total-1
                else:
                    a,b=map(int,range_header[6:].split('-'));assert (lo,hi)==(a,b)
                q.update(start=lo,end=hi,total=total)
            else:
                if response.status!=200:raise RuntimeError('Official code response must200')
                expected=int(q['Content_Length']) if q['Content_Length'] else 131072
            if expected>3145728-R['response_body_bytes']:raise RuntimeError('Body budget would exceed3MiB; no read')
            chunks=[];remaining=expected
            while remaining:
                b=response.read(min(65536,remaining))
                if not b:break
                chunks.append(b);remaining-=len(b);q['body_bytes']+=len(b);R['response_body_bytes']+=len(b)
            data=b''.join(chunks)
            if range_header and len(data)!=expected:raise RuntimeError('Truncated range body')
            q['sha256']=sha(data);return data,q
    finally:q['seconds']=time.monotonic()-start
with (D/'STARTED.json').open('x') as f:json.dump({'pid':os.getpid()},f)
try:
    tail,tq=get(C['source_url'],'bytes=-65557');(D/'zip_tail.bin').write_bytes(tail)
    pos=tail.rfind(b'PK\x05\x06')
    if pos<0 or pos+22>len(tail):raise RuntimeError('EOCD absent')
    sig,disk,cd_disk,n_disk,n_total,cd_size,cd_offset,comment=struct.unpack_from('<4s4H2LH',tail,pos)
    if pos+22+comment!=len(tail):raise RuntimeError('EOCD comment/end mismatch')
    if disk or cd_disk or n_disk!=n_total or n_total==65535 or cd_size==4294967295 or cd_offset==4294967295 or b'PK\x06\x07' in tail[max(0,pos-20):pos]:raise RuntimeError('Multi-disk/ZIP64 unsupported')
    total=tq['total'];eocd_absolute=tq['start']+pos
    if cd_offset+cd_size!=eocd_absolute or cd_offset<0 or cd_size>3145728-R['response_body_bytes']:raise RuntimeError('Central directory bounds unknown/over budget')
    R['EOCD']={'entries':n_total,'central_directory_size':cd_size,'central_directory_offset':cd_offset,'archive_bytes':total,'comment_length':comment,'sha256':sha(tail[pos:])}
    if cd_offset>=tq['start']:
        cd=tail[cd_offset-tq['start']:cd_offset-tq['start']+cd_size]
    else:cd,cq=get(C['source_url'],f'bytes={cd_offset}-{cd_offset+cd_size-1}')
    if len(cd)!=cd_size:raise RuntimeError('Central directory size mismatch')
    (D/'central_directory.bin').write_bytes(cd);entries=[];offset=0
    for _ in range(n_total):
        vals=struct.unpack_from('<4s6H3L5H2L',cd,offset)
        if vals[0]!=b'PK\x01\x02':raise RuntimeError('Bad central signature')
        _,made,needed,flags,method,mt,md,crc,compressed,size,nlen,xlen,clen,entry_disk,iattr,eattr,local=vals
        end=offset+46+nlen+xlen+clen
        if end>len(cd) or entry_disk or any(v==4294967295 for v in (compressed,size,local)):raise RuntimeError('Unsupported central record/ZIP64')
        raw=cd[offset+46:offset+46+nlen];name=raw.decode('utf-8' if flags&2048 else 'cp437')
        entries.append({'filename':name,'size':size,'compressed_size':compressed,'CRC32':f'{crc:08x}','local_header_offset':local,'flags':flags,'compression':method,'external_attributes':eattr})
        offset=end
    if offset!=len(cd):raise RuntimeError('Central directory trailing bytes unsupported')
    (D/'central_directory_entries.json').write_text(json.dumps(entries,separators=(',',':'))+'\n')
    # Third request verifies exact official hash rule without importing TensorFlow.
    code,codeq=get(C['official_which_set_source_url']);(D/'official_input_data.py.txt').write_bytes(code);tree=ast.parse(code.decode())
    function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='which_set')
    source=ast.get_source_segment(code.decode(),function);R['official_which_set_function']=source
    if 'MAX_NUM_WAVS_PER_CLASS + 1' not in source or '100.0 / MAX_NUM_WAVS_PER_CLASS' not in source:raise RuntimeError('Official hash arithmetic differs from frozen expectation')
    if not re.search(r'MAX_NUM_WAVS_PER_CLASS\s*=\s*2\s*\*\*\s*27\s*-\s*1',code.decode()):raise RuntimeError('Official hash constant differs')
    def split(group):
        percentage=(int(hashlib.sha1(group.encode()).hexdigest(),16)%(2**27))*(100.0/(2**27-1))
        return 'validation' if percentage<10 else 'testing' if percentage<20 else 'training'
    labels=set(C['word_labels_expected']);wav=[];anomalies=[];names=collections.Counter(e['filename'] for e in entries)
    for e in entries:
        p=pathlib.PurePosixPath(e['filename']);parts=p.parts
        if p.is_absolute() or '..' in parts or '\\' in e['filename']:anomalies.append({'path':e['filename'],'reason':'unsafe_path'})
        if len(parts)==3 and parts[0]=='mini_speech_commands' and parts[1] in labels and parts[2].endswith('.wav'):
            match=re.fullmatch(r'([^/]+)_nohash_(\d+)\.wav',parts[2])
            if not match:anomalies.append({'path':e['filename'],'reason':'unexpected_labelled_wav_name'});continue
            group=match[1];wav.append({'filename':e['filename'],'word':parts[1],'group':group,'split':split(group)})
    groups={g:{r['word'] for r in wav if r['group']==g} for g in {r['group'] for r in wav}}
    counts={word:dict(files=sum(r['word']==word for r in wav),groups=len({r['group'] for r in wav if r['word']==word}),splits={s:dict(files=sum(r['word']==word and r['split']==s for r in wav),groups=len({r['group'] for r in wav if r['word']==word and r['split']==s})) for s in ['training','validation','testing']}) for word in sorted(labels)}
    splitgroups={s:{r['group'] for r in wav if r['split']==s} for s in ['training','validation','testing']}
    R.update(status='completed_metadata_only',labelled_WAV_files=len(wav),per_word=counts,total_nohash_groups=len(groups),cross_word_groups=sum(len(words)>1 for words in groups.values()),group_word_coverage_histogram=dict(collections.Counter(len(words) for words in groups.values())),duplicates=[{'path':n,'count':c} for n,c in names.items() if c>1],path_anomalies=anomalies,other_entries=len(entries)-len(wav),group_split_counts={s:len(g) for s,g in splitgroups.items()},group_split_intersections={s+'__'+t:len(splitgroups[s]&splitgroups[t]) for s,t in [('training','validation'),('training','testing'),('validation','testing')]},group_scope='filename prefix, not verified speaker identity',limitations=['directory word labels not checked against recordings','no member content/README/WAV/header read','CC BY from official documentation supplied by Root; archive README/license not inspected','pretraining exposure unknown','group hashing is local reconstruction of official which_set; no TensorFlow installed/imported'])
    (D/'label_group_split_metadata.json').write_text(json.dumps(wav,separators=(',',':'))+'\n')
except BaseException as e:
    import traceback
    R['status']='failed_no_retry';R['failures'].append({'error':repr(e),'traceback':traceback.format_exc()})
finally:
    signal.alarm(0);R.update(seconds=time.monotonic()-begin,requests_used=len(R['requests']),config_sha256=sha((D/'config.yaml').read_bytes()),code_sha256=sha(pathlib.Path(__file__).read_bytes()),disk_free_bytes=shutil.disk_usage(D).free)
    if R['response_body_bytes']>3145728 or R['seconds']>60:R['status']='failed_budget'
    (D/'metrics.json').write_text(json.dumps(R,indent=2)+'\n')
print(json.dumps({k:v for k,v in R.items() if k not in ['official_which_set_function','failures']}))
raise SystemExit(0 if R['status']=='completed_metadata_only' else 1)
