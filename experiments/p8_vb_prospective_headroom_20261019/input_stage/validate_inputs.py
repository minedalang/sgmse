"""Independent CPU reconstruction after Worker preparation. Does not import driver."""
from pathlib import Path
import hashlib,json,math,struct,wave,zipfile
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pcm(p):
    with wave.open(str(p),'rb') as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())==(1,2,16000,'NONE')
        n=w.getnframes();b=w.readframes(n);assert len(b)==2*n
    return [x[0] for x in struct.iter_unpack('<h',b)]
def power(x):return math.fsum(v*v for v in x)/len(x)
def validate():
    cfg=json.loads((OUT/'config.yaml').read_text());reg=json.loads((OUT/'input_registry.json').read_text());met=json.loads((OUT/'metrics.json').read_text())
    assert met['status']=='completed' and met['model_phase_authorized'] is False
    for p,h in cfg['source_locks'].items():assert sha(p)==h,p
    for name,h in json.loads((OUT/'preparation_manifest.json').read_text()).items():assert sha(OUT/name)==h,name
    assert sha(OUT/'input_registry.json')==met['input_registry_sha256']
    for n,item in met['artifact_manifest'].items():assert (OUT/n).stat().st_size==item['bytes'] and sha(OUT/n)==item['sha256'],n
    assert len(reg['groups'])==8 and len(reg['cases'])==32
    noise={n['id']:[v/32768 for v in pcm(n['path'])] for n in cfg['noises']}
    # Independent publisher-byte comparison and historical full decoded-noise hashes.
    refs={r['id']:r for r in reg['references']}
    source={json.loads(line)['id']:json.loads(line) for line in Path(cfg['text_manifest']).read_text().splitlines()}
    with zipfile.ZipFile(cfg['text_zip']) as z:
        for sample in cfg['samples']:
            sid=sample['id']; t=source[sid]['transcript']; raw=z.read(t['archive_entry']); r=refs[sid]
            assert raw.hex()==r['raw_text_hex'] and hashlib.sha256(raw).hexdigest()==t['raw_text_SHA256']==r['raw_text_sha256']
            assert raw.decode('utf-8-sig')==t['original_text']==r['original_text']
    for n in cfg['noises']:
        vals=noise[n['id']]
        assert hashlib.sha256(struct.pack('<'+'d'*len(vals),*vals)).hexdigest()==n['historical_float64_sha256']
    checked=0
    def check_artifact(item,expected):
        nonlocal checked
        wav=OUT/item['wav'];raw=OUT/item['float32'];assert sha(wav)==item['wav_sha256'] and sha(raw)==item['float32_sha256']
        q=pcm(wav);assert len(q)==len(expected)==item['samples']
        assert q==[math.floor(v*32768) for v in expected]
        assert raw.read_bytes()==struct.pack('<'+'f'*len(expected),*expected)
        assert max(abs(v) for v in expected)<=.90004 and all(-32768<=v<=32767 for v in q)
        assert item['peak']==max(abs(v) for v in expected) and item['pcm_peak']==max(abs(v/32768) for v in q)
        floats=[v[0] for v in struct.iter_unpack('<f',raw.read_bytes())]
        assert item['max_quantization_error']==max(abs(v/32768-x) for v,x in zip(q,expected))
        assert item['max_float32_rounding_error']==max(abs(v-x) for v,x in zip(floats,expected))
        assert item['max_pcm_vs_float32_error']==max(abs(v/32768-x) for v,x in zip(q,floats))
        assert item['saturation_count']==0
        assert max(abs(v/32768-x) for v,x in zip(q,expected))<=1/32768+1e-15
        checked+=1
    order=[]
    for sample,group in zip(cfg['samples'],reg['groups']):
        sid=sample['id'];assert group['id']==sid and group['speaker']==sample['speaker']
        s=[v/32768 for v in pcm(sample['audio'])];L=len(s);p=power(s);recipes=[];maximum=max(abs(v) for v in s)
        for nid in ['N1','N2']:
            n=noise[nid][80000:80000+L];assert len(n)==L;np=power(n)
            for snr in [-18,0]:
                g=math.sqrt(p/np)*10**(-snr/20);x=[v+g*w for v,w in zip(s,n)];maximum=max(maximum,max(abs(g*v) for v in n),max(abs(v) for v in x));recipes.append((nid,snr,g,x,n))
        a=min(1.,.9/maximum);assert a==group['scale'] and maximum==group['group_max_before_scale'];check_artifact(group['scaledclean'],[a*v for v in s])
        for nid,snr,g,x,n in recipes:
            row=reg['cases'][len(order)];order.append((sid,nid,snr));assert (row['id'],row['noise'],row['snr_db'])==order[-1] and row['gain']==g and row['scale']==a
            name=sid+'_'+nid+'_snr_'+('m18' if snr==-18 else 'p0')+'.wav';assert row['basename']==name and Path(row['wav']).name==name
            assert row['seed']==(2026+int.from_bytes(hashlib.blake2b(name.encode(),digest_size=4).digest(),'little'))%2**32
            assert abs(10*math.log10(power([a*v for v in s])/power([a*g*v for v in n]))-snr)<1e-10
            expected=[a*v for v in x]; clean=[a*v for v in s]; nc=[a*g*v for v in n]
            assert row['component_snr_db']==10*math.log10(power(clean)/power(nc))
            f32=[v[0] for v in struct.iter_unpack('<f',struct.pack('<'+'f'*L,*expected))]
            quant=[v/32768 for v in pcm(OUT/row['wav'])]
            assert row['float32_residual_noise_snr_db']==10*math.log10(power(clean)/power([v-c for v,c in zip(f32,clean)]))
            assert row['pcm_residual_noise_snr_db']==10*math.log10(power(clean)/power([v-c for v,c in zip(quant,clean)]))
            assert row['max_pcm_mixture_residual']==max(abs(v-c-nn) for v,c,nn in zip(quant,clean,nc))
            check_artifact(row,expected)
    assert checked==40 and met['audio_source_decodes']==10 and met['network_attempts']==[] and met['model_loads']==0
    return dict(status='passed',input_registry_sha256=sha(OUT/'input_registry.json'),reconstructed_artifacts=checked,cases=32,scaledclean_controls=8,model_loads=0,network_bytes=0)
if __name__=='__main__':
    result=validate();path=OUT/'input_independent_validation.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
