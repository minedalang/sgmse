"""Independent CPU reconstruction after Worker preparation. Does not import driver."""
from pathlib import Path
import hashlib,json,math,struct,wave
OUT=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pcm(p):
    with wave.open(str(p),'rb') as w:
        assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getcomptype())==(1,2,16000,'NONE')
        n=w.getnframes();b=w.readframes(n);assert len(b)==2*n
    return [x[0] for x in struct.iter_unpack('<h',b)]
def power(x):return math.fsum(v*v for v in x)/len(x)
def validate():
    cfg=json.loads((OUT/'config.yaml').read_text());reg=json.loads((OUT/'input_registry.json').read_text());met=json.loads((OUT/'input_metrics.json').read_text())
    assert met['status']=='completed' and met['model_phase_authorized'] is False
    for p,h in cfg['source_locks'].items():assert sha(p)==h,p
    for name,h in json.loads((OUT/'preparation_manifest.json').read_text()).items():assert sha(OUT/name)==h,name
    assert sha(OUT/'input_registry.json')==met['input_registry_sha256']
    for n,item in met['artifact_manifest'].items():assert (OUT/n).stat().st_size==item['bytes'] and sha(OUT/n)==item['sha256'],n
    assert len(reg['groups'])==4 and len(reg['cases'])==16
    noise={n['id']:[v/32768 for v in pcm(n['path'])] for n in cfg['noises']}
    checked=0
    def check_artifact(item,expected):
        nonlocal checked
        wav=OUT/item['wav'];raw=OUT/item['float32'];assert sha(wav)==item['wav_sha256'] and sha(raw)==item['float32_sha256']
        q=pcm(wav);assert len(q)==len(expected)==item['samples']
        assert q==[math.floor(v*32768) for v in expected]
        assert raw.read_bytes()==struct.pack('<'+'f'*len(expected),*expected)
        assert max(abs(v) for v in expected)<=.90004 and all(-32768<=v<=32767 for v in q)
        assert max(abs(v/32768-x) for v,x in zip(q,expected))<=1/32768+1e-15
        checked+=1
    order=[]
    for sample,group in zip(cfg['samples'],reg['groups']):
        sid=sample['id'];assert group['id']==sid
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
            check_artifact(row,[a*v for v in x])
    assert checked==20 and met['audio_source_decodes']==6 and met['network_attempts']==[] and met['model_loads']==0
    return dict(status='passed',input_registry_sha256=sha(OUT/'input_registry.json'),reconstructed_artifacts=checked,cases=16,scaledclean_controls=4,model_loads=0,network_bytes=0)
if __name__=='__main__':
    result=validate();path=OUT/'input_independent_validation.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
