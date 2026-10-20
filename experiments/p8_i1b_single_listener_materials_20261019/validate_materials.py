"""Narrow independent94-file PCM/geometry/public-field check; no listening or model."""
from pathlib import Path
import hashlib,io,json,signal,wave
D=Path(__file__).resolve().parent
sha=lambda b:hashlib.sha256(b).hexdigest()
def validate():
    signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('120s validation deadline')));signal.alarm(120)
    cfg=json.loads((D/'config.yaml').read_text());allocation_bytes=Path(cfg['allocation']).read_bytes();assert sha(allocation_bytes)==cfg['allocation_sha256'];allocation=json.loads(allocation_bytes)
    public=json.loads((D/'trials.json').read_text());operator=json.loads((D/'operator_sources.json').read_text());metrics=json.loads((D/'metrics.json').read_text());assert metrics['status']=='completed' and metrics['human_responses']==0 and not (D/'hard_timeout.json').exists()
    assert len(public['trials'])==len(operator['trials'])==94 and len(set(t['trial_id'] for t in public['trials']))==94
    source={};expected=[]
    for stage,arm in enumerate(['noisy','clean'],1):
        rows=[]
        for o in allocation['observations']:
            order=sha(('p8_i1b_single_listener_v1|'+arm+'|'+o['operator_observation_id']).encode());rows.append((order,o))
        expected.extend((stage,arm,order,o) for order,o in sorted(rows,key=lambda r:r[0]))
    for (stage,arm,order,o),pub,op in zip(expected,public['trials'],operator['trials']):
        tid='T'+order[:16];assert pub==dict(trial_id=tid,stage=stage,audio='audio/'+tid+'.wav')
        src=o['source_files'][arm];path=Path(src['path']);assert op['source_path']==str(path) and op['context_start']==o['context_start'] and op['context_stop']==o['context_stop'] and op['observation_id']==o['operator_observation_id']
        if str(path) not in source:
            b=path.read_bytes();assert sha(b)==src['sha256'] and len(b)==src['bytes']
            with wave.open(io.BytesIO(b),'rb') as w:assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,src['header']['frames']);source[str(path)]=w.readframes(w.getnframes())
        raw=(D/pub['audio']).read_bytes();assert sha(raw)==op['crop_sha256']
        with wave.open(io.BytesIO(raw),'rb') as w:assert (w.getnchannels(),w.getsampwidth(),w.getframerate(),w.getnframes())==(1,2,16000,7680);pcm=w.readframes(7680)
        assert pcm==source[str(path)][o['context_start']*2:o['context_stop']*2] and sha(pcm)==op['crop_pcm_sha256'] and op['target_relative_start']==3200 and op['target_relative_stop']==4480
    assert len(source)==24 and len(list((D/'audio').glob('*.wav')))==94
    html=(D/'index.html').read_text();js=(D/'player.js').read_text();assert html.count('<audio ')==1 and 'localStorage' in js and 'fetch(' in js and '.post(' not in js and "method:'POST'" not in js
    public_text=json.dumps(public)+html+js
    assert not any(o['utterance_id'] in public_text or o['speaker'] in public_text for o in allocation['observations'])
    assert sum(p.stat().st_size for p in D.rglob('*') if p.is_file())<cfg['max_directory_bytes']
    result=dict(status='passed',verified_crops=94,source_containers=24,pcm_bytes_exact=True,normalization='none',human_responses=0,real_browser_playback_tested=False,metrics_sha256=sha((D/'metrics.json').read_bytes()))
    with (D/'independent_validation.json').open('x') as f:json.dump(result,f,indent=2)
    signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':validate()
