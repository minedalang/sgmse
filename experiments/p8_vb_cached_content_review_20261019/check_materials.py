"""Narrow independent128links/104cachedsource hash+header check; no PCMdecode."""
from pathlib import Path
import hashlib,json,os,signal,wave
D=Path(__file__).resolve().parent
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check():
    signal.signal(signal.SIGALRM,lambda *a:(_ for _ in ()).throw(TimeoutError('120scheck')));signal.alarm(120)
    cfg=json.loads((D/'config.yaml').read_text());reg=json.loads(Path(cfg['input_registry']).read_text());met=json.loads(Path(cfg['model_metrics']).read_text());public=json.loads((D/'cases.json').read_text());mapping=json.loads((D/'operator_mapping.json').read_text());metrics=json.loads((D/'metrics.json').read_text());protected=sha(D/'metrics.json')
    assert metrics['status']=='completed' and metrics['human_responses']==0 and not (D/'hard_timeout.json').exists();assert sha(cfg['input_registry'])==cfg['input_registry_sha256'] and sha(cfg['model_metrics'])==cfg['model_metrics_sha256']
    hashstr=lambda s:hashlib.sha256(s.encode()).hexdigest();ordered=sorted(reg['cases'],key=lambda r:hashstr(cfg['case_order_salt']+'|'+r['basename']));native={r['basename']:r for r in met['native_cases']};clean={r['id']:r['scaledclean'] for r in reg['groups']};refs={r['id']:r['original_text'] for r in reg['references']};ip=Path(cfg['input_registry']).parent;mp=Path(cfg['model_metrics']).parent;sources={};links=[]
    assert len(public['cases'])==len(mapping['cases'])==32
    for r,pub,op in zip(ordered,public['cases'],mapping['cases']):
        cid='C'+hashstr(cfg['case_order_salt']+'|'+r['basename'])[:16];n=native[r['basename']];g=clean[r['id']];parity=int(hashstr(cfg['AB_parity_salt']+'|'+r['basename']),16)%2;ab={'A':'frontend' if parity==0 else 'off','B':'off' if parity==0 else 'frontend'}
        assert pub['case_id']==cid and pub['reference']==refs[r['id']] and pub['duration_seconds']==r['samples']/16000 and op['AB_mapping']==ab
        source={'clean':(ip/g['wav'],g['wav_sha256']),'noisy':(ip/r['wav'],r['wav_sha256']),'frontend':(mp/n['frontend_pcm']['path'],n['frontend_pcm']['sha256']),'off':(mp/n['off_pcm']['path'],n['off_pcm']['sha256'])}
        for label,role in [('clean','clean'),('noisy','noisy'),('A',ab['A']),('B',ab['B'])]:
            path,h=source[role];link=D/pub['audio'][label];assert link.is_symlink() and link.resolve()==path.resolve() and op['entries'][label]['path']==str(path);links.append(link)
            assert link.name==hashstr('p8-vb-content-audio-v1|'+cid+'|'+label)+'.wav'
            if str(path) not in sources:
                assert sha(path)==h
                with wave.open(str(path),'rb') as w:assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype(),w.getnframes())==(16000,1,2,'NONE',r['samples'])
                sources[str(path)]=dict(sha256=h,bytes=path.stat().st_size)
    assert len(sources)==104 and len(links)==128 and len(set(str(p) for p in links))==128
    text=(D/'index.html').read_text()+(D/'player.js').read_text()+json.dumps(public)
    assert not any(r['basename'] in text or r['speaker'] in text or r['id'] in text for r in reg['cases']);assert 'frontend' not in text and 'OFF' not in text and 'ASR' not in text
    assert (D/'index.html').read_text().count('<audio ')==1 and 'localStorage' in text and 'uploaded_to_server:false' in text
    regular=sum(p.stat().st_size for p in D.rglob('*') if p.is_file() and not p.is_symlink());assert regular<cfg['max_new_regular_bytes'] and sha(D/'metrics.json')==protected
    result=dict(status='passed',entries=128,unique_sources=104,source_header_checks=104,PCM_sample_decodes=0,human_responses=0,regular_bytes_before_record=regular,symlink_own_bytes=sum(p.lstat().st_size for p in links),referenced_unique_bytes=sum(r['bytes'] for r in sources.values()),referenced_entry_total_bytes=sum(p.stat().st_size for p in links),metrics_sha256=protected,actual_browser_playback_tested=False)
    with (D/'independent_validation.json').open('x') as f:json.dump(result,f,indent=2)
    signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':check()
