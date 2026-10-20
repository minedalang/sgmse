"""CPU-only byte copies and read-only local validation; no model imports/calls.
Run: env/bin/python -B .../attempt02/listening/build_validate.py
Only writes within this script's listening directory. HTTP requests use loopback only.
"""
import hashlib, io, json, os, runpy, shutil, sqlite3, subprocess, urllib.request, wave
from pathlib import Path
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
ATTEMPT = HERE.parent
ROOT = ATTEMPT.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(name, value): (HERE/name).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
def snapshot():
    return {str(p.relative_to(ROOT)): {'sha256':sha(p),'bytes':p.stat().st_size}
            for p in sorted(ROOT.rglob('*')) if p.is_file() and HERE not in p.parents}
def identity(pid):
    return {'pid':pid,'stat':Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[19],
            'cmdline':Path(f'/proc/{pid}/cmdline').read_bytes().decode().replace('\0',' ')}
before=snapshot()
server=identity(16022); bridge=identity(39452)
assert 'feedback_server.py' in server['cmdline'] and '--port 8766' in server['cmdline']
assert 'bridge.py' in bridge['cmdline']
con=sqlite3.connect('file:/data/codex-state/state_5.sqlite?mode=ro',uri=True)
rows=con.execute('select id,agent_path,model_provider from threads where agent_path=?',('/root/engineer_native_listening',)).fetchall()
assert rows and all(r[2]=='codex_tools_compat' for r in rows)
registry=json.loads((ROOT/'input_registry.json').read_text())
metrics=json.loads((ATTEMPT/'metrics.json').read_text())
assert metrics['status']=='completed'
assert (metrics['frontend_calls'], metrics['backend_calls'],metrics['transcribe_api_calls'])==(16,16,52)
sources={'selection':'Lead-fixed post-hoc exploratory diagnostics selected using existing ASR differences; not independent samples or blind listening; no population win rate',
         'operations':'8 byte-for-byte copies only; complete original amplitude; no normalization, trim, resample or audio regeneration',
         'records':[], 'source_locks':{}, 'registry_sha256':sha(ROOT/'input_registry.json'),
         'metrics_sha256':sha(ATTEMPT/'metrics.json')}
def lock(path, expected=None):
    actual=sha(path)
    if expected is not None: assert actual==expected,(str(path),actual,expected)
    sources['source_locks'][str(path)]={'sha256':actual,'bytes':Path(path).stat().st_size}
    return actual
(HERE/'audio').mkdir(exist_ok=True)
for group, basename, length in [('A','p287_054_N1_snr_p0.wav',51089),('B','p226_028_N2_snr_m18.wav',67653)]:
    case=next(c for c in registry['cases'] if c['basename']==basename)
    g=next(g for g in registry['groups'] if g['id']==case['id'])
    native=next(c for c in metrics['native_cases'] if c['basename']==basename)
    assert case['scale']==g['scale']
    source_wave=next(Path(p) for p in registry['sources'] if p.endswith(case['id']+'_derived16k.wav'))
    lock(source_wave,registry['sources'][str(source_wave)])
    source_samples,rate=sf.read(source_wave,dtype='float64'); assert rate==16000
    clean_expected=np.floor(source_samples*g['scale']*32768).astype(np.int16)
    for arm in ['clean','noisy','frontend','off']:
        evidence={}
        if arm in ['clean','noisy']:
            r=g['scaledclean'] if arm=='clean' else case
            src=ROOT/r['wav']; expected=r['wav_sha256']; raw=ROOT/r['float32']
            lock(raw,r['float32_sha256'])
            floats=np.fromfile(raw,dtype='<f4')
            pcm,_=sf.read(src,dtype='int16')
            if arm=='clean': assert np.array_equal(pcm,clean_expected)
            assert len(floats)==length
            assert np.max(np.abs(pcm.astype(np.float64)/32768-floats))<=1/32768+1e-7
            evidence={'raw_float32':str(raw),'raw_format':r['float32_format'],
                      'max_pcm_vs_float32_error':float(np.max(np.abs(pcm.astype(np.float64)/32768-floats))),
                      'registry_quantizer':r['quantizer'], 'shared_group_scale':g['scale'],
                      'source_wave':str(source_wave), 'clean_scaled_pcm_exact':True if arm=='clean' else None}
        else:
            r=native['frontend_pcm' if arm=='frontend' else 'off_pcm']
            rawr=native['frontend_raw' if arm=='frontend' else 'backend_raw']
            src=ATTEMPT/r['path']; expected=r['sha256']; raw=ATTEMPT/rawr['path']
            lock(raw)
            floats=np.load(raw,allow_pickle=False)
            assert hashlib.sha256(floats.tobytes()).hexdigest()==rawr['sha256']
            assert floats.dtype==np.float32 and floats.size==length
            buf=io.BytesIO();sf.write(buf,floats.reshape(-1),16000,format='WAV',subtype='PCM_16')
            assert buf.getvalue()==src.read_bytes()
            evidence={'raw_npy':str(raw),'raw_shape':list(floats.shape),'raw_array_sha256':rawr['sha256'],
                      'native_serialization':'raw float32 -> in-memory libsndfile WAV PCM_16; exact bytes match existing WAV',
                      'soundfile_version':sf.__version__,'libsndfile_version':sf.__libsndfile_version__,
                      'native_no_loudness_normalization':True}
        original_hash=lock(src,expected)
        dest=HERE/'audio'/f'{group}_{arm}.wav'
        shutil.copyfile(src,dest)
        assert dest.read_bytes()==src.read_bytes()
        with wave.open(str(dest),'rb') as w:
            header={'rate':w.getframerate(),'channels':w.getnchannels(),'width':w.getsampwidth(),'frames':w.getnframes(),'compression':w.getcomptype()}
        assert header=={'rate':16000,'channels':1,'width':2,'frames':length,'compression':'NONE'}
        sources['records'].append({'group':group,'case':basename,'arm':arm,'source':str(src),'copy':str(dest.relative_to(HERE)),
                                   'sha256':original_hash,'bytes':dest.stat().st_size,'header':header,'evidence':evidence})
dump('sources.json',sources)
runpy.run_path(str(HERE/'render_page.py'), run_name='__main__')
html=(HERE/'index.html').read_text()
assert html.count('<audio ')==1 and html.count('data-file=')==8
assert '<form' not in html and 'localStorage' not in html
node=subprocess.run(['node',str(HERE/'test_player.cjs')],capture_output=True,text=True,check=True)
base='http://localhost:8766/p8_voicebank_native_content_pilot_20261019/attempt02/listening/'
http=[]
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
for relative in ['index.html','player.js']+[r['copy'] for r in sources['records']]:
    disk=(HERE/relative).read_bytes()
    with opener.open(base+relative,timeout=10) as r:
        body=r.read(); assert r.status==200 and body==disk
        content_type=r.headers.get('Content-Type')
    with opener.open(urllib.request.Request(base+relative,method='HEAD'),timeout=10) as r:
        assert r.status==200 and int(r.headers['Content-Length'])==len(disk)
    http.append({'path':relative,'get_bytes_equal':True,'head_status':200,'content_length':len(disk),'content_type':content_type,'sha256':hashlib.sha256(disk).hexdigest()})
assert snapshot()==before
for p,r in sources['source_locks'].items(): assert sha(p)==r['sha256']
assert identity(16022)==server and identity(39452)==bridge
dump('original_preservation.json',{'pre_post_current_equal':True,'files':before})
result={'status':'passed','url':base+'index.html','provider_rows':rows,'server_unchanged':server,'bridge_unchanged':bridge,
        'eight_full_length_byte_copies':True,'source_pre_post_hashes_equal':True,'original_parent_files_unchanged':True,
        'http_actual_get_and_head':http,'node_mock':json.loads(node.stdout),'audio_elements':1,
        'real_browser_playback_tested':False,'new_model_calls':0,'new_asr_calls':0,'external_network_requests':0,
        'human_feedback_collected':True,'human_feedback_source':'user_chat',
        'feedback_sha256':sha(HERE/'user_feedback_20261019.json'),
        'snapshot_note':'Budget totals include listening and existing experiment artifacts, uniformly excluding symlink duplicate counts to match the frozen runtime budget and original report. parent_logical_bytes_including_symlinks separately includes symlink targets. Original report historical byte counts are unchanged.'}
dump('validation.json',result)
for _ in range(5):
    listening_total=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file() and not p.is_symlink())
    parent_total=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
    parent_logical=sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
    result.update(listening_total_bytes=listening_total,parent_total_bytes=parent_total,parent_logical_bytes_including_symlinks=parent_logical)
    dump('validation.json',result)
assert result['listening_total_bytes']==sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file() and not p.is_symlink())
assert result['parent_total_bytes']==sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file() and not p.is_symlink())
assert result['parent_logical_bytes_including_symlinks']==sum(p.stat().st_size for p in ROOT.rglob('*') if p.is_file())
assert result['listening_total_bytes']<=2_000_000 and result['parent_total_bytes']<=40_000_000
print(json.dumps({'status':'passed','listening_bytes':listening_total,'parent_bytes':parent_total,'node_mock':result['node_mock']}))
