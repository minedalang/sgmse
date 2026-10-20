"""Only temporary synthetic metadata/header inputs. Never opens actual corpus WAVs."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json,os,struct,subprocess,sys,tempfile,time
OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('i0driver',OUT/'register_metadata.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
spec=importlib.util.spec_from_file_location('i0ind',OUT/'validate_metadata.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
cfg=json.loads((OUT/'config.yaml').read_text());checks=[]
def ok(value,label):assert value,label;checks.append(label)
def reject(fn,label):
    try:fn()
    except (RuntimeError,AssertionError,ValueError,KeyError,FileNotFoundError):checks.append(label);return
    raise AssertionError('accepted '+label)
def row(s,n,frames=48000,part='train'):
    sid=f'{s}_{n:03}';h=dict(sample_rate=16000,channels=1,frames=frames,subtype='PCM_16')
    return dict(id=sid,speaker=s,local_partition=part,clean_path='/synthetic/train/clean/'+sid+'.wav',noisy_path='/synthetic/train/noisy/'+sid+'.wav',clean_header=h,noisy_header=h.copy(),transcript=dict(original_text='synthetic words'))
with tempfile.TemporaryDirectory(prefix='synthetic_',dir=OUT) as tmp:
    root=Path(tmp);manifest=root/'manifest.jsonl'
    rows=[row('p'+str(300+i),n) for i in range(6) for n in [100,27,26]]+[row('p226',26),row('p299',26,part='valid'),row('p298',26,part='test'),row('p297',25),row('p296',26,47999),row('p295',26,80001)]
    def save(r):manifest.write_text(''.join(json.dumps(x)+'\n' for x in r))
    save(rows);a=d.derive(manifest,cfg);b=v.expected_selection(manifest,cfg)
    ok([r['source_row']['id'] for r in a]==[t[2] for t in b],'independent selection agreement')
    ok(len(a)==12 and len({r['source_row']['speaker'] for r in a})==6,'12pairs6speaker')
    ok([r['source_row']['id'].rsplit('_',1)[1] for r in a]==['026','027']*6,'numeric suffix sorting not lexical')
    ok(all(r['source_row']['local_partition']=='train' for r in a),'train only')
    save([row('p'+str(300+i),n,80000) for i in range(6) for n in [26,27]])
    ok(len(d.derive(manifest,cfg))==12,'5second inclusive boundary')
    save(rows+[rows[0]]);reject(lambda:d.derive(manifest,cfg),'duplicate ID')
    bad=copy.deepcopy(rows);bad[0]['clean_header']['sample_rate']=0;save(bad);reject(lambda:d.derive(manifest,cfg),'bad schema zero sample rate')
    bad=copy.deepcopy(rows);bad[1]['noisy_header']['frames']+=1;save(bad);reject(lambda:d.derive(manifest,cfg),'pair metadata header mismatch')
    save([row('p300',26)]);reject(lambda:d.derive(manifest,cfg),'insufficient eligible speakers no relaxation')
    reject(lambda:d.verify_lock(root/'missing','0'*64),'missing source')
    save(rows);reject(lambda:d.verify_lock(manifest,'0'*64),'bad source SHA')
    # Mock a parent-directory symlink resolution: no actual dataset filesystem read.
    from unittest.mock import patch
    with patch.object(Path,'is_symlink',return_value=False),patch.object(Path,'resolve',return_value=root/'escaped.wav'):
        reject(lambda:d.approved_audio_path('/data/datasets/vb/voicebank_demand/16k/train/clean/p9999_026.wav','p9999_026','clean'),'resolved parent symlink escape')
    # Binary-line hashes preserve CRLF exactly rather than normalizing newlines.
    save(rows);original=manifest.read_bytes();manifest.write_bytes(original.replace(b'\n',b'\r\n'))
    a=d.derive(manifest,cfg);b=v.expected_selection(manifest,cfg)
    ok([r['source_line_sha256'] for r in a]==[t[4] for t in b],'raw CRLF source-line hash agreement')
    save(rows)
    # Synthetic PCM container only. Driver inspects getparams, never readframes.
    wav=root/'header.wav';payload=b'\0'*(48000*2)
    wav.write_bytes(struct.pack('<4sI4s4sIHHIIHH4sI',b'RIFF',36+len(payload),b'WAVE',b'fmt ',16,1,1,16000,32000,2,16,b'data',len(payload))+payload)
    ok(d.header(wav)['frames']==48000,'synthetic header accepted without PCM read')
    wav.write_bytes(wav.read_bytes()[:44]);reject(lambda:d.header(wav),'truncated declared PCM rejected')
    scanroot=root/'metadata';scanroot.mkdir();(scanroot/'testset_txt').mkdir();(scanroot/'testset_txt/secret.txt').write_text('must not read')
    (scanroot/'validation').mkdir();(scanroot/'validation/v.txt').write_text('must not read')
    outside=root/'outside.txt';outside.write_text('outside');(scanroot/'escape.txt').symlink_to(outside)
    for i in range(4):(scanroot/f'{i}.txt').write_text('clue'+str(i))
    sc=copy.deepcopy(cfg);sc['noise_metadata_roots']=[str(scanroot)];sc['noise_metadata_scan'].update(max_candidate_files=2,max_total_read_bytes=6)
    scanned=d.scan(sc,set());ok(scanned['candidate_files_seen']<=2 and scanned['total_read_bytes']<=6,'scan count byte caps')
    ok(not scanned['noise_identity_resolved'],'unknown noise never ready')
    ok(all('testset' not in r['path'] and 'validation' not in r['path'] and 'escape' not in r['path'] for r in scanned['files']),'excluded subtree and symlink escape not read')
    reject(lambda:d.check_budget(time.monotonic()-100,cfg,root),'timeout guard')
    small=copy.deepcopy(cfg);small['budgets']['artifact_bytes']=65537
    reject(lambda:d.check_budget(time.monotonic(),small,root),'artifact reserve guard')
    old=d.OUT;d.OUT=root;(root/'config.yaml').write_text(json.dumps(cfg));(root/'preparation_manifest.json').write_text('{"files":{}}');(root/'STARTED.json').write_text('{}')
    reject(d.run,'existing marker duplicate rejects before real source access');d.OUT=old
    pre="import importlib.util;from pathlib import Path;s=importlib.util.spec_from_file_location('x',"+repr(str(OUT/'register_metadata.py'))+");x=importlib.util.module_from_spec(s);s.loader.exec_module(x);x.guard(Path("+repr(str(root))+"),set(),[]);"
    for action,label in [('import torch','model import guard'),('import numpy','array import guard'),('import soundfile','audio import guard'),('import socket;socket.socket()','network guard'),("open('/data/forbidden_i0_write','w')",'source write guard'),("open('/data/datasets/forbidden.wav','rb')",'unapproved waveform read guard')]:
        r=subprocess.run([sys.executable,'-B','-c',pre+action],capture_output=True,text=True)
        ok(r.returncode!=0 and ('forbidden' in r.stderr or 'unapproved' in r.stderr),'fresh child '+label)
for p in OUT.glob('*.py'):ast.parse(p.read_text())
ok(True,'AST all scripts')
ok(not any((OUT/n).exists() for n in ['STARTED.json','registry.json','metrics.json']),'no real registration artifacts')
result=dict(status='passed',named_checks=checks,count=len(checks),real_waveform_header_reads=0,real_audio_sample_reads=0,real_metadata_selection=0,models=0,network_bytes=0)
(OUT/'synthetic_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
