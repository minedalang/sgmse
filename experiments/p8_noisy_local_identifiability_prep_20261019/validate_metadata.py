"""Independent I0 reconstruction. No import of registration driver, no PCM samples."""
from pathlib import Path
import hashlib,json,re,wave
OUT=Path(__file__).resolve().parent

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def expected_selection(source,cfg):
    # Independent collect/filter/sort; only manifest metadata, not source audio.
    seen=set();candidates=[]
    for i,line in enumerate(Path(source).read_bytes().splitlines(keepends=True),1):
        row=json.loads(line);sid=row['id'];assert sid not in seen;seen.add(sid)
        if row['local_partition']!='train':continue
        m=re.fullmatch('(p[0-9]+)_([0-9]+)',sid);assert m and m[1]==row['speaker']
        if row['speaker'] in cfg['excluded_speakers'] or int(m[2])<cfg['minimum_utterance_number']:continue
        h=row['clean_header'];duration=h['frames']/h['sample_rate']
        if not cfg['duration_seconds_inclusive'][0]<=duration<=cfg['duration_seconds_inclusive'][1]:continue
        assert h==row['noisy_header'] and h['sample_rate']==16000 and h['channels']==1 and h['subtype']=='PCM_16'
        candidates.append((row['speaker'],int(m[2]),sid,i,hashlib.sha256(line).hexdigest(),row))
    speakers=sorted({t[0] for t in candidates if sum(c[0]==t[0] for c in candidates)>=cfg['utterances_per_speaker']})
    assert len(speakers)>=cfg['speaker_count'];chosen=[]
    for speaker in speakers[:cfg['speaker_count']]:chosen.extend(sorted([r for r in candidates if r[0]==speaker])[:cfg['utterances_per_speaker']])
    return chosen

def validate():
    cfg=json.loads((OUT/'config.yaml').read_text());m=json.loads((OUT/'metrics.json').read_text());reg=json.loads((OUT/'registry.json').read_text());prep=json.loads((OUT/'preparation_manifest.json').read_text())
    assert not (OUT/'hard_timeout_failure.json').exists()
    assert m['status']=='completed' and m['pairs']==12 and m['speakers']==6 and m['audio_files_hashed']==24
    assert m['network_attempts']==[] and m['network_bytes']==m['audio_sample_decodes']==m['model_imports']==m['training_calls']==0 and m['seconds']<=60
    assert sha(OUT/'registry.json')==m['registry_sha256'] and sha(cfg['source_manifest'])==cfg['source_manifest_sha256']==reg['source_manifest_sha256']
    for n,h in prep['files'].items():assert sha(OUT/n)==h,n
    chosen=expected_selection(cfg['source_manifest'],cfg);assert len(chosen)==len(reg['records'])==12
    hashed_bytes=0
    for item,selection in zip(reg['records'],chosen):
        speaker,number,sid,line,linehash,row=selection
        assert item['id']==sid and item['speaker']==speaker and item['source_line_number']==line and item['source_line_sha256']==linehash and item['reference']==row['transcript']
        assert item['noise_recording_id']=='unknown' and item['residual_noise_purity']=='unknown' and item['alignment']==item['gain']=='unverified'
        for role in ['clean','noisy']:
            f=item['files'][role];p=Path(f['path']);assert str(p)==row[role+'_path'] and not p.is_symlink() and p.resolve().parent==Path('/data/datasets/vb/voicebank_demand/16k/train')/role and p.resolve().name==sid+'.wav'
            assert p.stat().st_size==f['bytes'] and sha(p)==f['sha256'];hashed_bytes+=f['bytes']
            with wave.open(str(p),'rb') as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth(),w.getcomptype())==(16000,1,2,'NONE')
                h=dict(sample_rate=16000,channels=1,frames=w.getnframes(),subtype='PCM_16')
            assert h==f['header']==row[role+'_header']
    assert m['wav_header_checks']==24 and m['full_wav_hash_bytes']==hashed_bytes
    assert not reg['noise_disjoint_ready'] and not reg['split_frozen'] and not reg['scientific_gate_ready'] and not m['noise_disjoint_ready']
    scan=reg['noise_metadata_scan'];policy=cfg['noise_metadata_scan'];assert not scan['noise_identity_resolved']
    assert scan['total_read_bytes']==sum(r['bytes'] for r in scan['files'])<=policy['max_total_read_bytes'] and scan['candidate_files_seen']<=policy['max_candidate_files']
    for r in scan['files']:
        p=Path(r['path']);roots=[Path(v).resolve() for v in cfg['noise_metadata_roots'] if p.resolve().is_relative_to(Path(v).resolve())];assert roots and not p.is_symlink()
        relative=p.resolve().relative_to(roots[0]);assert len(relative.parts)-1<=policy['max_depth'] and not any(c.lower().startswith(('test','valid')) for c in relative.parts)
        assert p.suffix.lower() in policy['allowed_suffixes'] and p.stat().st_size==r['bytes'] and sha(p)==r['sha256']
    assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.is_symlink())<cfg['budgets']['artifact_bytes']
    return dict(status='passed',pairs=12,speakers=6,rehashed_selected_files=24,actual_audio_samples_read=0,noise_disjoint_ready=False,split_frozen=False,registry_sha256=sha(OUT/'registry.json'))
if __name__=='__main__':
    r=validate()
    with (OUT/'independent_validation.json').open('x') as f:json.dump(r,f,indent=2)
    print(json.dumps(r))
