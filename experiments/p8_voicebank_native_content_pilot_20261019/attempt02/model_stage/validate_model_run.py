"""Independent CPU-only artifact/serialization/scoring verification after model run."""
from pathlib import Path
import ast,hashlib,importlib.util,io,json,math,collections,types,socket
STAGE=Path(__file__).resolve().parent;OUT=STAGE.parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def independent_aggregate(scores):
    result={}
    for kind in ['word','character']:
        sums={k:sum(s[kind][k] for s in scores) for k in ['S','D','I','H','N','N_hyp','edit_distance']}
        result[kind]=dict(sums,micro_rate=sums['edit_distance']/sums['N'],sentence_mean_rate=sum(s[kind]['rate'] for s in scores)/len(scores))
    return result

def validate():
    # Imports only array/serialization libraries. No torch/whisper import.
    import numpy as np
    import soundfile as sf
    cfg=read(STAGE/'execution_config.json');freeze=read(STAGE/'execution_manifest.json');m=read(OUT/'metrics.json');reg=read(OUT/'input_registry.json')
    assert m['status']=='completed' and m['frontend_calls']==m['backend_calls']==16 and m['transcribe_api_calls']==52
    assert m['enhancement_model_loads']==m['asr_model_loads']==1 and m['network_bytes']==m['training_calls']==0 and not m['network_attempts'] and not m['test_content_accessed']
    assert m['enhancement_release']['model_weakrefs_dead'] and m['supervisor_seconds']<=420
    # Independently verify every classified refusal; zero/multiple exact probes are both retained.
    probe=cfg['blocked_local_probe_source'];assert sha(probe['path'])==probe['sha256']
    tree=ast.parse(Path(probe['path']).read_text(),filename=probe['path']);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_has_ipv6')
    compiled=compile(ast.Module(body=[node],type_ignores=[]),probe['path'],'exec',dont_inherit=True);code=next(c for c in compiled.co_consts if isinstance(c,types.CodeType) and c.co_name=='_has_ipv6')
    code_hash=hashlib.sha256((code.co_code.hex()+repr(code.co_names)+repr(code.co_consts)).encode()).hexdigest()
    assert m['blocked_events']==m['local_probe_refusals'] and not m['network_attempts'] and m['unknown_network_refusal_count']==0
    assert m['blocked_local_probe_count']==len(m['local_probe_refusals'])
    for event in m['local_probe_refusals']:
        assert event['classification']=='blocked_local_ipv6_probe' and event['event']['event']=='network_refused'
        assert event['constructor']==dict(family=int(socket.AF_INET6),type=int(socket.SOCK_STREAM),proto=0)
        assert event['probe']==dict(path=probe['path'],function='_has_ipv6',module='urllib3.util.connection',host='::1',code_signature=code_hash,source_sha256=probe['sha256'])
        stack=event['stack'];assert [e['function'] for e in stack[:3]]==['refuse','__init__','_has_ipv6']
        assert stack[0]['path']==stack[1]['path']==cfg['calibration_driver'] and stack[2]['path']==probe['path']

    assert m['source_pre']==m['source_post']==m['supervisor_post_locks']==freeze['locks']
    for p,h in freeze['locks'].items():assert sha(p)==h,p
    for p,item in m['artifact_manifest'].items():assert sha(OUT/p)==item['sha256'] and (OUT/p).stat().st_size==item['bytes'],p
    assert sum(p.stat().st_size for p in OUT.parent.rglob('*') if p.is_file() and not p.is_symlink())<=40_000_000
    assert m['execution_manifest_sha256']==sha(STAGE/'execution_manifest.json')
    spec=importlib.util.spec_from_file_location('frozen_independent_score',cfg['calibration_validator']);ind=importlib.util.module_from_spec(spec);spec.loader.exec_module(ind)
    byname={r['basename']:r for r in reg['cases']};native={r['basename']:r for r in m['native_cases']};assert len(native)==16
    for basename,r in native.items():
        item=byname[basename];assert r['seed']==r['actual_torch_seed']==item['seed'] and r['input_sha256']==item['wav_sha256'] and r['nfe']==30
        assert r['backend_actual_norm_factor']==r['noisy_peak_native_reference'] and r['backend_internal_T_orig']==item['samples']
        assert r['backend_anchor']['sha256']==r['frontend_raw']['sha256'] and r['backend_condition']['sha256']==r['input']['sha256']
        geometry=r['native_length_match'];assert geometry['reference']==geometry['output']==item['samples'];assert geometry['left']==abs(geometry['input']-geometry['reference'])//2
        assert r['sampler_args']['N']==30 and r['sampler_args']['sde']=='OUVESDE' and r['sampler_args']['corrector']=='none' and not r['sampler_args']['trace_sink_present']
        for arm in ['frontend','off']:
            raw=np.load(OUT/r[arm+'_raw' if arm=='frontend' else 'backend_raw']['path'],allow_pickle=False);assert raw.dtype==np.float32 and raw.size==item['samples'] and np.isfinite(raw).all()
            desc=r[arm+'_raw' if arm=='frontend' else 'backend_raw'];assert hashlib.sha256(raw.tobytes()).hexdigest()==desc['sha256'] and list(raw.shape)==desc['shape']
            stats=r['frontend_preclip' if arm=='frontend' else 'off_preclip'];flat=raw.reshape(-1)
            for k,v in dict(above_one=int(np.sum(flat>1)),below_minus_one=int(np.sum(flat< -1)),at_positive_endpoint=int(np.sum(flat==1)),at_negative_endpoint=int(np.sum(flat== -1)),serialization_above_max_pcm16=int(np.sum(flat>32767/32768)),serialization_below_min_pcm16=int(np.sum(flat< -1))).items():assert stats[k]==v
            assert stats['peak']==float(np.abs(flat).max())
            bio=io.BytesIO();sf.write(bio,np.clip(flat,-1.,1.) if arm=='off' else flat,16000,format='WAV',subtype='PCM_16');wav=OUT/r[arm+'_pcm']['path'];assert bio.getvalue()==wav.read_bytes() and sha(wav)==r[arm+'_pcm']['sha256']
    references={r['id']:r for r in reg['references']};items=m['items'];assert len(items)==52 and len({r['key'] for r in items})==52
    for item in items:
        raw=read(OUT/item['raw_file']);sc=read(OUT/item['score_file']);assert sc==item['score']
        ref=references[item['id']];assert raw['publisher_reference']==ref and hashlib.sha256(bytes.fromhex(ref['raw_text_hex'])).hexdigest()==ref['raw_text_sha256'];assert bytes.fromhex(ref['raw_text_hex']).decode('utf-8-sig')==ref['original_text']
        rw,rc=ind.normal(ref['original_text']);hw,hc=ind.normal(raw['raw_result']['text'])
        assert sc['word']==ind.independent_align(rw,hw) and sc['character']==ind.independent_align(rc,hc)
        assert sc['reference_word_tokens']==rw and sc['hypothesis_word_tokens']==hw and sc['reference_scoring_chars']==''.join(rc) and sc['hypothesis_scoring_chars']==''.join(hc)
        assert sc['original_reference']==ref['original_text'] and sc['original_hypothesis']==raw['raw_result']['text']
        assert item['input']['wav_sha256']==sha(OUT/item['path']) and raw['request']['input']==item['input']
        audio,sr=sf.read(OUT/item['path'],dtype='float32');assert sr==16000 and len(audio)==item['input']['samples'] and hashlib.sha256(audio.tobytes()).hexdigest()==item['input']['array_sha256']
        assert item['decode_api_calls']==len(raw['decode_calls']) and all(r['options']==m['effective_decoding_options']==cfg['calibration_effective_options'] for r in raw['decode_calls'])
    assert m['internal_decode_api_calls']==sum(x['decode_api_calls'] for x in items)
    assert collections.Counter(x['arm'] for x in items)==dict(scaledclean=4,noisy=16,frontend=16,off=16)
    scores={x['key']:x['score'] for x in items}
    for label,strata in m['summary']['strata'].items():
        rows=reg['cases']
        if 'noise_N1' in label:rows=[r for r in rows if r['noise']=='N1']
        if 'noise_N2' in label:rows=[r for r in rows if r['noise']=='N2']
        if 'snr_-18' in label:rows=[r for r in rows if r['snr_db']==-18]
        elif 'snr_0' in label:rows=[r for r in rows if r['snr_db']==0]
        for arm in ['noisy','frontend','off']:assert strata[arm]==independent_aggregate([scores[r['basename']+'__'+arm] for r in rows])
    assert m['summary']['scaledclean']==independent_aggregate([x['score'] for x in items if x['arm']=='scaledclean'])
    for comp in m['summary']['per_case']:
        f=scores[comp['basename']+'__frontend'];b=scores[comp['basename']+'__off'];counts=collections.Counter()
        fp={r['reference_index']:r['op'] for r in f['word']['alignment'] if r['op']!='I'};bp={r['reference_index']:r['op'] for r in b['word']['alignment'] if r['op']!='I'}
        assert len(comp['transitions']['positions'])==f['word']['N']
        for row in comp['transitions']['positions']:
            i=row['reference_index'];assert row['frontend_op']==fp[i] and row['off_op']==bp[i]
            key={(True,True):'bothH',(True,False):'frontendH_to_OFFnotH',(False,True):'frontendnotH_to_OFFH',(False,False):'bothnotH'}[(fp[i]=='H',bp[i]=='H')];assert row['transition']==key;counts[key]+=1
        assert all(counts[k]==v for k,v in comp['transitions']['counts'].items())
        assert comp['transitions']['frontend_insertions']==f['word']['I'] and comp['transitions']['off_insertions']==b['word']['I']
        for kind in ['word','character']:
            for k,v in comp['off_minus_frontend'][kind].items():assert v==b[kind][k]-f[kind][k]
    return dict(status='passed',serialization_pairs=32,scoring_items=52,native_cases=16,metrics_sha256=sha(OUT/'metrics.json'),model_loads=0,network_bytes=0,source_locks=len(freeze['locks']),blocked_local_ipv6_probe_refusals=len(m['local_probe_refusals']),unknown_network_refusals=0,all_socket_creation_still_refused=True)
if __name__=='__main__':
    result=validate()
    with (OUT/'model_independent_validation.json').open('x') as f:json.dump(result,f,indent=2)
    print(json.dumps(result))
