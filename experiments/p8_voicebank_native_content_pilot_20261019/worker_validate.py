"""Supplemental standard-library-only provenance and recorded-guard audit."""
from pathlib import Path
import hashlib, json, math, struct, wave, zipfile

P = Path(__file__).resolve().parent
read = lambda n: json.loads((P / n).read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
cfg, reg, met = read('config.yaml'), read('input_registry.json'), read('input_metrics.json')
def pcm(p):
    with wave.open(str(p), 'rb') as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) == (1, 2, 16000, 'NONE')
        return [v[0] / 32768 for v in struct.iter_unpack('<h', w.readframes(w.getnframes()))]
def power(x): return math.fsum(v*v for v in x)/len(x)
for key in ['source_pre', 'source_post', 'supervisor_source_post']:
    assert met[key] == cfg['source_locks'] == {p: sha(p) for p in cfg['source_locks']}
prepared = {str(P/n): h for n,h in read('preparation_manifest.json').items()}
for key in ['prepared_pre', 'prepared_post', 'supervisor_prepared_post']:
    assert met[key] == prepared == {p: sha(p) for p in prepared}
assert met['failures'] == met['network_attempts'] == []
assert met['model_loads'] == met['network_bytes'] == met['training_calls'] == 0
assert met['test_content_accessed'] is False and met['seconds'] < 120
assert sha(P/'input_registry.json') == met['input_registry_sha256']
manifest = {}
for line in Path(cfg['text_manifest']).read_text().splitlines():
    row = json.loads(line)
    if row['id'] in [g['id'] for g in reg['groups']]:
        assert row['id'] not in manifest
        manifest[row['id']] = row
references = []
with zipfile.ZipFile(cfg['text_zip']) as z:
    for ref in reg['references']:
        t = manifest[ref['id']]['transcript']; raw = z.read(t['archive_entry'])
        assert raw == bytes.fromhex(ref['raw_text_hex'])
        assert hashlib.sha256(raw).hexdigest() == ref['raw_text_sha256'] == t['raw_text_SHA256']
        assert len(raw) == t['raw_text_bytes']
        assert raw.decode('utf-8-sig') == ref['original_text'] == t['original_text']
        assert ref['archive_entry'] == t['archive_entry']
        references.append({'id': ref['id'], 'bytes': len(raw), 'sha256': ref['raw_text_sha256'], 'archive_entry': ref['archive_entry']})
noises = {n['id']: pcm(n['path']) for n in cfg['noises']}
clean_sources = {s['id']: pcm(s['audio']) for s in cfg['samples']}
checked = []
for group in reg['groups']:
    a = group['scale']; s = clean_sources[group['id']]; clean = [a*v for v in s]
    assert group['noise_start_sample'] == 80000 and group['noise_end_sample_exclusive'] == 80000+len(s)
    for row in [group['scaledclean']] + [r for r in reg['cases'] if r['id'] == group['id']]:
        if 'noise' in row:
            n = noises[row['noise']][80000:80000+len(s)]
            noise = [a*row['gain']*v for v in n]
            expected = [a*(v+row['gain']*w) for v,w in zip(s,n)]
            component = 10*math.log10(power(clean)/power(noise))
            assert component == row['component_snr_db'] and abs(component-row['snr_db']) < 1e-10
        else: expected = clean
        q = pcm(P/row['wav']); raw = [v[0] for v in struct.iter_unpack('<f', (P/row['float32']).read_bytes())]
        assert len(q) == len(raw) == len(expected) == row['samples']
        fields = dict(peak=max(map(abs,expected)), pcm_peak=max(map(abs,q)),
                      max_quantization_error=max(abs(v-w) for v,w in zip(q,expected)),
                      max_float32_rounding_error=max(abs(v-w) for v,w in zip(raw,expected)),
                      max_pcm_vs_float32_error=max(abs(v-w) for v,w in zip(q,raw)))
        for k,v in fields.items(): assert v == row[k], (row['wav'],k)
        assert fields['peak'] <= .90004 and fields['pcm_peak'] <= .90004
        assert fields['max_quantization_error'] <= 1/32768+1e-15
        assert fields['max_pcm_vs_float32_error'] <= 1/32768+fields['max_float32_rounding_error']+1e-15
        assert all(math.isfinite(v) for v in expected+q+raw)
        assert row['saturation_count'] == 0 and all(-1 < v < 32767/32768 for v in q)
        if 'noise' in row:
            residual = max(abs(v-c-n) for v,c,n in zip(q,clean,noise))
            assert residual == row['max_pcm_mixture_residual']
            assert residual <= 1/32768+fields['max_float32_rounding_error']+1e-14
            for values,key in [(q,'pcm_residual_noise_snr_db'),(raw,'float32_residual_noise_snr_db')]:
                assert 10*math.log10(power(clean)/power([v-c for v,c in zip(values,clean)])) == row[key]
        checked.append({'wav':row['wav'],**fields})
result = dict(status='passed', source_locks=12, preparation_locks=8, source_pre_post_current_equal=True,
              preparation_pre_post_current_equal=True, publisher_references=references,
              guard_artifacts=checked, input_metrics_sha256=sha(P/'input_metrics.json'),
              input_registry_sha256=sha(P/'input_registry.json'), model_loads=0, network_bytes=0)
with (P/'worker_validation.json').open('x') as f: json.dump(result,f,indent=2)
assert sum(p.stat().st_size for p in P.rglob('*') if p.is_file()) < 12_000_000
print(json.dumps({'status':'passed','artifacts':len(checked),'references':len(references)}))
