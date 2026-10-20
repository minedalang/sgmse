"""No real WAV samples opened. All WAV IO below uses a temporary synthetic signal."""
import importlib.util, pathlib, tempfile, math, struct, wave, json, subprocess, sys, ast
OUT=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prep',OUT/'prepare_inputs.py'); m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def main():
    checks=[]
    def ok(x,name): assert x,name;checks.append(name)
    for f in OUT.glob('*.py'):ast.parse(f.read_text())
    ok(True,'AST all scripts')
    s=[.2*math.sin(i*.17) for i in range(320)]
    ns=[('N1',[.4*math.cos(i*.11) for i in range(320)]),('N2',[.1*math.sin(i*.21)+.07 for i in range(320)])]
    a,M,rows=m.recipe(s,ns);ok(len(rows)==4 and 0<a<=1,'four recipes common downward scale')
    for r in rows:
        c=[a*v for v in s];n=[a*r['gain']*v for v in r['noise_samples']]
        ok(abs(10*math.log10(m.power(c)/m.power(n))-r['snr_db'])<1e-10,'physical component SNR '+r['noise']+str(r['snr_db']))
    ok(m.quantize([-.000001,0,.000001])==[-1,0,0],'floor not round/truncate')
    for bad in [[1.],[-1.00001]]:
        try:m.quantize(bad)
        except RuntimeError:pass
        else:raise AssertionError('saturation accepted')
    ok(True,'saturation rejection')
    with tempfile.TemporaryDirectory() as d:
        m.OUT=pathlib.Path(d)
        artifacts=[m.store(m.OUT,'signal'+str(i),[a*v for v in r['mix']]) for i,r in enumerate(rows)]
        ok(all(m.header(m.OUT/r['wav'])['frames']==320 for r in artifacts),'synthetic PCM16 headers and lengths')
        ok(all(r['saturation_count']==0 for r in artifacts),'synthetic floor serialization')
        bad=pathlib.Path(d)/'bad.wav'
        with wave.open(str(bad),'wb') as w:w.setparams((1,2,8000,0,'NONE',''));w.writeframes(b'\0'*64)
        try:m.header(bad)
        except RuntimeError:pass
        else:raise AssertionError('8k header accepted')
        ok(True,'reject mismatched WAV header')
    # Guard tests in fresh children: driver guard installed before forbidden action.
    pre="import importlib.util; s=importlib.util.spec_from_file_location('p',"+repr(str(OUT/'prepare_inputs.py'))+"); p=importlib.util.module_from_spec(s);s.loader.exec_module(p);p.guard([]);"
    for action in ['import torch','import numpy','import soundfile','import socket; socket.socket()','import subprocess;subprocess.run(["true"])']:
        r=subprocess.run([sys.executable,'-B','-c',pre+action],capture_output=True)
        ok(r.returncode!=0 and b'forbidden' in r.stderr,'guard '+action)
    try:m.child()
    except RuntimeError as e:ok('Supervisor required' in str(e),'missing supervisor rejected before source decoding')
    else:raise AssertionError('child without supervisor accepted')
    with tempfile.TemporaryDirectory() as d:
        m.OUT=pathlib.Path(d);m.BASE=m.OUT
        (m.OUT/'preparation_manifest.json').write_text('{}')
        (m.OUT/'INPUT_PREPARATION_STARTED.json').write_text('{}')
        try:m.supervise()
        except RuntimeError as e:ok('Existing CPU execution artifact' in str(e),'existing marker rejects duplicate execution')
        else:raise AssertionError('duplicate accepted')
    m.OUT=OUT;m.BASE=OUT.parent
    cfg=m.read(OUT/'config.yaml');ok(len(cfg['samples'])==8 and len(set(x['speaker'] for x in cfg['samples']))==4,'metadata dimensions')
    ok(m.locks(cfg['source_locks'])==cfg['source_locks'],'all source hashes bytes only')
    ok(not any((OUT/n).exists() for n in ['inputs','INPUT_PREPARATION_STARTED.json','metrics.json','input_registry.json']),'no actual execution artifacts')
    return dict(status='passed',checks=checks,real_audio_sample_decodes=0,model_imports=0,network_bytes=0,synthetic_signals=4)
if __name__=='__main__':
    r=main();(OUT/'synthetic_test_results.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r))
