"""Only synthetic numerical/PCM and guard mocks; never open real audio samples."""
import importlib.util, json, math, pathlib, subprocess, sys, tempfile
HERE=pathlib.Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('prep',HERE/'prepare_inputs.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
def reject(fn):
    try: fn()
    except RuntimeError: return
    raise AssertionError('Expected guard rejection')
s=[.2,-.3,.1,0.,.04]*20
ns=[('N1',[.1,-.2,.2,.1,-.05]*20),('N2',[-.1,.2,-.2,-.1,.05]*20)]
a,M,rows=m.recipe(s,ns)
assert 0<a<=1 and a==min(1,.9/M) and len(rows)==4
assert [(r['noise'],r['snr_db']) for r in rows]==[('N1',-18),('N1',0),('N2',-18),('N2',0)]
for r in rows:
    assert abs(10*math.log10(m.power(s)/m.power([r['gain']*v for v in r['noise_samples']]))-r['snr_db'])<1e-10
assert m.recipe([1e-7,-1e-7],[('N1',[1e-7,-1e-7]),('N2',[1e-7,-1e-7])])[0]==1
reject(lambda:m.recipe([0.,0.],ns));reject(lambda:m.power([float('nan')]));reject(lambda:m.quantize([1.]));reject(lambda:m.recipe(s,[('N1',[1.])]))
assert m.quantize([-1/65536,0,1/65536])==[-1,0,0]
with tempfile.TemporaryDirectory() as t:
    m.OUT=pathlib.Path(t)
    for i,r in enumerate(rows):
        x=[a*v for v in r['mix']];rec=m.store(m.OUT,'synthetic'+str(i),x)
        assert rec['samples']==len(s) and rec['saturation_count']==0
        assert m.header(m.OUT/rec['wav'])['frames']==len(s)
# Audit hook stays in fresh disposable process. No network connection or model import happens.
program="""import importlib.util,socket,sys
s=importlib.util.spec_from_file_location('p',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
e=[];m.guard(e)
for f in [lambda:socket.socket(),lambda:socket.getaddrinfo('localhost',80),lambda:__import__('torch'),lambda:__import__('whisper')]:
 try:f()
 except RuntimeError:pass
 else:raise AssertionError('guard missed')
assert len(e)==2
"""
subprocess.run([sys.executable,'-B','-c',program,str(HERE/'prepare_inputs.py')],check=True)
print(json.dumps(dict(status='passed',synthetic_pcm_outputs=4,guards=['socket','DNS','torch','whisper','zero_power','nonfinite','saturation','length'],real_audio_decodes=0,model_loads=0,network_bytes=0)))
