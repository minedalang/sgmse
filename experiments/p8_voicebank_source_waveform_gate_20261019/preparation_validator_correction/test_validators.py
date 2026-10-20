"""CPU mocks invoke actual RangeReader.get; no HTTP, audio or model imports."""
from pathlib import Path
import ast, hashlib, importlib.util, json, sys, tempfile
BASE=Path(__file__).resolve().parent.parent
spec=importlib.util.spec_from_file_location('gate_cpu_test',BASE/'run_gate.py')
g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
assert not any(n in sys.modules for n in ['requests','torch','torchaudio','soundfile'])
ast.parse((BASE/'run_gate.py').read_text())
results=[]
class Raw:
    def __init__(self):self.reads=0
    def read(self,n,decode_content=False):self.reads+=1;return b'P'[:n]
class Response:
    def __init__(self,extra,status=206):
        self.status_code=status;self.headers={'Content-Range':'bytes 0-0/10','Content-Length':'1','Content-Encoding':'identity',**extra};self.raw=Raw();self.closed=False
    def close(self):self.closed=True

def scenario(name,headers,expected,statuses=None):
    with tempfile.TemporaryDirectory(prefix='cpu_mock_',dir=BASE/'preparation_validator_correction') as d:
        state={'requests':[],'network_bytes_accounted':0}
        reader=g.RangeReader({'archive_size':10},state,Path(d),lambda:None)
        for i,(h,ok) in enumerate(zip(headers,expected)):
            response=Response(h,(statuses or [206]*len(headers))[i]);reader.open_response=lambda _,r=response:r
            try:
                assert reader.get(0,0,'case'+str(i))==b'P';passed=True
            except RuntimeError:passed=False
            assert passed==ok,(name,i,state)
            assert response.closed
            assert response.raw.reads==(1 if ok else 0),(name,i,'body read on refusal')
            assert state['requests'][-1]['payload_bytes']==(1 if ok else 0)
            assert len(list(Path(d).glob('*.bin')))==sum(expected[:i+1])
        results.append({'name':name,'passed':True,'responses':len(headers),'actual_body_reads':[r['payload_bytes'] for r in state['requests']]})
scenario('absent_first',[{}],[False])
scenario('blank_first',[{'ETag':'  ','Last-Modified':'\t'}],[False])
scenario('empty_first',[{'ETag':'','Last-Modified':''}],[False])
scenario('etag_only_stable',[{'ETag':'"v1"'}]*2,[True,True])
scenario('last_modified_only_stable',[{'Last-Modified':'Mon, 19 Oct 2026 00:00:00 GMT'}]*2,[True,True])
scenario('both_stable',[{'ETag':'"v1"','Last-Modified':'date'}]*2,[True,True])
scenario('blank_secondary_not_version',[{'ETag':'"v1"','Last-Modified':'  '},{'ETag':'"v1"'}],[True,True])
scenario('etag_changed',[{'ETag':'"v1"'},{'ETag':'"v2"'}],[True,False])
scenario('modified_changed',[{'Last-Modified':'date1'},{'Last-Modified':'date2'}],[True,False])
scenario('both_then_one_missing',[{'ETag':'"v1"','Last-Modified':'date'},{'ETag':'"v1"'}],[True,False])
scenario('etag_then_missing_with_other',[{'ETag':'"v1"'},{'Last-Modified':'date'}],[True,False])
scenario('all_missing_later',[{'ETag':'"v1"'},{}],[True,False])
scenario('blank_later',[{'ETag':'"v1"'},{'ETag':' '}],[True,False])
scenario('status200_no_body',[{'ETag':'"v1"'}],[False],[200])
out={'status':'passed','actual_RangeReader_get':True,'AST_parse':True,'safe_import':True,'network_calls':0,'models':0,'audio_decodes':0,'resample_calls':0,'scenarios':results,'driver_SHA256':hashlib.sha256((BASE/'run_gate.py').read_bytes()).hexdigest()}
(BASE/'preparation_validator_correction/test_results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
