"""Targeted AST CPU test only. No torch/native/Whisper imports, no subprocess."""
from pathlib import Path
import ast,hashlib,json,socket,urllib.request
from helpers import read,module
from blocked_events import BlockedEvents
S=Path(__file__).resolve().parent;cfg=read(S/'execution_config.json');source=cfg['blocked_local_probe_source'];path=source['path'];cal=module('calibration_targeted_test',cfg['calibration_driver'])
base=socket.socket;real_constructor_calls=[]
class ConstructorSentinel(base):
 def __init__(self,*args,**kwargs):real_constructor_calls.append(1);raise AssertionError('Base socket constructor must never run')
socket.socket=ConstructorSentinel
was_ipv6=socket.has_ipv6;socket.has_ipv6=True
events=BlockedEvents(path,source['sha256'],cfg['calibration_driver']);refuse=cal.offline_guard(events);events.bind_refuse(refuse)
def extracted(filename=path,module_name='urllib3.util.connection',family=None):
 tree=ast.parse(Path(path).read_text(),filename=filename);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_has_ipv6')
 if family:
  for node in ast.walk(fn):
   if isinstance(node,ast.Attribute) and node.attr=='AF_INET6':node.attr=family
 env=dict(socket=socket,__name__=module_name);exec(compile(ast.Module(body=[fn],type_ignores=[]),filename,'exec',dont_inherit=True),env);return env['_has_ipv6']
real=extracted();assert real('::1') is False;assert len(events.local_probe_refusals)==1 and len(events)==0
local=events.local_probe_refusals[0];assert local['constructor']['family']==int(socket.AF_INET6) and local['probe']['host']=='::1' and local['classification']=='blocked_local_ipv6_probe'
# Different host, module, filename, modified family/code, direct AF_INET, DNS and URL must remain unknown.
assert real('::2') is False
assert extracted(module_name='other_module')('::1') is False
assert extracted(filename='/tmp/untrusted_caller.py')('::1') is False
assert extracted(family='AF_INET')('::1') is False
for fn in [lambda:socket.socket(socket.AF_INET),lambda:socket.getaddrinfo('example.invalid',80),lambda:urllib.request.urlopen('https://example.invalid')]:
 try:fn()
 except RuntimeError:pass
 else:raise AssertionError('Original refusal disappeared')
assert len(events)==7 and len(events.all_events)==8 and not real_constructor_calls
# Repeat legitimate refusal; no expected-count=1 threshold exists.
assert real('::1') is False and len(events.local_probe_refusals)==2 and len(events)==7
assert not real_constructor_calls
# Execute only the extracted checkpoint body with mocks; unknown refusal stops before load.
driver=ast.parse((S/'run_pilot.py').read_text());child=next(n for n in driver.body if isinstance(n,ast.FunctionDef) and n.name=='child');checkpoint_node=next(n for n in child.body if isinstance(n,ast.FunctionDef) and n.name=='checkpoint')
from helpers import require
writes=[];env=dict(write=lambda *a:writes.append(a),OUT=Path('/unused'),d=dict(network_attempts=[{'unknown':True}],network_bytes=0),check=lambda:None,require=require)
exec(compile(ast.Module(body=[checkpoint_node],type_ignores=[]),'checkpoint_mock','exec'),env)
model_loads=[]
try:
 env['checkpoint']();model_loads.append('must never happen')
except RuntimeError:pass
else:raise AssertionError('Unknown refusal failed to stop checkpoint')
assert not model_loads and writes
print(json.dumps(dict(status='passed',local_probe_refusals=2,unknown_refusals=7,original_runtime_error_preserved=True,base_socket_constructor_calls=0,actual_network_bytes=0,model_imports=0,full_smoke_runs=0,tests=['::1 exact AST source/code/module','nonlocal_host','other_module','other_filename','AF_INET_modified_code','direct_AF_INET','DNS','URL','repeated_local_probe','unknown_checkpoint_before_load'])))
