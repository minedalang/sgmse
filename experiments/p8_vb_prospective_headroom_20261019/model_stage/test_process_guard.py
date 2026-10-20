from pathlib import Path
import json
from helpers import read,observe
from process_guard import allow_local_build
S=Path(__file__).resolve().parent;cfg=read(S/'execution_config.json')
def reject(argv,cwd=None):
 try:allow_local_build(argv[0],argv,cwd,cfg)
 except RuntimeError:return
 raise AssertionError('guard accepted forbidden argv')
assert allow_local_build('/sbin/ldconfig',['/sbin/ldconfig','-p'],None,cfg)['kind']=='ctypes_ldconfig_readonly'
assert allow_local_build('/usr/bin/uname',['uname','-p'],None,cfg)['kind']=='platform_processor'
cmd=['/data/coding/sgmse/env/bin/python','-m','cuda.pathfinder._dynamic_libs.dynamic_lib_subprocess','canary','cudart']
assert allow_local_build(cmd[0],cmd,cfg['cuda_query_cwd'],cfg)['kind']=='cuda_pathfinder_canary'
for args in [['/sbin/ldconfig'],['/sbin/ldconfig','-X'],['/bin/sh','-c','true'],['curl','https://example.invalid'],[cmd[0],'-c','print(1)'],cmd[:-1]+['unknown']]:reject(args)
reject(cmd,'/tmp')
original_args=(object(),);original_kw={'named':object()};result=object();records=[]
def mock(*args,**kw):assert args==original_args and kw==original_kw;return result
assert observe(mock,lambda a,k:records.append((a,k)))(*original_args,**original_kw) is result
print(json.dumps(dict(status='passed',readonly_ldconfig=True,cuda_query_grammar=True,unknown_query_shell_network_rejected=True,args_kwargs_return_unchanged=True,actual_subprocesses=0)))
