"""Classify already-refused creation of a strictly identified local IPv6 probe.
Never create/allow a socket; never retain frames or frame-local dictionaries.
"""
from pathlib import Path
import ast,hashlib,json,socket,sys,types

def signature(code):
    return hashlib.sha256((code.co_code.hex()+repr(code.co_names)+repr(code.co_consts)).encode()).hexdigest()
def source_signature(path):
    tree=ast.parse(Path(path).read_text(),filename=str(path));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_has_ipv6']
    if len(nodes)!=1:raise RuntimeError('Missing unique frozen IPv6 probe')
    compiled=compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec',dont_inherit=True)
    return signature(next(c for c in compiled.co_consts if isinstance(c,types.CodeType) and c.co_name=='_has_ipv6'))

class BlockedEvents(list):
    def __init__(self,probe_path,probe_sha256,calibration_path):
        super().__init__();self.all_events=[];self.local_probe_refusals=[];self.probe_path=str(Path(probe_path).resolve());self.calibration_path=str(Path(calibration_path).resolve());self.refuse_code=None
        if hashlib.sha256(Path(self.probe_path).read_bytes()).hexdigest()!=probe_sha256:raise RuntimeError('IPv6 probe source lock changed')
        self.probe_sha256=probe_sha256;self.probe_signature=source_signature(self.probe_path)
    def bind_refuse(self,fn):self.refuse_code=fn.__code__
    def append(self,event):
        frame=sys._getframe(1);frames=[];record=dict(event=event if isinstance(event,str) else dict(event),classification='unknown_network_refusal',stack=[])
        try:
            cursor=frame
            while cursor is not None and len(frames)<20:
                frames.append(cursor);record['stack'].append(dict(path=str(Path(cursor.f_code.co_filename).resolve()),function=cursor.f_code.co_name,line=cursor.f_lineno));cursor=cursor.f_back
            if len(frames)>=3:
                refusal,creation,probe=frames[:3]
                if refusal.f_code is self.refuse_code and creation.f_code.co_name=='__init__' and str(Path(creation.f_code.co_filename).resolve())==self.calibration_path:
                    family=creation.f_locals.get('family');kind=creation.f_locals.get('type');proto=creation.f_locals.get('proto');args=refusal.f_locals.get('args')
                    if all(isinstance(x,int) for x in [family,kind,proto]):record['constructor']=dict(family=int(family),type=int(kind),proto=int(proto))
                    source_ok=str(Path(probe.f_code.co_filename).resolve())==self.probe_path and probe.f_code.co_name=='_has_ipv6' and probe.f_globals.get('__name__')=='urllib3.util.connection' and signature(probe.f_code)==self.probe_signature
                    if source_ok:
                        host=probe.f_locals.get('host');record['probe']=dict(path=self.probe_path,function='_has_ipv6',module='urllib3.util.connection',host=host if isinstance(host,(str,int,float,bool)) or host is None else 'non_scalar',code_signature=self.probe_signature,source_sha256=self.probe_sha256)
                    constructor_ok=isinstance(args,tuple) and len(args)==3 and all(isinstance(x,int) for x in args) and args==(family,kind,proto) and family==socket.AF_INET6 and kind==socket.SOCK_STREAM and proto==0 and creation.f_locals.get('fileno') is None
                    event_ok=isinstance(event,dict) and event.get('event')=='network_refused'
                    if source_ok and probe.f_locals.get('host')=='::1' and constructor_ok and event_ok:
                        record['classification']='blocked_local_ipv6_probe'
            self.all_events.append(record)
            if record['classification']=='blocked_local_ipv6_probe':self.local_probe_refusals.append(record)
            else:super().append(record)
        finally:
            # Explicitly discard all temporary frame references before returning to original refusal.
            frames.clear();del frame
            if 'cursor' in locals():del cursor
            if 'refusal' in locals():del refusal,creation,probe
