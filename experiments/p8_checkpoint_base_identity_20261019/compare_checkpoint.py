"""Narrow static float-storage checkpoint byte-identity audit. Standard library only.
Restricted data.pkl symbols become inert descriptors; no torch/numpy import.
Only --self-test is preparation; --execute-after-root-review is Worker-only."""
from pathlib import Path
from collections import OrderedDict,Counter,defaultdict
import hashlib,importlib.abc,io,json,math,os,pickle,signal,sys,tempfile,time,zipfile
ROOT=Path(__file__).resolve().parent

class FloatStorageSymbol:pass
class DataModuleSymbol:pass
class Storage:
    def __init__(self,key,location,numel):self.key=key;self.location=location;self.numel=numel
class Tensor:
    def __init__(self,storage,offset,shape,stride):self.storage=storage;self.offset=offset;self.shape=shape;self.stride=stride

def rebuild(storage,offset,shape,stride,requires_grad,hooks,*extra):
    if not isinstance(storage,Storage) or type(offset) is not int or not isinstance(shape,(tuple,list)) or not isinstance(stride,(tuple,list)) or len(shape)!=len(stride) or extra:raise ValueError('Unsupported tensor metadata')
    if any(type(v) is not int for v in list(shape)+list(stride)):raise ValueError('Noninteger geometry')
    if not isinstance(requires_grad,bool) or not isinstance(hooks,(dict,OrderedDict,type(None))):raise ValueError('Unsupported rebuild flags')
    return Tensor(storage,offset,tuple(shape),tuple(stride))
class Restricted(pickle.Unpickler):
    def find_class(self,module,name):
        mapping={('collections','OrderedDict'):OrderedDict,('torch','FloatStorage'):FloatStorageSymbol,('torch._utils','_rebuild_tensor_v2'):rebuild,('sgmse.joint_data_module','JointWaveScpDataModule'):DataModuleSymbol}
        if (module,name) not in mapping:raise pickle.UnpicklingError('Forbidden GLOBAL '+module+'.'+name)
        return mapping[module,name]
    def persistent_load(self,pid):
        if not isinstance(pid,tuple) or len(pid)!=5 or pid[0]!='storage' or pid[1] is not FloatStorageSymbol or not isinstance(pid[2],str) or not pid[2].isdigit() or not isinstance(pid[3],str) or type(pid[4]) is not int or pid[4]<0:raise pickle.UnpicklingError('Unsupported persistent storage ID')
        return Storage(pid[2],pid[3],pid[4])

def parse(data):
    stream=io.BytesIO(data);value=Restricted(stream).load()
    if stream.read(1):raise ValueError('Trailing pickle bytes')
    if not isinstance(value,dict):raise ValueError('Expected checkpoint mapping')
    return value

def clean(value,depth=0):
    if depth>5:return '<depth-limit>'
    if value is None or isinstance(value,(str,bool,int)):return value
    if isinstance(value,float):return value if math.isfinite(value) else repr(value)
    if isinstance(value,(list,tuple)):return [clean(v,depth+1) for v in value[:100]]
    if isinstance(value,dict):return {str(k):clean(v,depth+1) for k,v in value.items() if not isinstance(v,Tensor)}
    return '<inert '+type(value).__name__+'>'

def hashfile(path):
    h=hashlib.sha256();n=0
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1048576),b''):h.update(block);n+=len(block)
    return dict(bytes=n,sha256=h.hexdigest())

class Checkpoint:
    def __init__(self,path,cap=1048576):
        self.path=str(path);self.zip=zipfile.ZipFile(path);names=self.zip.namelist()
        if len(names)!=len(set(names)):raise ValueError('Duplicate ZIP member')
        data=[n for n in names if n.endswith('/data.pkl')]
        if len(data)!=1:raise ValueError('Unique data.pkl required')
        self.prefix=data[0][:-len('data.pkl')];info=self.zip.getinfo(data[0])
        if info.file_size>cap:raise ValueError('Metadata size cap')
        raw=self.zip.read(data[0]);self.metadata_sha256=hashlib.sha256(raw).hexdigest();self.metadata_bytes=len(raw);self.value=parse(raw)
        order=self.prefix+'byteorder';self.byteorder=self.zip.read(order).decode() if order in names else 'unknown';self.storage_bytes_read=0;self.effective_tensor_bytes_hashed=0;self.cache={}
    def tensor(self,t):
        if not isinstance(t,Tensor):return dict(status='unknown',reason='not restricted tensor descriptor')
        desc=dict(dtype='float32',shape=list(t.shape),offset=t.offset,strides=list(t.stride),byteorder=self.byteorder,storage_key=t.storage.key,storage_numel=t.storage.numel,storage_location=t.storage.location)
        def unknown(reason):return dict(desc,status='unknown',reason=reason,effective_sha256=None)
        if self.byteorder not in ['little','big']:return unknown('unknown storage byteorder')
        if t.offset<0 or any(v<0 for v in t.shape) or any(v<0 for v in t.stride):return unknown('negative shape/offset/stride')
        step=1
        for size,stride in zip(reversed(t.shape),reversed(t.stride)):
            if size>1 and stride!=step:return unknown('non-contiguous or unsupported effective layout')
            step*=size
        count=math.prod(t.shape);length=count*4;offset=t.offset*4
        member=self.prefix+'data/'+t.storage.key
        if member not in self.zip.namelist():return unknown('missing storage ZIP member')
        info=self.zip.getinfo(member)
        if info.file_size!=t.storage.numel*4 or t.offset+count>t.storage.numel:return unknown('storage size/effective range mismatch')
        signature=(member,offset,length,self.byteorder)
        if signature not in self.cache:
            h=hashlib.sha256()
            with self.zip.open(member) as f:
                remaining=offset
                while remaining:
                    b=f.read(min(1048576,remaining));self.storage_bytes_read+=len(b);remaining-=len(b)
                    if not b:raise ValueError('Truncated skipped storage')
                remaining=length
                while remaining:
                    b=f.read(min(1048576,remaining));self.storage_bytes_read+=len(b);remaining-=len(b)
                    if not b or len(b)%4:raise ValueError('Truncated float32 span')
                    if self.byteorder=='big':b=b''.join(b[k:k+4][::-1] for k in range(0,len(b),4))
                    h.update(b)
            self.cache[signature]=h.hexdigest();self.effective_tensor_bytes_hashed+=length
        return dict(desc,status='supported',reason=None,effective_sha256=self.cache[signature],effective_bytes=length,canonical_byteorder='little')
    def close(self):self.zip.close()

def tensor_match(a,b):
    if a['status']!='supported' or b['status']!='supported':return 'unknown'
    return 'equal' if a['dtype']==b['dtype'] and a['shape']==b['shape'] and a['effective_sha256']==b['effective_sha256'] else 'different'

def auxiliary_ema(cp,raw):
    result=[]
    # Only direct top-level EMA containers; report actual structure, do not invent ordering semantics.
    for key,value in cp.value.items():
        if 'ema' not in str(key).lower():continue
        if isinstance(value,dict):lists=[(str(key)+'.'+str(k),v) for k,v in value.items() if isinstance(v,(list,tuple))]
        elif isinstance(value,(list,tuple)):lists=[(str(key),value)]
        else:lists=[]
        container=dict(path=str(key),metadata=clean(value),lists=[])
        for path,values in lists:
            records=[]
            for index,tensor in enumerate(values):
                record=cp.tensor(tensor);matches=[name for name,r in raw.items() if tensor_match(record,r)=='equal']
                records.append(dict(index=index,tensor=record,content_matches_raw_names=matches,matching_is_named_identity=False))
            container['lists'].append(dict(path=path,length=len(values),ordering='serialized list order only; parameter-name mapping unknown',entries=records))
        result.append(container)
    return result

def relevant_hparams(cp):
    h=cp.value.get('hyper_parameters',{})
    if not isinstance(h,dict):return dict(status='unknown, no hyperparameter mapping')
    fields=['train_type','mode2_include_noisy_main','frontend','frontend_trainable','rrsd_freeze_base','backbone','sde','transform_type','spec_abs_exponent','spec_factor']
    return {k:clean(h[k]) if k in h else '<not saved, do not infer runtime default>' for k in fields}

def audit(a,b,prefixes):
    ra=a.value.get('state_dict');rb=b.value.get('state_dict')
    if not isinstance(ra,dict) or not isinstance(rb,dict):raise ValueError('Missing state_dict')
    da={k:a.tensor(v) for k,v in ra.items() if any(str(k).startswith(p) for p in prefixes)};db={k:b.tensor(v) for k,v in rb.items() if any(str(k).startswith(p) for p in prefixes)}
    rows=[]
    for key in sorted(set(da)|set(db)):
        left=da.get(key);right=db.get(key);status=tensor_match(left,right) if left and right else 'unknown'
        rows.append(dict(key=key,status=status,reason='missing same RAW key in one checkpoint' if left is None or right is None else None,A=left,B=right))
    summary={p:dict(Counter(r['status'] for r in rows if r['key'].startswith(p))) for p in prefixes}
    ema_a=auxiliary_ema(a,db);ema_b=auxiliary_ema(b,da)
    return dict(primary_comparison='same_named_RAW_state_dict',per_tensor=rows,module_summary=summary,A_EMA=ema_a,B_EMA=ema_b,EMA_named_identity='unknown; list order/content matches never establish parameter-name mapping',A_relevant_hparams=relevant_hparams(a),B_relevant_hparams=relevant_hparams(b),A_hparams=clean(a.value.get('hyper_parameters')),B_hparams=clean(b.value.get('hyper_parameters')),limits='RAW identity is not EMA identity, historical initialization provenance, training-budget fairness, inference-configuration identity or output equality')

def execute():
    cfg=json.loads((ROOT/'config.yaml').read_text());start=time.monotonic()
    if (ROOT/'STARTED.json').exists():raise RuntimeError('Already started; no retry')
    def hard(*args):
        try:(ROOT/'hard_timeout.json').write_text('{"status":"failed","reason":"120s"}')
        finally:os._exit(124)
    signal.signal(signal.SIGALRM,hard);signal.setitimer(signal.ITIMER_REAL,cfg['max_seconds'])
    (ROOT/'STARTED.json').write_text(json.dumps(dict(pid=os.getpid())))
    allowed={str(Path(cfg[k]).resolve()) for k in ['source_A','source_B']};events=[]
    class Block(importlib.abc.MetaPathFinder):
        def find_spec(self,name,path=None,target=None):
            if name.split('.')[0] in {'torch','numpy','whisper','soundfile','torchaudio','transformers'}:events.append(name);raise RuntimeError('Forbidden model/library import')
    sys.meta_path.insert(0,Block())
    def guard(event,args):
        if event.startswith('socket.') or event in ['subprocess.Popen','os.system','os.posix_spawn','os.fork']:events.append(event);raise RuntimeError('Forbidden network/process')
        if event=='open' and isinstance(args[0],(str,bytes,os.PathLike)):
            p=Path(os.fsdecode(args[0])).resolve();mode=args[1];flags=args[2];write=(isinstance(mode,str) and any(k in mode for k in 'wax+')) or bool(flags&(os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND))
            if write and not p.is_relative_to(ROOT):events.append('outside write');raise RuntimeError('Checkpoint/source write forbidden')
            if not write and (str(p).startswith('/data/datasets/') or p.suffix.lower()=='.wav'):events.append('datasetread');raise RuntimeError('Waveform/dataset forbidden')
    sys.addaudithook(guard);d=dict(status='running',failures=[],forbidden_attempts=events,waveform_reads=0,model_imports=0,network_bytes=0,training_calls=0)
    handles=[];code_pre={p.name:hashfile(p) for p in ROOT.iterdir() if p.name in ['compare_checkpoint.py','config.yaml','analysis_plan.md','preparation_report.md']}
    try:
        d['source_pre']={k:hashfile(cfg[k]) for k in ['source_A','source_B']}
        if d['source_pre']['source_A']['sha256']!=cfg['source_A_expected_sha256']:raise ValueError('A source SHA mismatch')
        handles=[Checkpoint(cfg[k],cfg['max_pickle_metadata_bytes_per_source']) for k in ['source_A','source_B']]
        d['comparison']=audit(*handles,cfg['compare_prefixes']);d['metadata']=[dict(path=c.path,bytes=c.metadata_bytes,sha256=c.metadata_sha256,byteorder=c.byteorder,storage_bytes_read=c.storage_bytes_read,effective_tensor_bytes_hashed=c.effective_tensor_bytes_hashed) for c in handles]
        d['status']='completed'
    except BaseException as e:
        import traceback
        d['status']='failed';d['failures'].append(dict(error=repr(e),traceback=traceback.format_exc()))
    finally:
        for c in handles:c.close()
    try:
        d['source_post']={k:hashfile(cfg[k]) for k in ['source_A','source_B']}
        if d['source_post']!=d['source_pre'] or events:raise ValueError('Source changed or forbidden attempt')
    except BaseException as e:d['status']='failed';d['failures'].append(dict(postcheck=repr(e)))
    d['code_config_locks_pre']=code_pre;d['code_config_locks_post']={p.name:hashfile(p) for p in ROOT.iterdir() if p.name in code_pre}
    if d['code_config_locks_post']!=code_pre:d['status']='failed';d['failures'].append(dict(error='Code/config changed'))
    d['seconds']=time.monotonic()-start
    output=json.dumps(d,indent=2,allow_nan=False)+'\n'
    used=sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())
    if used+len(output.encode())+8192>cfg['max_directory_bytes']:
        d=dict(status='failed',failures=['Full output would exceed2MiB; no rows silently removed for success'],source_pre=d.get('source_pre'),source_post=d.get('source_post'),seconds=time.monotonic()-start);output=json.dumps(d,indent=2)+'\n'
    (ROOT/'metrics.json').write_text(output);(ROOT/'experiment_report.md').write_text('# Static checkpoint RAW identity: '+d['status']+'\n\nSee metrics for all same-key RAW descriptors/effectivebytes. Named EMA identity remainsunknown without proven positional mapping; identicalshape or contentmatch does not supply it. This is not performance evaluation or retrospective fair-budget proof. No waveform/model/torch/numpy/network/training. Sources unchanged pre/post or failure recorded; no automaticretry.\n')
    if time.monotonic()-start>cfg['max_seconds']:raise RuntimeError('Final120s exceeded')
    signal.setitimer(signal.ITIMER_REAL,0);print(json.dumps(dict(status=d['status'],seconds=d['seconds'])));return 0 if d['status']=='completed' else 1

def self_test():
    """Nine small synthetic safety/geometry tests. No true checkpoint/storage IO."""
    import struct
    checks=[]
    def ok(x,name):
        if not x:raise AssertionError(name)
        checks.append(name)
    class FixtureTensor:
        def __init__(self,offset=0,shape=(2,),stride=(1,)):self.offset=offset;self.shape=shape;self.stride=stride
        def __reduce__(self):return rebuild,(Storage('0','cpu',4),self.offset,self.shape,self.stride,False,OrderedDict())
    def metadata(value):
        b=io.BytesIO()
        class Writer(pickle.Pickler):
            def persistent_id(self,obj):return ('storage',FloatStorageSymbol,obj.key,obj.location,obj.numel) if isinstance(obj,Storage) else None
        Writer(b,protocol=2).dump(value);raw=b.getvalue()
        for old,new in [(f'c{__name__}\nrebuild\n','ctorch._utils\n_rebuild_tensor_v2\n'),(f'c{__name__}\nFloatStorageSymbol\n','ctorch\nFloatStorage\n')]:raw=raw.replace(old.encode(),new.encode())
        return raw
    def fixture(path,values=(1,2,3,4),offset=0,stride=(1,),big=False,duplicate=False):
        state=OrderedDict([('dnn.weight',FixtureTensor(offset,stride=stride)),('frontend.weight',FixtureTensor(offset,stride=stride))]);raw=metadata(dict(state_dict=state,ema=dict(shadow_params=[FixtureTensor(offset),FixtureTensor(offset)])))
        with zipfile.ZipFile(path,'w') as z:
            z.writestr('archive/data.pkl',raw);z.writestr('archive/byteorder',b'big' if big else b'little');z.writestr('archive/data/0',struct.pack(('>' if big else '<')+'ffff',*values))
        return path
    with tempfile.TemporaryDirectory(prefix='synthetic_',dir=ROOT) as temp:
        root=Path(temp);p=fixture(root/'a.zip');q=fixture(root/'b.zip',offset=1)
        a=Checkpoint(p);b=Checkpoint(q)
        try:
            va=a.tensor(a.value['state_dict']['dnn.weight']);vb=b.tensor(b.value['state_dict']['dnn.weight']);ok(va['effective_sha256']==hashlib.sha256(struct.pack('<ff',1,2)).hexdigest() and vb['effective_sha256']==hashlib.sha256(struct.pack('<ff',2,3)).hexdigest(),'effective storage offset exactbytes')
            ok(tensor_match(va,va)=='equal' and tensor_match(va,vb)=='different','equal and unequal effective tensor')
            records=auxiliary_ema(a,{k:a.tensor(v) for k,v in a.value['state_dict'].items()});entry=records[0]['lists'][0]['entries'][0]
            ok(len(entry['content_matches_raw_names'])==2 and not entry['matching_is_named_identity'],'duplicate content matches not named EMA mapping')
        finally:a.close();b.close()
        c=Checkpoint(fixture(root/'stride.zip',stride=(2,)))
        try:ok(c.tensor(c.value['state_dict']['dnn.weight'])['status']=='unknown','unsupported layout unknown')
        finally:c.close()
        c=Checkpoint(fixture(root/'big.zip',big=True));d=Checkpoint(p)
        try:ok(tensor_match(c.tensor(c.value['state_dict']['dnn.weight']),d.tensor(d.value['state_dict']['dnn.weight']))=='equal','byteorder canonicalization')
        finally:c.close();d.close()
        c=Checkpoint(fixture(root/'range.zip',offset=4))
        try:ok(c.tensor(c.value['state_dict']['dnn.weight'])['status']=='unknown','out-of-storage range unknown')
        finally:c.close()
        try:parse(b"cos\nsystem\n(S'NEVER_EXECUTE'\ntR.")
        except pickle.UnpicklingError:ok(True,'arbitrary GLOBAL rejected without import/execution')
        else:raise AssertionError('Unknown GLOBAL accepted')
        bad=Restricted(io.BytesIO(b''))
        try:bad.persistent_load(('storage',FloatStorageSymbol,'../escape','cpu',1))
        except pickle.UnpicklingError:ok(True,'unknown persistent storage format rejected')
        else:raise AssertionError('Bad persistentID accepted')
        try:parse(metadata(dict(state_dict={}))+b'garbage')
        except ValueError:ok(True,'trailing pickle rejected')
        else:raise AssertionError('Trailing accepted')
    result=dict(status='passed',count=len(checks),checks=checks,true_storage_reads=0,true_checkpoint_comparisons=0,torch_numpy_imports=0,waveform_reads=0)
    (ROOT/'synthetic_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result));return 0

if __name__=='__main__':
    if sys.argv[1:]==['--self-test']:sys.exit(self_test())
    elif sys.argv[1:]==['--execute-after-root-review']:sys.exit(execute())
    else:raise RuntimeError('Explicit preparation self-test or approved Worker command only')
