"""Synthetic scoring/guard tests only. Never import torch/whisper or decode audio."""
from pathlib import Path
import ast, hashlib, importlib.util, itertools, json, os, subprocess, sys, tempfile
P=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('calibration_cpu',P/'run_calibration.py');g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
vspec=importlib.util.spec_from_file_location('independent_cpu',P/'validate_completed_run.py');v=importlib.util.module_from_spec(vspec);vspec.loader.exec_module(v)
assert not any(m in sys.modules for m in ['torch','whisper','numpy','soundfile'])
assert g.scoring_tokens("Don’t ‘STOP’ 12.5 re-think café") == (["don't",'stop','12','5','re','think','caf'],list('dontstop125rethinkcaf'))
for a,b,counts in [('a b','a b',(0,0,0)),('a b','a c',(1,0,0)),('a b','a',(0,1,0)),('a','a b c',(0,0,2)),('a b','',(0,2,0))]:
 s=g.score(a,b)['word'];assert (s['S'],s['D'],s['I'])==counts
assert g.score('a','a b c')['word']['rate']==2
assert g.score('','x')['word']['rate'] is None
assert g.score('','')['word']['rate_status']=='undefined_empty_reference'
# Two substitutions wins the equal-distance reverse-token tie via diagonal preference.
tie=g.alignment(['a','b'],['b','a']);assert [s['op'] for s in tie['alignment']]==['S','S']
exhaustive=0
sequences=[list(t) for n in range(4) for t in itertools.product(['a','b'],repeat=n)]
for a in sequences:
 for b in sequences:assert g.alignment(a,b)==v.independent_align(a,b);exhaustive+=1
avg=g.aggregate([g.score('a b','a'),g.score('a','b')]);assert avg['word']['micro_rate']==2/3 and avg['word']['sentence_mean_rate']==.75
refs=g.references(g.read(P/'config.yaml'));assert [len(g.scoring_tokens(r['original_text'])[0]) for r in refs]==[12,9,9,9]
assert sum(len(g.scoring_tokens(r['original_text'])[0]) for r in refs)==39
for name,h in g.read(P/'config.yaml')['source_locks'].items():assert g.sha(name)==h
try:g.runtime();raise AssertionError('unguarded runtime accepted')
except RuntimeError as e:assert 'Supervisor required' in str(e)
with tempfile.TemporaryDirectory(prefix='cpu_guard_',dir=P) as d:
 p=Path(d);(p/'frozen_manifest.json').write_text('{}');(p/'STARTED.json').write_text('{}');saved=g.OUT;g.OUT=p
 try:g.supervise();raise AssertionError('STARTED accepted')
 except RuntimeError as e:assert 'Existing execution artifact STARTED.json' in str(e)
 finally:g.OUT=saved
 try:g.verify_locks({str(p/'STARTED.json'):'wrong'});raise AssertionError('wrong hash accepted')
 except RuntimeError:pass
# Child process exercises socket+URL guard without opening any connection.
code='''import importlib.util,sys,socket,urllib.request
spec=importlib.util.spec_from_file_location("g",sys.argv[1]);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
a=[];g.offline_guard(a)
for f in [lambda:socket.socket(),lambda:socket.create_connection(("127.0.0.1",1)),lambda:socket.getaddrinfo("example.invalid",443),lambda:urllib.request.urlopen("https://example.invalid")]:
 try:f();raise AssertionError("network not refused")
 except RuntimeError:pass
assert len(a)==4
'''
subprocess.run([sys.executable,'-c',code,str(P/'run_calibration.py')],check=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1'))
# Compile only the installed options dataclass AST: no package/module import.
import dataclasses, typing
cfg=g.read(P/'config.yaml');pkg=Path(cfg['whisper_package'])
node=next(n for n in ast.parse((pkg/'decoding.py').read_text()).body if isinstance(n,ast.ClassDef) and n.name=='DecodingOptions')
ns={'dataclass':dataclasses.dataclass,**vars(typing)}
exec(compile(ast.Module(body=[node],type_ignores=[]),'installed-options-AST','exec'),ns)
opts=cfg['transcribe_options'];effective={k:v for k,v in opts.items() if k in ns['DecodingOptions'].__dataclass_fields__}
effective.pop('temperature',None);effective.pop('best_of',None);effective['prompt']=[]
instance=ns['DecodingOptions'](**effective,temperature=0.0)
assert instance.beam_size==1 and instance.best_of is None and instance.fp16 is False and instance.prompt==[]
transcribe_node=next(n for n in ast.parse((pkg/'transcribe.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='transcribe')
api_names={a.arg for a in transcribe_node.args.kwonlyargs}|set(ns['DecodingOptions'].__dataclass_fields__)
assert set(opts)<=api_names
for name in ['run_calibration.py','validate_completed_run.py','test_preparation.py']:ast.parse((P/name).read_text())
assert not any((P/n).exists() for n in ['STARTED.json','outputs','metrics.json','experiment_report.md'])
result={'status':'passed','AST_parse':True,'safe_import_without_heavy_libraries':True,'exhaustive_independent_alignment_pairs':exhaustive,'normalization_and_ties':True,'empty_hypothesis_and_reference':True,'unclipped_WER_over_1':True,'micro_vs_sentence_mean':True,'raw_publisher_reference_hashes':True,'expected_reference_word_counts':[12,9,9,9],'source_hashes':True,'missing_supervisor_and_existing_STARTED_and_bad_hash_refused':True,'offline_socket_URL_guards':True,'actual_installed_options_dataclass_AST_only':True,'models':0,'audio_decodes':0,'network_calls':0,'test_script_SHA256':hashlib.sha256((P/'test_preparation.py').read_bytes()).hexdigest()}
(P/'preparation_test_results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
