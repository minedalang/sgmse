import ast, hashlib, json, pathlib, subprocess
import torch
R=pathlib.Path('/data/coding/sgmse'); O=R/'experiments/p8_noisy_condition_path_audit_20261018'
C=R/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def defaults(path):
 tree=ast.parse(path.read_text()); out={}
 for cls in tree.body:
  if not isinstance(cls,ast.ClassDef):continue
  for fn in cls.body:
   if isinstance(fn,ast.FunctionDef) and fn.name=='__init__':
    d={}
    for a,v in zip(fn.args.args[-len(fn.args.defaults):],fn.args.defaults):
     try:d[a.arg]=ast.literal_eval(v)
     except (ValueError,TypeError):d[a.arg]=ast.unparse(v)
    out[cls.name]=d
 return out
pre=sha(C)
ck=torch.load(C,map_location='cpu',weights_only=False)
hp=dict(ck.get('hyper_parameters',{})); sd=ck['state_dict']
defs={p:defaults(R/p) for p in ['sgmse/model.py','sgmse/joint_model.py','sgmse/backbones/ncsnpp_v2.py']}
keys=['backbone','train_type','mode2_include_noisy_main','cross_attn','film_modulate','film_levels','selector_type','evidence_selector_type','use_tfgrid_selector','selector_attn_levels','nf','ch_mult','num_res_blocks','attn_resolutions','sde','loss_type','loss_weighting','network_scaling','c_in','c_out','c_skip','progressive_input','progressive','backend_adapter_enable','residual_refine_enable','frontend','rrsd_enable','rrsd_freeze_base']
resolved={}
for k in keys:
 sources={f'{p}:{cl}':d[k] for p,cs in defs.items() for cl,d in cs.items() if k in d}
 resolved[k]={'present':k in hp,'saved_value':hp.get(k),'current_constructor_defaults':sources}
 if k in hp:resolved[k]['effective_value']=hp[k];resolved[k]['basis']='saved hyperparameter'
 elif sources:
  vals=list(sources.values()); resolved[k]['effective_value']=vals[0] if all(v==vals[0] for v in vals) else None;resolved[k]['basis']='current constructor default; see source-specific defaults'
 else:resolved[k]['basis']='not a constructor argument; do not invent saved/default value'
resolved['use_tfgrid_selector']['derived_current_value']=str(hp.get('selector_type','none')).lower() in ('tfgrid_a','tfgrid_b')
patterns={'bottleneck_cross_attention':['bottleneck_cross_attn'], 'tfgrid_evidence_selector':['evidence_selector','selector_context_agg'], 'decoder_film':['.cond_net.','.block.'], 'residual_regen':['res_encoder','h_pool_proj','gate_scale'], 'noise_prior_guidance':['film_head','noise_encoder'], 'all_dnn_attention_projections':['q_proj','k_proj','v_proj'], 'backend_adapter':['backend_adapter']}
presence={n:{'matching_dnn_keys':[k for k in sd if k.startswith('dnn.') and any(s in k for s in ss)]} for n,ss in patterns.items()}
for v in presence.values():v['count']=len(v['matching_dnn_keys'])
selected={k:list(v.shape) for k,v in sd.items() if k in ['dnn.all_modules.3.weight','dnn.all_modules.6.Conv_0.weight','dnn.all_modules.17.weight'] or k.startswith('dnn.bottleneck_cross_attn')}
codepaths=['sgmse/model.py','sgmse/backbones/ncsnpp_v2.py','sgmse/joint_model.py','sgmse/sampling/__init__.py','sgmse/sdes.py']
codehash={p:sha(R/p) for p in codepaths}
prior=R/'experiments/p8_10720_tail_condition20_20261018/launch_provenance.json'; pj=json.loads(prior.read_text())
match={p:{'prior_sha256':s,'current_sha256':sha(R/p),'matches':s==sha(R/p)} for p,s in pj['code_sha256'].items()}
post=sha(C)
m={'experiment_id':O.name,'status':'completed_read_only_no_inference','checkpoint':{'path':str(C),'sha256_before':pre,'sha256_after':post,'unchanged':pre==post,'global_step':ck.get('global_step'),'epoch':ck.get('epoch'),'cpu_load_count':1},'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip(),'git_status_before_record_creation':subprocess.check_output(['git','status','--short'],cwd=R,text=True).strip(),'hyperparameter_resolution':resolved,'conditioning_related_saved_hyperparameters':{k:v for k,v in hp.items() if any(s in k for s in ['selector','film','cross_attn','include_noisy','residual_refine'])},'state_dict':{'total_keys':len(sd),'dnn_keys':[k for k in sd if k.startswith('dnn.')],'selected_shapes':selected,'presence_tests':presence},'current_code_sha256':codehash,'historical_pilot_provenance':{'path':str(prior.relative_to(R)),'recorded_code_hash_comparison':match,'backbone_hash_not_recorded_in_prior':True},'safety':{'model_instantiated':False,'inference_run':False,'training_run':False,'checkpoint_written':False,'gpu_used':False},'prior_audits_read':['experiments/backend_adapter_architecture_audit_20261018/metrics.json','experiments/backend_adapter_architecture_audit_20261018/experiment_report.md','experiments/checkpoint_data_condition_audit_20261018_metrics.json','experiments/p8_10720_tail_condition20_20261018/config.yaml','experiments/p8_10720_tail_condition20_20261018/experiment_report.md']}
(O/'metrics.json').write_text(json.dumps(m,indent=2,default=str)+'\n')
print(json.dumps({'checkpoint_unchanged':pre==post,'resolved':{k:v.get('effective_value',v.get('derived_current_value')) for k,v in resolved.items()},'presence_counts':{k:v['count'] for k,v in presence.items()},'selected_shapes':selected,'prior_code_matches':{p:v['matches'] for p,v in match.items()},'metadata_bytes':(O/'metrics.json').stat().st_size},indent=2))
