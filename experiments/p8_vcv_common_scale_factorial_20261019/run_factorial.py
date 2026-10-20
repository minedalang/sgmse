#!/usr/bin/env python3
"""Prepared common-scale 2x2 conditional diagnostic; worker runs once after review."""
from pathlib import Path
import gc, hashlib, importlib.util, json, os, sys, time, traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def json_scalar(x):
    if hasattr(x,'item'):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError('Unsupported JSON value: '+type(x).__name__)

def dump(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False,default=json_scalar)+'\n')
def require(x,msg):
    if not x:raise ValueError(msg)

LIMITS=('Whole VCV identity changes include coarticulation and duration; not a surgical phoneme intervention. '
        'Common reference removes condition peak >1, while conflicting anchor/condition identities remain jointly OOD. '
        'CTC candidate likelihood responses are descriptive proxies, not correctness on conflicting cells, human recovery, PER or population statistics. '
        'Comparison with earlier native-reference rows is descriptive because scales differ. No further design iteration follows this experiment.')

def main():
    cfg=json.loads((HERE/'config.yaml').read_text())
    require(cfg['arms']==['off'] and cfg['training'] is False and cfg['downloads'] is False,'Strict configuration')
    out=HERE/'outputs';out.mkdir(exist_ok=False);(out/'ctc_arrays').mkdir()
    t0=time.monotonic();d=dict(status='running',experiment_id=HERE.name,frontend_forwards=0,backend_forwards=0,recognizer_forwards=0,native_k_gate_passed=False,cases=[],groups=[],limits=LIMITS)
    locks=dict(cfg['sha256_locks']);locks.update({str(p):sha(p) for p in [HERE/'config.yaml',Path(__file__)]})
    dump(out/'RUNNING.json',dict(started_at=time.time()))
    def checkpoint():
        require(time.monotonic()-t0<=600,'600 second task-boundary cap')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=30_000_000,'30MB artifact cap')
        dump(out/'progress.json',d)
    try:
        import numpy as np
        import torch
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.chdir(ROOT)
        require(all(sha(p)==h for p,h in locks.items()),'Frozen source lock mismatch')
        helper=load(ROOT/cfg['helper'],'verified_fixed_norm_off')
        follow=load(ROOT/cfg['scorer'],'factorial_follow');follow.HERE=HERE
        fm=load(ROOT/cfg['recipe_helper'],'factorial_recipe')
        old=json.loads((ROOT/cfg['original_recipe']).read_text());fresh=json.loads((ROOT/cfg['fresh_recipe']).read_text())
        previous=json.loads((ROOT/cfg['previous_metrics']).read_text())
        require(previous['status']=='completed' and previous['diagonal_gate_passed'] and previous['backend_forwards']==16,'Previous exact native gates unverified')
        priorbase=(ROOT/cfg['previous_metrics']).parent
        noises={r['noise_id']:np.load(fm.HERE/r['array_path'],allow_pickle=False)['waveform'] for r in old['noise']}
        rows={};inputs={};anchors={};natives={};groups=[]
        for speaker in ['f6','m5']:
            frozen=old if speaker=='f6' else fresh
            for phone in ['g','k']:
                sid=speaker+'a'+phone+'a1';row=next(r for r in frozen.get('clean',frozen.get('rows')) if r['sample_id']==sid);rows[sid]=row
                clean=np.load(fm.HERE/row['array_path'],allow_pickle=False)['waveform'] if speaker=='f6' else np.load(ROOT/'experiments/p8_q_vcv_speaker_replication_20261019/outputs'/(sid+'_clean.npy'),allow_pickle=False)
                require(clean.dtype==np.float64 and clean.shape==(32000,) and abs(float(np.sqrt(np.mean(clean*clean)))-.01)<1e-15,'Source RMS/dtype')
                for condition in cfg['conditions']:
                    rec=next(r for r in frozen['recipes'] if r['sample_id']==sid and r['condition']==condition)
                    x=fm.recipe_wave(rec,dict(noises,**{sid:clean}));inputs[condition,sid]=(x,clean,rec)
                    cached=next(r for r in previous['frontend_replays'] if r['sample_id']==sid and r['condition']==condition)
                    anchor=np.load(priorbase/cached['raw_path'],allow_pickle=False)
                    require(sha(priorbase/cached['raw_path'])==cached['raw_sha256'] and follow.ah(anchor)==cached['float32_sha256'] and anchor.shape==(32000,) and anchor.dtype==np.float32,'Cached frontend hash')
                    native=next(r for r in previous['cases'] if r['anchor_id']==sid and r['condition']==condition and not r['cross'])
                    require(native['exact_diagonal_raw_and_pcm_match'] and native['anchor_raw_sha256']==follow.ah(anchor),'Original/native anchor geometry differs')
                    anchors[condition,sid]=anchor;natives[condition,sid]=native
            g,k=speaker+'aga1',speaker+'aka1'
            for condition in cfg['conditions']:
                pg=float(np.max(np.abs(inputs[condition,g][0])));pk=float(np.max(np.abs(inputs[condition,k][0])));ref=max(pg,pk)
                require(ref==pk and pk>=pg,'Required k-native common-scale gate no longer applies')
                rg=inputs[condition,g][2];rk=inputs[condition,k][2]
                require(rg['noise_id']==rk['noise_id'] and rg['noise_gain']==rk['noise_gain'],'Physical noise/gain mismatch')
                noise=rg['noise_gain']*noises[rg['noise_id']];common_noise=noise/ref
                residual=(inputs[condition,g][0].astype(np.float64)-inputs[condition,k][0].astype(np.float64))-(inputs[condition,g][1]-inputs[condition,k][1])
                bound=float(4*np.finfo(np.float32).eps*ref)
                require(float(np.max(np.abs(residual)))<=bound,'Float32 decomposition differs')
                group=dict(speaker=speaker,condition=condition,g_id=g,k_id=k,common_ref=ref,g_native_peak=pg,k_native_peak=pk,seed=follow.seed(rows[g]['primary_pair_ids'][0]),normalized_noise_sha256=follow.ah(common_noise),normalized_noise_rms=float(np.sqrt(np.mean(common_noise**2))),float32_difference_error_max=float(np.max(np.abs(residual))),float32_error_bound=float(bound))
                require(group['seed']==follow.seed(rows[k]['primary_pair_ids'][0]),'Pair seed mismatch');groups.append(group)
        d['groups']=groups
        import joint_enhance_by_dir as mod
        checkpoint();model=mod.JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint'],map_location='cpu',strict=True,data_module_cls=mod.JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False);model.eval(no_ema=False).to('cuda:0')
        flags=dict(train_type=model.train_type,backbone=model.backbone,frontend=str(model.hparams.frontend),include_noisy_main=model.mode2_include_noisy_main,cross_attn=model.dnn.cross_attn,film_modulate=model.dnn.film_modulate,selector_type=model.dnn.selector_type,residual_refine_enable=model.residual_refine_enable,sde=type(model.sde).__name__,sampler=model.sde.sampler_type)
        require(flags==cfg['runtime_flags'],'Runtime flags changed');d['runtime_flags']=flags
        d['sampler_settings']=dict(N=30,predictor='reverse_diffusion',corrector='none',snr=.5,corrector_steps=1,t_eps=float(model.t_eps),ema=True,rrsd=False,regeneration='resolve(None,Y,C)',anchor_calibrator=None,score_calibrator=None)
        with torch.inference_mode():
            # Complete all four k/k exact gates before the other twelve cells.
            for stage in ['native_k_gate','remaining']:
                if stage=='remaining':
                    require(len(d['cases'])==4 and all(c['exact_native_k_raw_and_pcm_match'] for c in d['cases']),'Four native gates failed')
                    d['native_k_gate_passed']=True;dump(out/'NATIVE_K_GATES_PASSED.json',d['cases'])
                for group in groups:
                    for ap,cp in [('k','k')] if stage=='native_k_gate' else [('g','g'),('g','k'),('k','g')]:
                        checkpoint();condition=group['condition'];sid=group[ap+'_id'];donor=group[cp+'_id'];ref=group['common_ref'];torch.manual_seed(group['seed'])
                        x,meta=helper.fixed_norm_off(model,torch.from_numpy(anchors[condition,sid].copy()).to('cuda:0'),torch.from_numpy(inputs[condition,donor][0].copy()).to('cuda:0'),ref)
                        d['backend_forwards']+=1;require(meta['condition_normalized_peak']<=1 and not meta['outside_native_norm_range'],'Common scale failed normalized peak guard')
                        case_id=group['speaker']+'__'+condition+'__anchor_'+ap+'__noisy_'+cp
                        saved=follow.save_audio(out/'audio'/case_id,x)
                        case=dict(**saved,**meta,case_id=case_id,speaker=group['speaker'],condition=condition,anchor_id=sid,noisy_id=donor,anchor_origin=ap,noisy_donor=cp,common_ref=ref,seed=group['seed'],normalized_noise_sha256=group['normalized_noise_sha256'],anchor_raw_sha256=follow.ah(anchors[condition,sid]),quiet_padding=follow.quiet(x,rows[sid]),native_own_reference=natives[condition,sid]['refscale'])
                        if stage=='native_k_gate':
                            native=natives[condition,sid];case['exact_native_k_raw_and_pcm_match']=bool(np.array_equal(x,np.load(priorbase/native['raw_path'],allow_pickle=False)) and saved['raw_sha256']==native['raw_sha256'] and saved['wav_sha256']==native['wav_sha256'])
                            d['cases'].append(case);require(case['exact_native_k_raw_and_pcm_match'],'Native k/k exact replay failed; stop before non-native-scale cells')
                        else:
                            d['cases'].append(case)
                            if ap==cp:
                                native=natives[condition,sid];case['matches_own_native_reference_raw']=bool(np.array_equal(x,np.load(priorbase/native['raw_path'],allow_pickle=False)));case['scale_changed_from_native']=ref!=native['refscale']
                        checkpoint()
        for group in groups:
            cells=[c for c in d['cases'] if c['speaker']==group['speaker'] and c['condition']==group['condition']]
            require(len(cells)==4 and len({c['normalized_noise_sha256'] for c in cells})==1,'Four-cell noise geometry differs')
            for ap in ['g','k']:
                pair=[c for c in cells if c['anchor_origin']==ap]
                for key in ['normalized_anchor_sha256','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','regeneration_strength','common_ref','seed']:
                    require(pair[0][key]==pair[1][key],'Within-row geometry changed: '+key)
            group['same_noise_all_four_cells']=True;group['matched_initial_state_within_both_rows']=True
        del model;gc.collect();torch.cuda.empty_cache()
        from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        checkpoint();ex=Wav2Vec2FeatureExtractor.from_pretrained(str(fm.MODEL),local_files_only=True);recognizer=Wav2Vec2ForCTC.from_pretrained(str(fm.MODEL),local_files_only=True).to('cuda:0').eval();vocab=json.loads((fm.MODEL/'vocab.json').read_text());require(vocab['<pad>']==0 and recognizer.config.pad_token_id==0,'CTC blank mismatch');scoring=fm.prior()
        for case in d['cases']:
            checkpoint();s=follow.score(np.load(HERE/case['raw_path'],allow_pickle=False),rows[case['noisy_id']],ex,recognizer,vocab,scoring,'cuda:0',out/'ctc_arrays'/(case['case_id']+'.npz'));d['recognizer_forwards']+=1
            # Shared scorer uses nominal donor labels internally; conflict cells have no correctness endpoint.
            for key in ['top1_correct','binary_correct']:s.pop(key,None)
            case['ctc']=s;case['q_loglikelihood_g_minus_k']=float(s['all24_loglikelihoods']['g']-s['all24_loglikelihoods']['k'])
        for group in groups:
            q={c['anchor_origin']+c['noisy_donor']:c['q_loglikelihood_g_minus_k'] for c in d['cases'] if c['speaker']==group['speaker'] and c['condition']==group['condition']}
            group.update(q=q,anchor_effect=float(((q['gg']-q['kg'])+(q['gk']-q['kk']))/2),condition_effect=float(((q['gg']-q['gk'])+(q['kg']-q['kk']))/2),interaction=float((q['gg']-q['gk'])-(q['kg']-q['kk'])))
            group['previous_native_reference_rows_descriptive']=[dict(anchor_id=c['anchor_id'],noisy_id=c['noisy_id'],refscale=c['refscale'],q=c['ctc']['loglikelihood_ratio_g_minus_k']) for c in previous['cases'] if c['condition']==group['condition'] and c['anchor_id'] in [group['g_id'],group['k_id']]]
        require((d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards'])==(0,16,16),'Forward count mismatch')
        require(all(sha(p)==h for p,h in locks.items()),'Immutable sources changed');checkpoint();d.update(status='completed',immutable_sources_unchanged=True);dump(out/'COMPLETED.json',dict(status='completed'))
    except BaseException as exc:
        d.update(status='failed',error=str(exc),traceback=traceback.format_exc());dump(out/'FAILED.json',d);traceback.print_exc()
    d.update(wall_seconds=time.monotonic()-t0,sha256_locks=locks,all_preserved_audio_hashes_valid=all(sha(HERE/c['raw_path'])==c['raw_sha256'] and sha(HERE/c['wav_path'])==c['wav_sha256'] for c in d['cases']))
    dump(HERE/'metrics.json',d);dump(out/'audio_manifest.json',d['cases']);(out/'RUNNING.json').unlink(missing_ok=True)
    (HERE/'experiment_report.md').write_text('# Common-scale factorial conditional diagnostic\n\nStatus: '+d['status']+'. Frontend/backend/CTC calls: '+str((d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards']))+'.\n\n'+LIMITS+'\n\nAll four native k/k cells must match raw and PCM before the other twelve. Per-group four-cell q, anchor/condition effects and interaction are in metrics.json, without p-values or population claims. Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0. Failure outputs are preserved; diagnose rather than retry unchanged.\n')
    if d['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
