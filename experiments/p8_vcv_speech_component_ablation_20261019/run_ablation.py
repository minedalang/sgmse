#!/usr/bin/env python3
"""Frozen five-case speech removal diagnostic; prepared for one worker execution."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
sys.path.insert(0,str(ROOT))

def load(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()

def scalar(x):
    if hasattr(x,'item'):return x.item()
    if isinstance(x,Path):return str(x)
    raise TypeError(type(x).__name__)

def dump(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False,default=scalar)+'\n')
def require(x,msg):
    if not x:raise ValueError(msg)

LIMITS=('Selected natural nominal-label CTC transitions, not prevalence or human correctness. '
        'Whole known clean VCV speech component is removed from the noisy condition, not a surgical consonant. '
        'Frontend anchor remains unchanged. Joint disagreement and possible condition peaks above one are OOD; '
        'this is an oracle mechanism diagnostic, not deployable enhancement or proof of hearing/phoneme recovery. '
        'No variants, additional cases, parameter search or population statistics.')

def main():
    cfg=json.loads((HERE/'config.yaml').read_text());selection=json.loads((HERE/'selection_frozen.json').read_text())
    require(cfg['arms']==['off'] and cfg['training'] is False and cfg['downloads'] is False and len(selection['cases'])==5,'Frozen schema')
    out=HERE/'outputs';out.mkdir(exist_ok=False);(out/'ctc_arrays').mkdir()
    t0=time.monotonic();d=dict(status='running',experiment_id=HERE.name,frontend_forwards=0,backend_forwards=0,recognizer_forwards=0,native_gate_passed=False,cases=[],limits=LIMITS)
    locks=dict(cfg['sha256_locks']);locks.update({str(p):sha(p) for p in [HERE/'config.yaml',HERE/'selection_frozen.json',Path(__file__)]})
    dump(out/'RUNNING.json',dict(started_at=time.time()))
    def checkpoint():
        require(time.monotonic()-t0<=600,'600 second task-boundary cap')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file())<=30_000_000,'30MB artifact cap')
        dump(out/'progress.json',d)
    try:
        import numpy as np
        import torch
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.chdir(ROOT)
        require(all(sha(p)==h for p,h in locks.items()),'Immutable source lock mismatch')
        helper=load(ROOT/cfg['helper'],'speech_verified_helper');follow=load(ROOT/cfg['scorer'],'speech_score');follow.HERE=HERE
        fm=load(ROOT/cfg['recipe_helper'],'speech_recipe')
        old=json.loads((ROOT/cfg['original_recipe']).read_text());fresh=json.loads((ROOT/cfg['fresh_recipe']).read_text())
        noises={r['noise_id']:np.load(fm.HERE/r['array_path'],allow_pickle=False)['waveform'] for r in old['noise']}
        prepared={}
        for case in selection['cases']:
            sid=case['sample_id'];condition=case['condition'];base=ROOT/case['source_experiment'];frozen=old if case['source_group']=='original' else fresh
            row=next(r for r in frozen.get('clean',frozen.get('rows')) if r['sample_id']==sid)
            clean=np.load(fm.HERE/row['array_path'],allow_pickle=False)['waveform'] if case['source_group']=='original' else np.load(base/'outputs'/(sid+'_clean.npy'),allow_pickle=False)
            rec=next(r for r in frozen['recipes'] if r['sample_id']==sid and r['condition']==condition)
            noisy=fm.recipe_wave(rec,dict(noises,**{sid:clean}));noise64=rec['noise_gain']*noises[rec['noise_id']];noise=noise64.astype(np.float32)
            ref=float(np.max(np.abs(noisy)));bound=float(4*np.finfo(np.float32).eps*max(ref,float(np.max(np.abs(noise)))))
            require(np.array_equal((clean+noise64).astype(np.float32),noisy),'Original mixture differs')
            error=float(np.max(np.abs((noisy.astype(np.float64)-noise.astype(np.float64))-clean)))
            require(error<=bound,'Known speech removal decomposition exceeds float32 bound')
            a=case['frontend_audio'];anchor=np.load(base/a['raw_path'],allow_pickle=False)
            require(sha(base/a['raw_path'])==a['raw_sha256'] and follow.ah(anchor)==a['float32_sha256'] and anchor.shape==(32000,) and anchor.dtype==np.float32,'Cached native frontend differs')
            require(clean.dtype==np.float64 and clean.shape==(32000,) and abs(float(np.sqrt(np.mean(clean*clean)))-.01)<1e-15,'Clean RMS guard')
            active=slice(8000,8000+row['resampled_samples'])
            proof=dict(case_id=case['case_id'],refscale=ref,clean_float64_sha256=follow.ah(clean),noisy_float32_sha256=follow.ah(noisy),physical_noise_float64_sha256=follow.ah(noise64),noise_only_float32_sha256=follow.ah(noise),normalized_physical_noise_sha256=follow.ah(noise64/ref),float32_speech_removal_error_max=error,float32_error_bound=bound,source_rms=float(np.sqrt(np.mean(clean**2))),source_active_rms=float(np.sqrt(np.mean(clean[active]**2))),condition_active_change_rms=float(np.sqrt(np.mean((noisy[active].astype(np.float64)-noise[active].astype(np.float64))**2))),noise_only_peak=float(np.max(np.abs(noise))),normalized_noise_peak=float(np.max(np.abs(noise/ref))),noise_to_original_peak_ratio=float(np.max(np.abs(noise)))/ref)
            require(proof['condition_active_change_rms']>0,'No active speech component removed')
            prepared[case['case_id']]=(row,anchor,noisy,noise,proof)
        dump(out/'source_proofs.json',[v[4] for v in prepared.values()])
        import joint_enhance_by_dir as mod
        checkpoint();model=mod.JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint'],map_location='cpu',strict=True,data_module_cls=mod.JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False);model.eval(no_ema=False).to('cuda:0')
        flags=dict(train_type=model.train_type,backbone=model.backbone,frontend=str(model.hparams.frontend),include_noisy_main=model.mode2_include_noisy_main,cross_attn=model.dnn.cross_attn,film_modulate=model.dnn.film_modulate,selector_type=model.dnn.selector_type,residual_refine_enable=model.residual_refine_enable,sde=type(model.sde).__name__,sampler=model.sde.sampler_type)
        require(flags==cfg['runtime_flags'],'Runtime model flags differ');d['runtime_flags']=flags
        d['sampler_settings']=dict(N=30,predictor='reverse_diffusion',corrector='none',corrector_steps=1,snr=.5,t_eps=float(model.t_eps),ema=True,rrsd=False,regeneration='resolve(None,Y,C)',anchor_calibrator=None,score_calibrator=None)
        with torch.inference_mode():
            for ablated in [False,True]:
                if ablated:
                    require(len(d['cases'])==5 and all(c['exact_native_raw_and_pcm_match'] for c in d['cases']),'Five native gates failed')
                    d['native_gate_passed']=True;dump(out/'NATIVE_GATES_PASSED.json',d['cases'])
                for case in selection['cases']:
                    checkpoint();row,anchor,noisy,noise,proof=prepared[case['case_id']];torch.manual_seed(case['seed'])
                    x,meta=helper.fixed_norm_off(model,torch.from_numpy(anchor.copy()).to('cuda:0'),torch.from_numpy((noise if ablated else noisy).copy()).to('cuda:0'),proof['refscale']);d['backend_forwards']+=1
                    name=case['case_id']+('__noise_only' if ablated else '__native');a=follow.save_audio(out/'audio'/name,x)
                    r=dict(**a,**meta,case_id=case['case_id'],ablated=ablated,seed=case['seed'],refscale=proof['refscale'],anchor_raw_sha256=follow.ah(anchor),normalized_physical_noise_sha256=proof['normalized_physical_noise_sha256'],condition_source='noise_only_known_clean_removed' if ablated else 'original_noisy',quiet_padding=follow.quiet(x,row))
                    d['cases'].append(r)
                    if not ablated:
                        base=ROOT/case['source_experiment'];prior=case['off_audio'];r['exact_native_raw_and_pcm_match']=bool(np.array_equal(x,np.load(base/prior['raw_path'],allow_pickle=False)) and a['raw_sha256']==prior['raw_sha256'] and a['wav_sha256']==prior['wav_sha256'])
                        require(r['exact_native_raw_and_pcm_match'],'Native exact replay failed; no ablation permitted')
                    else:
                        native=next(v for v in d['cases'] if v['case_id']==case['case_id'] and not v['ablated'])
                        for key in ['normalized_anchor_sha256','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','regeneration_strength','seed','refscale','normalized_physical_noise_sha256']:
                            require(r[key]==native[key],'Native/ablation geometry changed: '+key)
                        r['matched_anchor_initial_state_and_noise_geometry']=True
                    checkpoint()
        del model;gc.collect();torch.cuda.empty_cache()
        from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        checkpoint();ex=Wav2Vec2FeatureExtractor.from_pretrained(str(fm.MODEL),local_files_only=True);recognizer=Wav2Vec2ForCTC.from_pretrained(str(fm.MODEL),local_files_only=True).to('cuda:0').eval();vocab=json.loads((fm.MODEL/'vocab.json').read_text());require(vocab['<pad>']==0 and recognizer.config.pad_token_id==0,'Blank mismatch');scoring=fm.prior()
        for case in selection['cases']:
            row=prepared[case['case_id']][0];native=next(r for r in d['cases'] if r['case_id']==case['case_id'] and not r['ablated']);ablated=next(r for r in d['cases'] if r['case_id']==case['case_id'] and r['ablated'])
            checkpoint();s=follow.score(np.load(HERE/ablated['raw_path'],allow_pickle=False),row,ex,recognizer,vocab,scoring,'cuda:0',out/'ctc_arrays'/(case['case_id']+'.npz'));d['recognizer_forwards']+=1;ablated['ctc']=s;native['ctc']=case['off_score'];native['ctc_reused']=True
            target=case['nominal_target'];rival=case['preset_rival'];pred=s['top1_prediction']
            outcome=('retained' if pred==target else 'lost') if case['transition_kind']=='repair' else ('reverted_to_nominal' if pred==target else 'remained_original_error' if pred==case['off_prediction'] else 'third_class')
            d.setdefault('comparisons',[]).append(dict(case_id=case['case_id'],transition_kind=case['transition_kind'],nominal_target=target,preset_rival=rival,pre_prediction=case['pre_prediction'],native_off_prediction=case['off_prediction'],ablated_prediction=pred,outcome=outcome,native_target_vs_rival=float(native['ctc']['all24_loglikelihoods'][target]-native['ctc']['all24_loglikelihoods'][rival]),ablated_target_vs_rival=float(s['all24_loglikelihoods'][target]-s['all24_loglikelihoods'][rival]),ablated_target_vs_prelabel=float(s['all24_loglikelihoods'][target]-s['all24_loglikelihoods'][case['pre_prediction']]),target_vs_prelabel_degenerate=target==case['pre_prediction'],interpretation='External nominal clean identity remains target despite noise-only condition; CTC proxy only'))
        require((d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards'])==(0,10,5),'Forward counts differ')
        require(all(sha(p)==h for p,h in locks.items()),'Immutable sources changed');checkpoint();d.update(status='completed',immutable_sources_unchanged=True);dump(out/'COMPLETED.json',dict(status='completed'))
    except BaseException as exc:
        d.update(status='failed',error=str(exc),traceback=traceback.format_exc());dump(out/'FAILED.json',d);traceback.print_exc()
    d.update(wall_seconds=time.monotonic()-t0,sha256_locks=locks,all_preserved_audio_hashes_valid=all(sha(HERE/r['raw_path'])==r['raw_sha256'] and sha(HERE/r['wav_path'])==r['wav_sha256'] for r in d['cases']))
    dump(HERE/'metrics.json',d);dump(out/'audio_manifest.json',d['cases']);(out/'RUNNING.json').unlink(missing_ok=True)
    (HERE/'experiment_report.md').write_text('# Speech-component ablation\n\nStatus: '+d['status']+'. Frontend/backend/new CTC calls: '+str((d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards']))+'.\n\n'+LIMITS+'\n\nFive native RAW/NPY/PCM gates precede all five noise-only calls. Per-case repair/harm outcomes, full candidate scores and fixed-rival margins appear in metrics.json. Quiet padding RMS and speech/noise source proofs are retained. Source: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0. Failures are preserved; diagnose before any retry.\n')
    if d['status']!='completed':raise SystemExit(1)

if __name__=='__main__':main()
