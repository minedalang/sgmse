#!/usr/bin/env python3
"""Run once by experiment worker after review. All diagonal gates precede crosses."""
from pathlib import Path
import gc, hashlib, importlib.util, json, os, sys, time, traceback
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(ROOT))
CONDITIONS = ['N1_snr-6', 'N2_snr-6']

def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()

def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def require(value, message):
    if not value: raise ValueError(message)

LIMITS = ('Controlled conditional/OOD mechanism diagnostic, not a new enhancement method. '
          'Whole two-second VCV identity is swapped, including coarticulation and duration; '
          'this is not a surgical central phoneme intervention. Comparisons are within frozen '
          'anchor rows only; row reference scales differ. Condition peaks may exceed the '
          'native normalized range; no clipping or renormalization. Nominal labels and CTC '
          'proxies are not human correctness, PER, input identifiability or population statistics.')

def main():
    import yaml
    cfg = yaml.safe_load((HERE/'config.yaml').read_text())
    require(cfg['arms'] == ['off'] and all(type(cfg[k]) is bool and not cfg[k] for k in ['training', 'downloads']), 'Strict schema')
    out = HERE/'outputs'; out.mkdir(exist_ok=False); (out/'ctc_arrays').mkdir()
    t0 = time.monotonic()
    d = dict(experiment_id=HERE.name, status='running', frontend_forwards=0,
             backend_forwards=0, recognizer_forwards=0, diagonal_gate_passed=False,
             limits=LIMITS, cases=[], source_guards=[], comparisons=[])
    dump(out/'RUNNING.json', dict(started_at=time.time()))
    locks = dict(cfg['sha256_locks'])
    locks.update({str(p): sha(p) for p in [HERE/'config.yaml', Path(__file__), HERE/'fixed_norm_off.py']})
    def checkpoint():
        require(time.monotonic()-t0 <= 600, '600 second task-boundary budget exceeded')
        require(sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file()) <= 50_000_000, '50MB cap exceeded')
        dump(out/'progress.json', d)
    try:
        import numpy as np
        import torch
        os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'
        os.chdir(ROOT)
        require(all(sha(p) == h for p,h in locks.items()), 'Frozen source/checkpoint lock mismatch')
        follow = load(ROOT/cfg['scoring_helper'], 'swap_isolated_follow'); follow.HERE = HERE
        fm = load(ROOT/cfg['recipe_helper'], 'swap_fixed_recipe')
        helper = load(HERE/'fixed_norm_off.py', 'swap_helper')
        old = json.loads((ROOT/cfg['original_recipe']).read_text())
        fresh = json.loads((ROOT/cfg['fresh_recipe']).read_text())
        noises = {}
        for r in old['noise']:
            p = fm.HERE/r['array_path']
            require(sha(p)==r['array_sha256'], 'Noise archive mismatch')
            noises[r['noise_id']] = np.load(p,allow_pickle=False)['waveform']
            require(follow.ah(noises[r['noise_id']])==r['float64_sha256'], 'Noise waveform mismatch')
        rows = {}; inputs = {}; priors = {}; baseline_scores = {}; frontends = {}; refs = {}; seeds = {}
        for source in cfg['sources']:
            base = ROOT/source['experiment']; frozen = old if source['speaker']=='f6' else fresh
            require(json.loads((base/'metrics.json').read_text())['status']=='completed', 'Prior incomplete')
            manifest = json.loads((base/'outputs/audio_manifest.json').read_text())
            scores = json.loads((base/'outputs/per_sample.json').read_text())
            originals = json.loads((ROOT/cfg['original_baseline']).read_text()) if source['speaker']=='f6' else scores
            for phone in ['g','k']:
                sid = source['speaker']+'a'+phone+'a1'
                row = next(r for r in frozen.get('clean',frozen.get('rows')) if r['sample_id']==sid)
                rows[sid] = row; seeds[sid] = follow.seed(row['primary_pair_ids'][0])
                if source['speaker']=='f6': clean = np.load(fm.HERE/row['array_path'],allow_pickle=False)['waveform']
                else: clean = np.load(base/'outputs'/(sid+'_clean.npy'),allow_pickle=False)
                require(clean.dtype == np.float64 and clean.shape==(32000,), 'Source shape/dtype')
                require(abs(float(np.sqrt(np.mean(clean*clean)))-.01)<1e-15, 'Source RMS guard')
                for condition in CONDITIONS:
                    rec = next(r for r in frozen['recipes'] if r['sample_id']==sid and r['condition']==condition)
                    c = dict(noises); c[sid] = clean
                    x = fm.recipe_wave(rec,c); inputs[condition,sid] = (x,clean,rec)
                    refs[condition,sid] = float(np.max(np.abs(x)))
                    require(0<refs[condition,sid]<1, 'Native reference peak guard')
                    for arm in ['clean','noisy','y_pre','off']:
                        a = next(r for r in manifest if r['sample_id']==sid and r['condition']==condition and r['arm']==arm)
                        require(sha(base/a['raw_path'])==a['raw_sha256'] and sha(base/a['wav_path'])==a['wav_sha256'], 'Prior audio changed')
                        priors[condition,sid,arm] = (base,a)
                    for arm in ['y_pre','off']:
                        baseline_scores[condition,sid,arm] = next(r for r in scores if r['sample_id']==sid and r['condition']==condition and r['arm']==arm)
                    baseline_scores[condition,sid,'noisy'] = next(r for r in originals if r['sample_id']==sid and r['condition']==condition and (source['speaker']=='f6' or r['arm']=='baseline'))
                    baseline_scores[condition,sid,'clean'] = next(r for r in originals if r['sample_id']==sid and r['condition']=='padded_clean')
        for source in cfg['sources']:
            g,k = [source['speaker']+'a'+p+'a1' for p in ['g','k']]
            require(seeds[g]==seeds[k], 'Pair seed mismatch')
            for condition in CONDITIONS:
                xg,cg,rg=inputs[condition,g]; xk,ck,rk=inputs[condition,k]
                require(rg['noise_id']==rk['noise_id'] and rg['noise_gain']==rk['noise_gain'], 'Physical noise differs')
                noise = rg['noise_gain']*noises[rg['noise_id']]
                require(np.array_equal(cg+noise, cg+rk['noise_gain']*noises[rk['noise_id']]), 'Float64 noise mismatch')
                residual = (xg.astype(np.float64)-xk.astype(np.float64))-(cg-ck)
                bound = 4*np.finfo(np.float32).eps*max(float(np.max(np.abs(xg))),float(np.max(np.abs(xk))))
                require(float(np.max(np.abs(residual))) <= bound, 'Float32 decomposition guard')
                d['source_guards'].append(dict(speaker=source['speaker'],condition=condition,
                    physical_noise_float64_sha256=follow.ah(noise),float32_difference_error_max=float(np.max(np.abs(residual))),
                    float32_error_bound=bound,noise_identical_before_float32_rounding=True,
                    clean_g_rms=float(np.sqrt(np.mean(cg*cg))),clean_k_rms=float(np.sqrt(np.mean(ck*ck)))))
        for (condition,sid),(x,clean,rec) in inputs.items():
            donor=sid.replace('aga1','aka1') if 'aga1' in sid else sid.replace('aka1','aga1')
            ref=refs[condition,sid]; noise=rec['noise_gain']*noises[rec['noise_id']]
            normalized_noise=noise/ref
            normalized_error=(x.astype(np.float64)/ref-inputs[condition,donor][0].astype(np.float64)/ref)-(clean-inputs[condition,donor][1])/ref
            bound=8*np.finfo(np.float32).eps*max(float(np.max(np.abs(x))),float(np.max(np.abs(inputs[condition,donor][0]))))/ref
            require(float(np.max(np.abs(normalized_error)))<=bound,'Normalized decomposition differs')
            d.setdefault('row_guards',[]).append(dict(condition=condition,anchor_id=sid,refscale=ref,
                normalized_physical_noise_sha256=follow.ah(normalized_noise),noise_rms_after_frozen_reference=float(np.sqrt(np.mean(normalized_noise**2))),
                source_support_rms=float(np.sqrt(np.mean(clean[8000:8000+rows[sid]['resampled_samples']]**2))),
                normalized_float32_difference_error_max=float(np.max(np.abs(normalized_error))),normalized_error_bound=bound))
        dump(out/'baseline_scores_reused.json', [dict(condition=c,sample_id=sid,arm=a,score=s) for (c,sid,a),s in baseline_scores.items()])
        import joint_enhance_by_dir as mod
        checkpoint()
        model=mod.JointSemambaSgmseModel.load_from_checkpoint(cfg['checkpoint'],map_location='cpu',strict=True,data_module_cls=mod.JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False)
        model.eval(no_ema=False).to('cuda:0')
        flags=dict(train_type=model.train_type,backbone=model.backbone,frontend=str(model.hparams.frontend),include_noisy_main=model.mode2_include_noisy_main,cross_attn=model.dnn.cross_attn,film_modulate=model.dnn.film_modulate,selector_type=model.dnn.selector_type,residual_refine_enable=model.residual_refine_enable,sde=type(model.sde).__name__,sampler=model.sde.sampler_type)
        require(flags==cfg['runtime_flags'], 'Runtime settings mismatch'); d['runtime_flags']=flags
        d['sampler_settings']=dict(predictor='reverse_diffusion',corrector='none',N=30,corrector_steps=1,snr=.5,t_eps=float(model.t_eps),regeneration='resolve(None,Y,C)',rrsd=False,anchor_calibrator=None,score_calibrator=None,ema=True)
        # Recompute all eight native frontend outputs without changing RNG.
        with torch.inference_mode():
            for (condition,sid),(x,clean,rec) in inputs.items():
                checkpoint(); y=torch.from_numpy(x.copy()).to('cuda:0')[None,None,:]
                torch.manual_seed(seeds[sid]); cpu=torch.get_rng_state().clone(); cuda=torch.cuda.get_rng_state().clone()
                norm=y.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)
                post=mod._match_length(model._frontend_forward(y/norm)*norm,y)
                require(torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state()), 'Frontend consumes RNG')
                d['frontend_forwards']+=1
                base,a=priors[condition,sid,'y_pre']; raw=post.squeeze().cpu().numpy().astype(np.float32)
                require(np.array_equal(raw,np.load(base/a['raw_path'],allow_pickle=False)) and follow.ah(raw)==a['float32_sha256'], 'Frontend native raw mismatch')
                frontends[condition,sid]=post.squeeze()
                saved=follow.save_audio(out/'frontend'/condition/sid,raw)
                d.setdefault('frontend_replays',[]).append(dict(condition=condition,sample_id=sid,raw_float32_sha256=follow.ah(raw),rng_unchanged=True,**saved))
            # Two explicit stages: no cross can occur until every diagonal has passed.
            for cross in [False,True]:
                if cross:
                    require(len(d['cases'])==8 and all(c['exact_diagonal_raw_and_pcm_match'] for c in d['cases']), 'Diagonal gate failed')
                    d['diagonal_gate_passed']=True; dump(out/'DIAGONALS_PASSED.json',d['cases'])
                for (condition,sid),(x,clean,rec) in inputs.items():
                    checkpoint(); donor=sid.replace('aga1','aka1') if 'aga1' in sid else sid.replace('aka1','aga1')
                    if not cross: donor=sid
                    torch.manual_seed(seeds[sid])
                    result,meta=helper.fixed_norm_off(model,frontends[condition,sid],torch.from_numpy(inputs[condition,donor][0].copy()).to('cuda:0'),refs[condition,sid])
                    d['backend_forwards']+=1
                    case_id=condition+'__anchor_'+sid+'__condition_'+donor
                    a=follow.save_audio(out/'audio'/case_id,result)
                    a.update(meta,case_id=case_id,condition=condition,anchor_id=sid,noisy_id=donor,anchor_label=rows[sid]['consonant'],noisy_label=rows[donor]['consonant'],cross=cross,refscale=refs[condition,sid],seed=seeds[sid],anchor_raw_sha256=priors[condition,sid,'y_pre'][1]['float32_sha256'],noisy_raw_sha256=follow.ah(inputs[condition,donor][0]),quiet_padding=follow.quiet(result,rows[sid]))
                    if not cross:
                        base,p=priors[condition,sid,'off']
                        exact=np.array_equal(result,np.load(base/p['raw_path'],allow_pickle=False)) and a['wav_sha256']==p['wav_sha256'] and a['raw_sha256']==p['raw_sha256']
                        a['exact_diagonal_raw_and_pcm_match']=bool(exact); d['cases'].append(a)
                        require(exact, 'STOP: native helper replay differs; no cross inference permitted')
                    else:
                        native=next(c for c in d['cases'] if c['condition']==condition and c['anchor_id']==sid and not c['cross'])
                        for key in ['normalized_anchor_sha256','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','regeneration_strength','refscale','seed']:
                            require(a[key]==native[key], 'Within-row geometry changed: '+key)
                        a['matched_initial_state_and_anchor_geometry']=True; d['cases'].append(a)
                    checkpoint()
        del model,post,y; frontends.clear(); gc.collect(); torch.cuda.empty_cache()
        from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        checkpoint(); ex=Wav2Vec2FeatureExtractor.from_pretrained(str(fm.MODEL),local_files_only=True)
        recognizer=Wav2Vec2ForCTC.from_pretrained(str(fm.MODEL),local_files_only=True).to('cuda:0').eval()
        vocab=json.loads((fm.MODEL/'vocab.json').read_text()); require(vocab['<pad>']==0 and recognizer.config.pad_token_id==0,'Blank mismatch')
        scoringmod=fm.prior()
        for case in d['cases']:
            if case['cross']:
                checkpoint(); raw=np.load(HERE/case['raw_path'],allow_pickle=False)
                s=follow.score(raw,rows[case['noisy_id']],ex,recognizer,vocab,scoringmod,'cuda:0',out/'ctc_arrays'/(case['case_id']+'.npz'))
                d['recognizer_forwards']+=1
            else: s=baseline_scores[case['condition'],case['anchor_id'],'off']
            ll=s['all24_loglikelihoods']; signed=float(ll['g']-ll['k'])
            binary='g' if signed>0 else 'k' if signed<0 else None
            case['ctc']=dict(follows_anchor_binary=binary==case['anchor_label'],follows_noisy_binary=binary==case['noisy_label'],score=s,reused_diagonal=not case['cross'],loglikelihood_ratio_g_minus_k=signed,
                binary_prediction_gk='g' if signed>0 else 'k' if signed<0 else None,
                follows_anchor_top1=s['top1_prediction']==case['anchor_label'],follows_noisy_top1=s['top1_prediction']==case['noisy_label'])
        for native in [c for c in d['cases'] if not c['cross']]:
            cross=next(c for c in d['cases'] if c['cross'] and c['condition']==native['condition'] and c['anchor_id']==native['anchor_id'])
            delta=cross['ctc']['loglikelihood_ratio_g_minus_k']-native['ctc']['loglikelihood_ratio_g_minus_k']
            d['comparisons'].append(dict(condition=native['condition'],anchor_id=native['anchor_id'],refscale=native['refscale'],native_prediction=native['ctc']['score']['top1_prediction'],swapped_prediction=cross['ctc']['score']['top1_prediction'],delta_loglikelihood_ratio_g_minus_k=delta,change_towards_swapped_nominal_identity=delta*(1 if cross['noisy_label']=='g' else -1)>0,interpretation='descriptive within-row; no candidate-count statistics'))
        require((d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards'])==(8,16,8), 'Forward budget mismatch')
        for c in d['cases']+d['frontend_replays']: require(sha(HERE/c['raw_path'])==c['raw_sha256'] and sha(HERE/c['wav_path'])==c['wav_sha256'],'Output changed')
        require(all(sha(p)==h for p,h in locks.items()), 'Prior/source/checkpoint changed')
        checkpoint(); d.update(status='completed',readonly_unchanged=True)
        dump(out/'COMPLETED.json',dict(status='completed'))
    except BaseException as exc:
        d.update(status='failed',error=str(exc),traceback=traceback.format_exc())
        dump(out/'FAILED.json',d); traceback.print_exc()
    d['all_preserved_output_hashes_valid']=all(sha(HERE/c['raw_path'])==c['raw_sha256'] and sha(HERE/c['wav_path'])==c['wav_sha256'] for c in d['cases']+d.get('frontend_replays',[]))
    d.update(wall_seconds=time.monotonic()-t0,sha256_locks=locks)
    dump(HERE/'metrics.json',d); dump(out/'audio_manifest.json',d['cases']); (out/'RUNNING.json').unlink(missing_ok=True)
    (HERE/'experiment_report.md').write_text('# Fixed normalization condition swap\n\nStatus: '+d['status']+'\n\n'+LIMITS+'\n\nFrontend/backend/new recognizer forwards: '+str((d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards']))+'. All native diagonals must match saved raw float32 and PCM exactly before cross cases. Failure details and partial outputs are preserved; diagnose before another attempt. Per-row predictions, g-minus-k CTC likelihood ratios, quiet padding RMS, initial state and normalization hashes are in metrics.json.\n\nSource: The Consonant Challenge Corpus, DOI 10.5281/zenodo.3538223, CC BY 4.0.\n')
    if d['status']!='completed': raise SystemExit(1)

if __name__=='__main__': main()
