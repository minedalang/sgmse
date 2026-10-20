#!/usr/bin/env python3
"""Fresh-speaker fixed VCV replication; worker execution only after frozen retrieval."""
from pathlib import Path
import gc,hashlib,importlib.util,json,os,sys,time,traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
FOLLOW=ROOT/'experiments/p8_q_vcv_enhance_follow_20261018/run_follow.py'
PREV=ROOT/'experiments/p8_q_vcv_fixed_noise_20261018'
CONDITIONS=['N1_snr-6','N2_snr-6','N1_snr+6','N2_snr+6']
def load(p,n):
    s=importlib.util.spec_from_file_location(n,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
h=load(FOLLOW,'isolated_follow');h.HERE=HERE;h.CONDITIONS=CONDITIONS;h.CAP=200_000_000
sha,dump,ah=h.sha,h.dump,h.ah
LIMITS='Fresh speakers held out from previous project analyses only; pretrained corpus exposure is unknown. Full two-second carrier SNR is not phoneme-local SNR; quiet fractions differ from historical30second nominal training SNR. External nominal labels are not model truth. No human recoverability, ignored input evidence, natural DNS generalization, significance, confidence interval, WER or DNSMOS claim. Failed noisy gates remain unqualified but all frozen conditions execute if the clean diagnostic gate passes.'
def main():
    cfg=json.loads((HERE/'config.yaml').read_text());out=HERE/'outputs';out.mkdir(exist_ok=False);(out/'ctc_arrays').mkdir();dump(out/'RUNNING.json',{'started':time.time()});t0=time.monotonic();d={'status':'running','experiment_id':HERE.name,'frontend_forwards':0,'backend_forwards':0,'recognizer_forwards':0,'limits':LIMITS};scores=[];audio={}
    try:
        import numpy as np
        import soundfile as sf
        from scipy.signal import resample_poly
        import torch
        from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';sys.path.insert(0,str(ROOT));os.chdir(ROOT)
        fm=load(PREV/'fixed_noise.py','fixed_helpers');scorer=fm.prior();manifest=json.loads((HERE/'retrieval/selected_manifest_frozen.json').read_text());rows=scorer.validate(manifest);assert len(rows)==16 and manifest['speakers']==cfg['speakers']
        priorrecipe=json.loads((PREV/'inputs/recipe_frozen.json').read_text());oldmanifest=json.loads(fm.MANIFEST.read_text());assert not {r['speaker'] for r in rows}&{r['speaker'] for r in oldmanifest['files']}
        locks={str(ROOT/k):v for k,v in cfg['driver_source_locks'].items()};locks.update({str(HERE/'config.yaml'):sha(HERE/'config.yaml'),str(Path(__file__)):sha(__file__),str(HERE/'retrieval/selected_manifest_frozen.json'):sha(HERE/'retrieval/selected_manifest_frozen.json')})
        for r in rows:locks[r['wav_path']]=r['sha256']
        for rel,digest in priorrecipe['model_files_sha256'].items():locks[str(fm.MODEL/rel)]=digest
        locks[str(h.CKPT)]=h.CKSHA
        assert all(sha(p)==digest for p,digest in locks.items()),'Source SHA lock mismatch'
        common=json.loads((ROOT/'experiments/p8_10720_snr_control_20261018/metrics.json').read_text())['enhance_kwargs'];assert common['N']==30 and common['snr']==.5
        def deadline():h.require(time.monotonic()-t0<=600,'10minute task-boundary budget reached')
        def rms(x):return float(np.sqrt(np.mean(x*x,dtype=np.float64)))
        caches={};recipes=[];cleanrows={};seedmatrix=[]
        for r in rows:
            w,sr=sf.read(r['wav_path'],dtype='float64',always_2d=True);assert sr==25000 and w.shape[1]==1 and np.isfinite(w).all();v=resample_poly(w[:,0],16,25);assert 0<len(v)<=24000
            x=np.zeros(32000,dtype=np.float64);x[8000:8000+len(v)]=v;gain=.01/rms(x);x*=gain;assert abs(rms(x)-.01)<1e-15
            row=dict(r,resampled_samples=len(v),derived_gain=gain,source_onset_sample=8000);cleanrows[r['sample_id']]=row;caches[r['sample_id']]=x
            np.save(out/(r['sample_id']+'_clean.npy'),x,allow_pickle=False)
        for nr in priorrecipe['noise']:
            p=PREV/nr['array_path'];assert sha(p)==nr['array_sha256'];locks[str(p)]=nr['array_sha256'];x=np.load(p,allow_pickle=False)['waveform'];assert ah(x)==nr['float64_sha256'];caches[nr['noise_id']]=x
        for condition in ['padded_clean']+CONDITIONS:
            for sid,row in cleanrows.items():
                ni=None if condition=='padded_clean' else condition[:2];snr=None if ni is None else int(condition.split('snr')[1]);gain=None if ni is None else float(.01*10**(-snr/20));x=caches[sid] if ni is None else caches[sid]+gain*caches[ni];assert np.max(np.abs(x))<1
                rec={'condition':condition,'sample_id':sid,'noise_id':ni,'snr_db':snr,'noise_gain':gain,'float64_sha256':ah(x),'model_input_float32_sha256':ah(x.astype(np.float32)),'peak':float(np.max(np.abs(x))),'actual_carrier_snr_db':None if ni is None else float(20*np.log10(rms(caches[sid])/rms(gain*caches[ni])))};recipes.append(rec)
                if ni is not None:ps=h.seed(row['primary_pair_ids'][0]);seedmatrix.append(dict(condition=condition,sample_id=sid,pair_id=row['primary_pair_ids'][0],seed=ps,arms={a:ps for a in h.ARMS}))
        frozen={'rows':list(cleanrows.values()),'pairs':manifest['pairs'],'recipes':recipes,'source_sha256':locks,'seed_matrix':seedmatrix,'helper_module_sha256':sha(FOLLOW),'waveform_dtype':'float64_then_hashed_float32','snr_definition':cfg['snr_definition']};dump(out/'recipe_frozen.json',frozen);d.update(seed_matrix=seedmatrix,source_sha256=locks,attribution=manifest['provenance'],phases={'retrieval':'completed','baseline':'running','enhancement':'not_started'})
        def classifier():
            deadline();ex=Wav2Vec2FeatureExtractor.from_pretrained(str(fm.MODEL),local_files_only=True);mo=Wav2Vec2ForCTC.from_pretrained(str(fm.MODEL),local_files_only=True).to('cuda:0').eval();vo=json.loads((fm.MODEL/'vocab.json').read_text());assert vo['<pad>']==0 and mo.config.pad_token_id==0;return ex,mo,vo
        ex,recognizer,vocab=classifier();baselines={};summaries={};gate=False
        for condition in ['padded_clean']+CONDITIONS:
            if condition!='padded_clean' and not gate:break
            rs=[]
            for rec in [r for r in recipes if r['condition']==condition]:
                deadline();sid=rec['sample_id'];x=fm.recipe_wave(rec,caches);s=h.score(x,cleanrows[sid],ex,recognizer,vocab,scorer,'cuda:0',out/'ctc_arrays'/f'{condition}__{sid}__baseline.npz');s.update(condition=condition,arm='baseline');baselines[(condition,sid)]=s;scores.append(s);rs.append(s);d['recognizer_forwards']+=1
            summaries[condition]={'noisy' if condition!='padded_clean' else 'clean':fm.summarize(rs,manifest['pairs'])}
            if condition=='padded_clean':gate=fm.eligible(summaries[condition]['clean'],pair_gate=False);dump(out/'clean_gate.json',{'passed':gate,'summary':summaries[condition]['clean'],'failure_action':'stop_all_no_remap_no_filter'})
            dump(out/'per_sample.json',scores);h.budget()
        del recognizer;gc.collect();torch.cuda.empty_cache();d['clean_gate_passed']=gate;d['summaries']=summaries
        if not gate:
            d.update(status='stopped_clean_gate_failed',phases={'retrieval':'completed','baseline':'stopped_clean_gate_failed','enhancement':'not_executed'})
        else:
            d['noisy_input_eligibility']={c:fm.eligible(summaries[c]['noisy']) for c in CONDITIONS};d['phases']['baseline']='completed';selected=[r for r in recipes if r['condition']!='padded_clean']
            for rec in selected:
                sid=rec['sample_id'];row=cleanrows[sid]
                for arm,x in [('clean',caches[sid].astype(np.float32)),('noisy',fm.recipe_wave(rec,caches))]:audio[(rec['condition'],sid,arm)]=h.save_audio(out/'audio'/rec['condition']/sid/arm,x);audio[(rec['condition'],sid,arm)]['quiet_padding']=h.quiet(x,row)
            import joint_enhance_by_dir as mod
            deadline();model=mod.JointSemambaSgmseModel.load_from_checkpoint(str(h.CKPT),map_location='cpu',strict=True,data_module_cls=mod.JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False);model.eval(no_ema=False).to('cuda:0')
            flags={'train_type':model.train_type,'backbone':model.backbone,'include_noisy_main':model.mode2_include_noisy_main,'cross_attn':model.dnn.cross_attn,'film_modulate':model.dnn.film_modulate,'selector_type':model.dnn.selector_type};assert flags=={'train_type':'mode2','backbone':'ncsnpp_v2','include_noisy_main':True,'cross_attn':False,'film_modulate':False,'selector_type':'none'} and not model.residual_refine_enable and str(model.hparams.frontend)=='storm';d['runtime_flags']=flags
            with torch.inference_mode():
                for rec,sm in zip(selected,seedmatrix):
                    deadline();sid=rec['sample_id'];y=torch.from_numpy(fm.recipe_wave(rec,caches).copy()).to('cuda:0')[None,None,:];torch.manual_seed(sm['seed']);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();norm=y.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12);pre=model._frontend_forward(y/norm)*norm;post=mod._match_length(pre,y);torch.cuda.synchronize();assert torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state()) and post.shape==y.shape;d['frontend_forwards']+=1
                    x=post.squeeze().cpu().numpy().astype(np.float32);key=(rec['condition'],sid,'y_pre');audio[key]=h.save_audio(out/'audio'/rec['condition']/sid/'y_pre',x);audio[key].update(quiet_padding=h.quiet(x,cleanrows[sid]),seed=sm['seed'],frontend_rng_unchanged=True)
                    for arm in ['off','router']:
                        deadline();torch.manual_seed(sm['seed']);enh,nfe,rtf=model.enhance(post.squeeze(),y_cond=y.squeeze(),**common,rrsd=arm=='router',timeit=True);torch.cuda.synchronize();x=np.asarray(enh,dtype=np.float32);d['backend_forwards']+=1;key=(rec['condition'],sid,arm);audio[key]=h.save_audio(out/'audio'/rec['condition']/sid/arm,x);audio[key].update(quiet_padding=h.quiet(x,cleanrows[sid]),seed=sm['seed'],nfe=int(nfe),model_rtf=float(rtf));h.budget()
                    dump(out/'audio_manifest.json',[dict(condition=k[0],sample_id=k[1],arm=k[2],**v) for k,v in audio.items()]);print(f'{rec["condition"]}/{sid}: enhancement done',flush=True)
            del model,pre,post,y;gc.collect();torch.cuda.empty_cache();ex,recognizer,vocab=classifier()
            for rec in selected:
                for arm in h.ARMS:
                    deadline();sid=rec['sample_id'];a=audio[(rec['condition'],sid,arm)];assert sha(HERE/a['raw_path'])==a['raw_sha256'];x=np.load(HERE/a['raw_path'],allow_pickle=False);assert ah(x)==a['float32_sha256'];s=h.score(x,cleanrows[sid],ex,recognizer,vocab,scorer,'cuda:0',out/'ctc_arrays'/f'{rec["condition"]}__{sid}__{arm}.npz');s.update(condition=rec['condition'],arm=arm);scores.append(s);d['recognizer_forwards']+=1;dump(out/'per_sample.json',scores);h.budget()
            transitions={};instability={}
            for c in CONDITIONS:
                transitions[c]={}
                for arm in h.ARMS:
                    rs=[s for s in scores if s['condition']==c and s['arm']==arm];summary=fm.summarize(rs,manifest['pairs']);summary['binary_correct_but_24way_wrong']=[s['sample_id'] for s in rs if s['binary_correct'] and not s['top1_correct']];summaries[c][arm]=summary;transitions[c][arm]={}
                    for refarm in ['noisy','y_pre']:
                        before={sid:baselines[(c,sid)] for sid in cleanrows} if refarm=='noisy' else {s['sample_id']:s for s in scores if s['condition']==c and s['arm']=='y_pre'}
                        transitions[c][arm][refarm]={metric:{'correct_to_wrong':sum(before[s['sample_id']][metric] and not s[metric] for s in rs),'wrong_to_correct':sum(not before[s['sample_id']][metric] and s[metric] for s in rs)} for metric in ['top1_correct','binary_correct']}
            for snr in [-6,6]:
                instability[str(snr)]={}
                for arm in ['baseline']+h.ARMS:
                    a={s['sample_id']:s for s in scores if s['condition']==f'N1_snr{snr:+d}' and s['arm']==arm};b={s['sample_id']:s for s in scores if s['condition']==f'N2_snr{snr:+d}' and s['arm']==arm};instability[str(snr)][arm]={'n':16,'prediction_changed':sum(a[sid]['top1_prediction']!=b[sid]['top1_prediction'] for sid in a),'both_correct':sum(a[sid]['top1_correct'] and b[sid]['top1_correct'] for sid in a),'stable_incorrect':sum(a[sid]['top1_prediction'] is not None and a[sid]['top1_prediction']==b[sid]['top1_prediction'] and not a[sid]['top1_correct'] for sid in a)}
            assert d['frontend_forwards']==64 and d['backend_forwards']==128 and d['recognizer_forwards']==272
            originals={}
            for name in ['p8_q_vcv_enhance_follow_20261018','p8_q_vcv_lowsnr_challenge_20261018']:
                original=json.loads((ROOT/'experiments'/name/'metrics.json').read_text());originals[name]={'status':original['status'],'summaries':original.get('summaries',{}),'comparison':'descriptive_different_speakers_no_significance_or_CIs'}
            d.update(status='completed',summaries=summaries,transitions=transitions,noise_instability=instability,original4speaker_descriptive=originals,phases={'retrieval':'completed','baseline':'completed','enhancement':'completed'})
        assert all(sha(p)==digest for p,digest in locks.items()),'Readonly source changed';d.update(source_checkpoint_unchanged=True,wall_seconds=time.monotonic()-t0,artifact_bytes=h.budget());dump(out/'COMPLETED.json',{'status':d['status']});(out/'RUNNING.json').unlink()
    except BaseException as exc:
        d.update(status='failed',error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-t0);dump(out/'FAILED.json',d);(out/'RUNNING.json').unlink(missing_ok=True);traceback.print_exc()
    dump(HERE/'metrics.json',d);report=f'# Fresh-speaker VCV replication\n\nStatus: {d["status"]}. Classifier forwards: {d["recognizer_forwards"]}.\n\n{LIMITS}\n\n'
    if d['status']=='completed':
        report+='|Condition|Arm|24-way|Binary|Both correct binary pairs|\n|---|---|---:|---:|---:|\n'
        for c,arms in d['summaries'].items():
            for arm,s in arms.items():report+=f'|{c}|{arm}|{s["top1_correct"]}/16|{s["binary_correct"]}/16|{s["both_correct_pairs_binary"]}/8|\n'
    report+='\nThe Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. Frozen recipes/source hashes and exact pair seeds are in outputs/recipe_frozen.json. Raw float32 outputs are primary; PCM16 copies are for optional listening. No old experiment was rerun.\n';(HERE/'experiment_report.md').write_text(report)
    if d['status']=='failed':raise SystemExit(1)
if __name__=='__main__':main()
