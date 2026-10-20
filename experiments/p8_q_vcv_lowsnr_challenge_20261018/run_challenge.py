#!/usr/bin/env python3
"""Independent frozen low-SNR VCV challenge; prior input gate remains failed. Run once by worker."""
from pathlib import Path
import gc, hashlib, html, importlib.util, json, os, subprocess, sys, time, traceback
HERE=Path(__file__).resolve().parent
ROOT=HERE.parent.parent
PREV=ROOT/'experiments/p8_q_vcv_fixed_noise_20261018'
CKPT=ROOT/'joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt'
CKSHA='ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9'
CONDITIONS=['N1_snr-6','N2_snr-6']
ARMS=['y_pre','off','router']
CAP=100_000_000
LIMITS='Independent low-SNR challenge using externally known nominal corpus phoneme identities. Both selected input conditions failed the prior eligibility gate and remain unqualified; the completed +6dB protocol and results are immutable. This challenge is not evidence of ignored noisy identity cues, input identifiability or human recoverability. Recognition failure does not prove that identity evidence is absent. Corpus VCV speech is out of distribution; this is not natural DNS generalization. Nominal identities are corpus labels, not model truth. Padding energy is not a word-hallucination metric. No WER, DNSMOS or population claims.'
def require(c,m):
    if not c: raise ValueError(m)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def ah(x):return hashlib.sha256(x.tobytes(order='C')).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def loadmod(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def seed(pair):return (2026+int.from_bytes(hashlib.blake2b(pair.encode(),digest_size=4).digest(),'little'))%2**32
def budget():
    n=sum(p.stat().st_size for p in HERE.rglob('*') if p.is_file() and not p.is_symlink());require(n<=CAP,'100MB artifact cap exceeded');return n
def quiet(x,row):
    import numpy as np
    def stats(a):return {'rms':float(np.sqrt(np.mean(np.square(a,dtype=np.float64)))),'mean_square':float(np.mean(np.square(a,dtype=np.float64))),'samples':len(a)}
    end=8000+row['resampled_samples']
    require(end<=20800,'True source extends into fixed post-pad; no adapting intervals')
    return {'pre_0_0p3':stats(x[:4800]),'post_1p3_2p0':stats(x[20800:]),'outside_true_source_support':stats(np.concatenate([x[:8000],x[end:]])),'source_onset_sample':8000,'source_end_sample_exclusive':end}
def save_audio(folder,x):
    import numpy as np
    import soundfile as sf
    x=np.asarray(x,dtype=np.float32);require(x.shape==(32000,) and np.isfinite(x).all(),'Invalid raw audio')
    folder.mkdir(parents=True,exist_ok=False)
    np.save(folder/'output_raw.npy',x,allow_pickle=False)
    sf.write(folder/'listening.wav',np.clip(x,-1,1),16000,subtype='PCM_16')
    return {'raw_path':str((folder/'output_raw.npy').relative_to(HERE)),'raw_sha256':sha(folder/'output_raw.npy'),'float32_sha256':ah(x),'wav_path':str((folder/'listening.wav').relative_to(HERE)),'wav_sha256':sha(folder/'listening.wav'),'raw_peak':float(np.max(np.abs(x))),'pcm_clipping_count':int(np.count_nonzero((x < -1)|(x >= 1))),'sample_rate':16000,'samples':32000,'score_source':'unclipped_raw_float32'}
def score(x,row,extractor,model,vocab,mod,device,artifact):
    import numpy as np
    import torch
    import torch.nn.functional as F
    labels=list(mod.CONSONANTS);left,right=mod.VOWELS[row['vowel_left']],mod.VOWELS[row['vowel_right']]
    inputs={k:v.to(device) for k,v in extractor(x,sampling_rate=16000,return_tensors='pt',padding=False).items()}
    targets=torch.tensor([[vocab[left],vocab[mod.CONSONANTS[c]],vocab[right]] for c in labels],dtype=torch.long)
    with torch.inference_mode():
        logits=model(**inputs).logits[0].float().cpu();require(logits.shape[1]==len(vocab),'Vocabulary shape mismatch')
        logp=logits.log_softmax(-1)
        scores=-F.ctc_loss(logp[:,None,:].expand(-1,24,-1).contiguous(),targets,torch.full((24,),len(logits),dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none',zero_infinity=False).numpy()
        probs=logp.exp().numpy()
    require(np.isfinite(scores).all(),'Nonfinite CTC scores')
    ranking=sorted(zip(labels,map(float,scores)),key=lambda z:(-z[1],labels.index(z[0])))
    winners=[c for c,s in ranking if s==ranking[0][1]];pred=winners[0] if len(winners)==1 else None
    truth=row['consonant'];rival=mod.COMPETITOR[truth];margin=float(scores[labels.index(truth)]-scores[labels.index(rival)])
    greedy=logits.argmax(-1).tolist();runs=[];i=0;id2token={v:k for k,v in vocab.items()}
    while i<len(greedy):
        j=i+1
        while j<len(greedy) and greedy[j]==greedy[i]:j+=1
        if greedy[i]!=0:runs.append({'token':id2token[greedy[i]],'token_id':greedy[i],'start_frame':i,'end_frame_exclusive':j,'mean_posterior':float(probs[i:j,greedy[i]].mean())})
        i=j
    np.savez_compressed(artifact,logits=logits.numpy().astype(np.float32),probabilities=probs.astype(np.float32),candidate_loglikelihoods=scores.astype(np.float32),candidate_labels=np.asarray(labels),candidate_token_ids=targets.numpy())
    return {'sample_id':row['sample_id'],'speaker':row['speaker'],'consonant':truth,'contrast':'k/g' if truth in ('k','g') else 't/d','top1_prediction':pred,'top1_correct':pred==truth,'top1_tied_labels':winners if pred is None else [],'binary_prediction':truth if margin>0 else rival if margin<0 else None,'binary_correct':margin>0,'binary_tie':margin==0,'binary_competitor':rival,'binary_loglikelihood_margin':margin,'top1_loglikelihood_margin':ranking[0][1]-ranking[1][1],'all24_loglikelihoods':dict(zip(labels,map(float,scores))),'all24_ranking':ranking,'all24_candidate_ipa_tokens':{c:[left,mod.CONSONANTS[c],right] for c in labels},'greedy_ipa':' '.join(r['token'] for r in runs),'greedy_runs_descriptive':runs,'ctc_frames':len(logits),'array_path':str(artifact.relative_to(HERE)),'array_sha256':sha(artifact)}
def page(records,rows):
    sections=[]
    for condition in CONDITIONS:
        for sid,row in rows.items():
            buttons=[]
            for arm in ['clean','noisy']+ARMS:
                r=records[(condition,sid,arm)];src=html.escape(r['wav_path'],quote=True);digest=r['wav_sha256']
                buttons.append(f'<button data-src="{src}" data-info="{condition} · {sid} · {arm} · 0–2s · SHA256 {digest}">{arm}</button>')
            sections.append(f'<section><h2>{condition} · {sid}</h2><details><summary>显示名义标签</summary>VCV /æ {html.escape(row["consonant"])} æ/；不是英语单词转写</details>'+''.join(buttons)+'</section>')
    (HERE/'listening.html').write_text('''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VCV paired listening</title><style>body{font:18px/1.6 system-ui;max-width:1100px;margin:25px auto;padding:15px}section{border:1px solid #aaa;padding:12px;margin:15px 0}button{font:inherit;margin:5px}audio{width:100%}#identity{overflow-wrap:anywhere}</style><h1>VCV 成对试听</h1><p>完整2秒；固定播放增益1，不做音量归一化。clean/noisy是参考，其他三项为增强输出。这是独立低信噪比挑战；输入未通过之前的识别器门槛，不代表输入不可辨识或人类可恢复。每次仅一个播放器。可明天试听，无需现在反馈。</p><button id="stop">停止</button><p id="identity" role="status">当前播放：无</p><audio id="player" controls preload="none"></audio>'''+''.join(sections)+'''<script>'use strict';const player=document.getElementById('player'),identity=document.getElementById('identity');let generation=0,current=null;function stop(){generation++;player.pause();current=null;identity.textContent='当前播放：无';}async function select(b){stop();const token=generation,info=b.dataset.info;current={token,info};identity.textContent='正在加载：'+info;player.src=b.dataset.src;player.volume=1;player.playbackRate=1;player.load();try{await player.play();if(token!==generation)return;identity.textContent='当前播放：'+info;}catch(e){if(token===generation){current=null;identity.textContent='播放失败：'+info+' · '+e.message;}}}for(const b of document.querySelectorAll('button[data-src]'))b.addEventListener('click',()=>select(b));document.getElementById('stop').addEventListener('click',stop);player.addEventListener('play',()=>{if(current&&current.token===generation)identity.textContent='当前播放：'+current.info;});player.addEventListener('ended',()=>{if(current&&current.token===generation)identity.textContent='播放结束：'+current.info;});window.addEventListener('pagehide',stop);</script></html>''')
def main():
    import yaml
    cfg=yaml.safe_load((HERE/'config.yaml').read_text())
    for k in ['training','downloads','dnsmos']:require(type(cfg[k]) is bool and cfg[k] is False,'Strict boolean schema')
    require(type(cfg['status']) is str and cfg['status']=='prepared','Strict status enum')
    require(cfg['conditions']==CONDITIONS and cfg['arms']==ARMS and cfg['N']==30,'Frozen design mismatch')
    out=HERE/'outputs';out.mkdir(exist_ok=False);(out/'ctc_arrays').mkdir();dump(out/'RUNNING.json',{'started_at':time.time()})
    t0=time.monotonic();d={'experiment_id':HERE.name,'status':'running','limits':LIMITS,'frontend_forwards':0,'backend_forwards':0,'recognizer_forwards':0};audio={};scores=[]
    try:
        import numpy as np
        import torch
        os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
        sys.path.insert(0,str(ROOT));os.chdir(ROOT)
        fm=loadmod(PREV/'fixed_noise.py','fixed_noise_frozen')
        frozen=json.loads((PREV/'inputs/recipe_frozen.json').read_text());pm=json.loads((PREV/'recognition_outputs/metrics.json').read_text())
        require(pm['status']=='completed' and pm['recognizer_forwards']==144,'Prior experiment incomplete')
        require(pm['padded_clean_gate_passed'] and sorted(pm['eligible_for_later_pilot_conditions'])==['N1_snr+6','N2_snr+6'],'Prior qualified stage changed')
        require(all(c not in pm['eligible_for_later_pilot_conditions'] and not fm.eligible(pm['conditions'][c]) for c in CONDITIONS),'Challenge must retain failed input eligibility')
        require(all(sha(ROOT/p)==h for p,h in cfg['source_sha256_locks'].items()),'Frozen source SHA lock mismatch')
        require(sha(PREV/'fixed_noise.py')==frozen['runner_sha256'] and sha(fm.SCORER)==frozen['prior_scoring_module_sha256'],'Frozen scoring modules changed')
        require(sha(PREV/'inputs/recipe_frozen.json')==pm['recipe_sha256'],'Recipe changed')
        readonly={str(p):sha(p) for p in [PREV/'fixed_noise.py',PREV/'inputs/recipe_frozen.json',PREV/'recognition_outputs/metrics.json',PREV/'recognition_outputs/per_sample.json',fm.SCORER,fm.MANIFEST,CKPT,HERE/'config.yaml',Path(__file__),ROOT/'experiments/p8_10720_snr_control_20261018/metrics.json',ROOT/'experiments/p8_noisy_condition_path_audit_20261018/metrics.json']}
        readonly.update({str(ROOT/p):h for p,h in cfg['source_sha256_locks'].items()})
        for rec in frozen['clean']:readonly[rec['wav_path']]=rec['sha256']
        for rec in frozen['noise']:readonly[rec['source_path']]=rec['source_sha256']
        for rel,digest in frozen['model_files_sha256'].items():readonly[str(fm.MODEL/rel)]=digest
        codefiles=['joint_enhance_by_dir.py','sgmse/joint_model.py','sgmse/joint_data_module.py','sgmse/model.py','sgmse/frontends/storm.py']
        prior_control=json.loads((ROOT/'experiments/p8_10720_snr_control_20261018/metrics.json').read_text());common=prior_control['enhance_kwargs']
        require(common['N']==30 and common['snr']==.5 and common['sampler_type']=='pc' and common['predictor']=='reverse_diffusion' and common['corrector']=='none','Original sampler mismatch')
        for rel in codefiles:
            digest=sha(ROOT/rel);require(digest==prior_control['code_sha256'][rel],'Core code changed since SNR control');readonly[str(ROOT/rel)]=digest
        require(sha(CKPT)==CKSHA,'80k checkpoint mismatch')
        require(all(sha(p)==h for p,h in readonly.items()),'Readonly source hash mismatch')
        caches={}
        for rec in frozen['clean']+frozen['noise']:
            p=PREV/rec['array_path'];require(sha(p)==rec['array_sha256'],'Cached array changed');readonly[str(p)]=rec['array_sha256'];x=np.load(p,allow_pickle=False)['waveform'];require(ah(x)==rec['float64_sha256'],'Waveform changed');caches[rec.get('sample_id',rec.get('noise_id'))]=x
        rows={r['sample_id']:r for r in frozen['clean']};require(len(rows)==16 and len({r['speaker'] for r in rows.values()})==4,'Retain all16 across4speakers')
        selected=[r for r in frozen['recipes'] if r['condition'] in CONDITIONS];require(len(selected)==32,'Need32 inputs')
        seedmatrix=[]
        for rec in selected:
            row=rows[rec['sample_id']];require(len(row['primary_pair_ids'])==1,'Ambiguous pair');ps=seed(row['primary_pair_ids'][0]);seedmatrix.append(dict(condition=rec['condition'],sample_id=rec['sample_id'],pair_id=row['primary_pair_ids'][0],seed=ps,arms={a:ps for a in ARMS}))
        d.update(readonly_sha256=readonly,git_head=subprocess.getoutput('git rev-parse HEAD'),git_status=subprocess.getoutput('git status --short'),seed_matrix=seedmatrix,enhance_kwargs=common,attribution={'corpus':'The Consonant Challenge Corpus','doi':'10.5281/zenodo.3538223','license':'CC-BY-4.0','source_manifest':str(fm.MANIFEST)},source_input_group=frozen['source_input_group'])
        d.update(stage='independent_low_snr_challenge',prior_input_gates={c:{'eligible':False,'summary':pm['conditions'][c]} for c in CONDITIONS})
        dump(out/'provenance_frozen.json',d)
        def deadline():require(time.monotonic()-t0<=600,'10 minute budget reached before next task; stop without killing an active forward')
        # Save exact raw reference waveforms; no baseline recognizer forward.
        for rec in selected:
            row=rows[rec['sample_id']];x=fm.recipe_wave(rec,caches);require(ah(x)==rec['model_input_float32_sha256'] and np.max(np.abs(x))<1,'Input hash/peak mismatch')
            for arm,v in [('clean',caches[rec['sample_id']].astype(np.float32)),('noisy',x)]:
                key=(rec['condition'],rec['sample_id'],arm);audio[key]=save_audio(out/'audio'/rec['condition']/rec['sample_id']/arm,v);audio[key]['quiet_padding']=quiet(v,row)
        import joint_enhance_by_dir as mod
        deadline();model=mod.JointSemambaSgmseModel.load_from_checkpoint(str(CKPT),map_location='cpu',strict=True,data_module_cls=mod.JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False)
        model.eval(no_ema=False).to('cuda:0')
        flags={'train_type':model.train_type,'backbone':model.backbone,'frontend':str(getattr(model.hparams,'frontend',None)),'include_noisy_main':model.mode2_include_noisy_main,'cross_attn':model.dnn.cross_attn,'film_modulate':model.dnn.film_modulate,'residual_refine_enable':model.residual_refine_enable,'selector_type':model.dnn.selector_type}
        require(flags=={'train_type':'mode2','backbone':'ncsnpp_v2','frontend':'storm','include_noisy_main':True,'cross_attn':False,'film_modulate':False,'residual_refine_enable':False,'selector_type':'none'},'Runtime flags differ from noisy condition path audit');d['runtime_flags']=flags
        with torch.inference_mode():
            for rec,sm in zip(selected,seedmatrix):
                deadline();sid=rec['sample_id'];row=rows[sid];x=fm.recipe_wave(rec,caches);y=torch.from_numpy(x.copy()).to('cuda:0')[None,None,:];torch.manual_seed(sm['seed']);cpu=torch.get_rng_state().clone();cuda=torch.cuda.get_rng_state().clone();norm=y.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)
                tt=time.monotonic();pre=model._frontend_forward(y/norm)*norm;post=mod._match_length(pre,y);torch.cuda.synchronize();require(torch.equal(cpu,torch.get_rng_state()) and torch.equal(cuda,torch.cuda.get_rng_state()),'Frontend consumes RNG');require(post.shape==y.shape,'Frontend length mismatch');d['frontend_forwards']+=1
                key=(rec['condition'],sid,'y_pre');v=post.squeeze().cpu().numpy().astype(np.float32);audio[key]=save_audio(out/'audio'/rec['condition']/sid/'y_pre',v);audio[key].update(quiet_padding=quiet(v,row),seed=sm['seed'],frontend_rng_unchanged=True,wall_seconds=time.monotonic()-tt)
                for arm in ['off','router']:
                    deadline();torch.manual_seed(sm['seed']);tt=time.monotonic();enh,nfe,rtf=model.enhance(post.squeeze(),y_cond=y.squeeze(),**common,rrsd=arm=='router',timeit=True);torch.cuda.synchronize();d['backend_forwards']+=1;v=np.asarray(enh,dtype=np.float32);key=(rec['condition'],sid,arm);audio[key]=save_audio(out/'audio'/rec['condition']/sid/arm,v);audio[key].update(quiet_padding=quiet(v,row),seed=sm['seed'],nfe=int(nfe),model_rtf=float(rtf),wall_seconds=time.monotonic()-tt);budget()
                dump(out/'audio_manifest.json',[dict(condition=k[0],sample_id=k[1],arm=k[2],**v) for k,v in audio.items()]);print(f'{rec["condition"]}/{sid}: frontend+off+router completed',flush=True)
        del model,post,pre,y;gc.collect();torch.cuda.empty_cache()
        from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2ForCTC
        deadline();extractor=Wav2Vec2FeatureExtractor.from_pretrained(str(fm.MODEL),local_files_only=True);recognizer=Wav2Vec2ForCTC.from_pretrained(str(fm.MODEL),local_files_only=True).to('cuda:0').eval();vocab=json.loads((fm.MODEL/'vocab.json').read_text());require(vocab['<pad>']==0 and recognizer.config.pad_token_id==0,'Blank mismatch');scoringmod=fm.prior()
        refs=json.loads((PREV/'recognition_outputs/per_sample.json').read_text());baseline={(r['condition'],r['sample_id']):r for r in refs}
        for rec in selected:
            for arm in ARMS:
                deadline();sid=rec['sample_id'];key=(rec['condition'],sid,arm);a=audio[key];require(sha(HERE/a['raw_path'])==a['raw_sha256'],'Raw changed');x=np.load(HERE/a['raw_path'],allow_pickle=False);require(ah(x)==a['float32_sha256'],'Raw hash mismatch');s=score(x,rows[sid],extractor,recognizer,vocab,scoringmod,'cuda:0',out/'ctc_arrays'/f'{rec["condition"]}__{sid}__{arm}.npz');s.update(condition=rec['condition'],arm=arm,raw_sha256=a['raw_sha256']);scores.append(s);d['recognizer_forwards']+=1;dump(out/'per_sample.json',scores);budget()
        summaries={};transitions={};backend_vs_y_pre={}
        for condition in CONDITIONS:
            summaries[condition]={'clean_reused':fm.summarize([baseline[('padded_clean',sid)] for sid in rows],frozen['pairs']),'noisy_reused':fm.summarize([baseline[(condition,sid)] for sid in rows],frozen['pairs'])}
            transitions[condition]={}
            for arm in ARMS:
                rs=[s for s in scores if s['condition']==condition and s['arm']==arm];summaries[condition][arm]=fm.summarize(rs,frozen['pairs']);tr={}
                for metric in ['top1_correct','binary_correct']:
                    tr[metric]={'correct_to_wrong':sum(baseline[(condition,s['sample_id'])][metric] and not s[metric] for s in rs),'wrong_to_correct':sum(not baseline[(condition,s['sample_id'])][metric] and s[metric] for s in rs),'correct_to_correct':sum(baseline[(condition,s['sample_id'])][metric] and s[metric] for s in rs),'wrong_to_wrong':sum(not baseline[(condition,s['sample_id'])][metric] and not s[metric] for s in rs)}
                before=summaries[condition]['noisy_reused'];after=summaries[condition][arm]
                tr['pair_transitions']=[{'pair_id':a['pair_id'],'before_both_binary_correct':b['both_correct_binary'],'after_both_binary_correct':a['both_correct_binary'],'before_both_24way_correct':b['both_correct_24way'],'after_both_24way_correct':a['both_correct_24way']} for a,b in zip(after['pairs'],before['pairs'])];transitions[condition][arm]=tr
        for condition in CONDITIONS:
            backend_vs_y_pre[condition]={};front={s['sample_id']:s for s in scores if s['condition']==condition and s['arm']=='y_pre'}
            for arm in ['off','router']:
                rs=[s for s in scores if s['condition']==condition and s['arm']==arm];backend_vs_y_pre[condition][arm]={metric:{'correct_to_wrong':sum(front[s['sample_id']][metric] and not s[metric] for s in rs),'wrong_to_correct':sum(not front[s['sample_id']][metric] and s[metric] for s in rs)} for metric in ['top1_correct','binary_correct']}
        for condition in CONDITIONS:
            for arm in ['clean_reused','noisy_reused']+ARMS:
                rs=[baseline[('padded_clean' if arm=='clean_reused' else condition,sid)] for sid in rows] if arm.endswith('_reused') else [s for s in scores if s['condition']==condition and s['arm']==arm]
                summaries[condition][arm]['binary_correct_but_24way_wrong']={'count':sum(s['binary_correct'] and not s['top1_correct'] for s in rs),'sample_ids':[s['sample_id'] for s in rs if s['binary_correct'] and not s['top1_correct']]}
        require(d['frontend_forwards']==32 and d['backend_forwards']==64 and d['recognizer_forwards']==96,'Forward count mismatch')
        require(all(sha(p)==h for p,h in readonly.items()),'Readonly artifacts changed during run')
        for a in audio.values():require(sha(HERE/a['raw_path'])==a['raw_sha256'] and sha(HERE/a['wav_path'])==a['wav_sha256'],'Audio changed before listening page')
        page(audio,rows);dump(out/'audio_manifest.json',[dict(condition=k[0],sample_id=k[1],arm=k[2],**v) for k,v in audio.items()]);d.update(status='completed',summaries=summaries,transitions_from_noisy=transitions,backend_vs_y_pre=backend_vs_y_pre,source_and_checkpoint_unchanged=True,output_bytes=budget(),wall_seconds=time.monotonic()-t0)
        dump(out/'COMPLETED.json',{'status':'completed','wall_seconds':d['wall_seconds']});(out/'RUNNING.json').unlink()
    except BaseException as exc:
        d.update(status='failed',error=str(exc),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-t0);dump(out/'FAILED.json',d);(out/'RUNNING.json').unlink(missing_ok=True);traceback.print_exc()
    dump(HERE/'metrics.json',d)
    report=f'# VCV independent low-SNR challenge\n\nStatus: {d["status"]}. Frontend/backend/recognizer forwards: {d["frontend_forwards"]}/{d["backend_forwards"]}/{d["recognizer_forwards"]}.\n\n{LIMITS}\n\n'
    if d['status']=='completed':
        report+='|Condition|Arm|24-way|Binary|Both correct binary pairs|Binary correct but 24-way wrong|\n|---|---|---:|---:|---:|---:|\n'
        for condition,arms in d['summaries'].items():
            for arm,s in arms.items():report+=f'|{condition}|{arm}|{s["top1_correct"]}/16|{s["binary_correct"]}/16|{s["both_correct_pairs_binary"]}/8|{s["binary_correct_but_24way_wrong"]["count"]}/16|\n'
        report+='\nPrimary recognition uses unclipped raw float32 arrays. PCM16 WAVs are listening copies with clipping counts recorded. Baseline clean/noisy scores are reused without forward passes. Fixed quiet padding RMS and true-support energy are in outputs/audio_manifest.json. Source corpus: The Consonant Challenge Corpus, DOI10.5281/zenodo.3538223, CC-BY-4.0. Optional paired listening page: listening.html. No listening judgment collected.\n'
    else:report+='Failure recorded in outputs/FAILED.json; diagnose before any new attempt.\n'
    (HERE/'experiment_report.md').write_text(report)
    if d['status']!='completed':raise SystemExit(1)
if __name__=='__main__':main()
