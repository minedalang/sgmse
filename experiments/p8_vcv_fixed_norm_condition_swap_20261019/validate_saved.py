"""Read-only validation of saved experiment artifacts; no model forwards."""
from pathlib import Path
import datetime, hashlib, importlib.util, io, json, platform, shutil, subprocess, sys
import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ah(x): return hashlib.sha256(x.tobytes()).hexdigest()
def load(p, name):
    s = importlib.util.spec_from_file_location(name, p)
    m = importlib.util.module_from_spec(s); s.loader.exec_module(m); return m
def dump(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

d = json.loads((HERE/'metrics.json').read_text())
cfg = json.loads((HERE/'config.yaml').read_text())
assert d['status'] == 'completed'
assert (d['frontend_forwards'],d['backend_forwards'],d['recognizer_forwards']) == (8,16,8)
assert all(sha(p)==h for p,h in d['sha256_locks'].items())
audio_count = 0
for c in d['cases']+d['frontend_replays']:
    raw=np.load(HERE/c['raw_path'],allow_pickle=False)
    pcm,sr=sf.read(HERE/c['wav_path'],dtype='int16')
    assert raw.dtype == np.float32 and raw.shape==(32000,) and np.isfinite(raw).all()
    assert sr==16000 and pcm.shape==raw.shape
    assert sha(HERE/c['raw_path'])==c['raw_sha256'] and sha(HERE/c['wav_path'])==c['wav_sha256']
    assert ah(raw)==c.get('float32_sha256',c.get('raw_float32_sha256'))
    encoded=io.BytesIO()
    sf.write(encoded,np.clip(raw,-1,1),16000,subtype='PCM_16',format='WAV')
    assert hashlib.sha256(encoded.getvalue()).hexdigest()==c['wav_sha256']
    audio_count+=1

ctc=[]
for c in d['cases']:
    s=c['ctc']['score']
    if c['cross']: base=HERE
    else: base=ROOT/next(x['experiment'] for x in cfg['sources'] if c['anchor_id'].startswith(x['speaker']))
    p=base/s['array_path']; assert sha(p)==s['array_sha256']
    with np.load(p,allow_pickle=False) as z:
        logits=torch.from_numpy(z['logits'].copy()); labels=z['candidate_labels'].tolist()
        targets=torch.from_numpy(z['candidate_token_ids'].copy()); n=len(logits)
        actual=-F.ctc_loss(logits.log_softmax(-1)[:,None,:].expand(-1,24,-1).contiguous(),targets,
            torch.full((24,),n,dtype=torch.long),torch.full((24,),3,dtype=torch.long),blank=0,reduction='none').numpy()
        ref=np.asarray([s['all24_loglikelihoods'][x] for x in labels],dtype=np.float32)
        err=float(np.max(np.abs(actual-ref)))
        assert np.allclose(actual,ref,rtol=0,atol=1e-4),(c['case_id'],err)
        assert np.array_equal(z['candidate_loglikelihoods'],ref)
        assert np.allclose(logits.softmax(-1).numpy(),z['probabilities'],atol=2e-6,rtol=0)
        q=float(ref[labels.index('g')]-ref[labels.index('k')])
        assert abs(q-c['ctc']['loglikelihood_ratio_g_minus_k'])<1e-5
        ctc.append(dict(case_id=c['case_id'],cross=c['cross'],candidate_scores_checked=24,max_abs_error=err))

fm=load(ROOT/cfg['recipe_helper'],'validate_recipe')
old=json.loads((ROOT/cfg['original_recipe']).read_text()); fresh=json.loads((ROOT/cfg['fresh_recipe']).read_text())
noises={r['noise_id']:np.load(fm.HERE/r['array_path'],allow_pickle=False)['waveform'] for r in old['noise']}
waves={}; clean={}; recipes={}
for src in cfg['sources']:
    base=ROOT/src['experiment']; frozen=old if src['speaker']=='f6' else fresh
    for phone in ['g','k']:
        sid=src['speaker']+'a'+phone+'a1'
        row=next(r for r in frozen.get('clean',frozen.get('rows')) if r['sample_id']==sid)
        clean[sid]=np.load(fm.HERE/row['array_path'],allow_pickle=False)['waveform'] if src['speaker']=='f6' else np.load(base/'outputs'/(sid+'_clean.npy'),allow_pickle=False)
        for cond in cfg['conditions']:
            rec=next(r for r in frozen['recipes'] if r['sample_id']==sid and r['condition']==cond)
            recipes[cond,sid]=rec; cache=dict(noises);cache[sid]=clean[sid]
            waves[cond,sid]=fm.recipe_wave(rec,cache)
rows=[]
for native in [c for c in d['cases'] if not c['cross']]:
    cross=next(c for c in d['cases'] if c['cross'] and c['anchor_id']==native['anchor_id'] and c['condition']==native['condition'])
    cond,sid=native['condition'],native['anchor_id']; donor=cross['noisy_id']; ref=native['refscale']
    assert ref==float(np.abs(waves[cond,sid]).max())
    for key in ['normalized_anchor_sha256','cpu_rng_sha256','cuda_rng_sha256','initial_xT_sha256','regeneration_strength','refscale','seed']:
        assert native[key]==cross[key]
    assert native['nfe']==cross['nfe']==native['trace_steps']==cross['trace_steps']==30
    r1,r2=recipes[cond,sid],recipes[cond,donor]
    assert (r1['noise_id'],r1['noise_gain'])==(r2['noise_id'],r2['noise_gain'])
    noise=r1['noise_gain']*noises[r1['noise_id']]/ref
    guard=next(r for r in d['row_guards'] if r['condition']==cond and r['anchor_id']==sid)
    assert ah(noise)==guard['normalized_physical_noise_sha256']
    err=(waves[cond,sid].astype(np.float64)-waves[cond,donor].astype(np.float64)-clean[sid]+clean[donor])/ref
    assert np.abs(err).max()<=guard['normalized_error_bound']
    qn=native['ctc']['loglikelihood_ratio_g_minus_k']; qc=cross['ctc']['loglikelihood_ratio_g_minus_k']
    qg,qk=(qn,qc) if native['anchor_label']=='g' else (qc,qn)
    pred=cross['ctc']['score']['top1_prediction']
    rows.append(dict(condition=cond,anchor_id=sid,refscale=ref,native_top1=native['ctc']['score']['top1_prediction'],cross_top1=pred,
        cross_top1_relation='follows anchor' if pred==native['anchor_label'] else 'follows noisy' if pred==cross['noisy_label'] else 'third class',
        native_binary=native['ctc']['binary_prediction_gk'],cross_binary=cross['ctc']['binary_prediction_gk'],
        q_native=qn,q_cross=qc,q_noisy_g=qg,q_noisy_k=qk,q_noisy_g_minus_q_noisy_k=qg-qk,
        donor_normalized_peak=cross['condition_normalized_peak'],outside_native_norm_range=cross['outside_native_norm_range']))
matrix=[]
for sid in sorted({r['anchor_id'] for r in rows}):
    rr=[next(r for r in rows if r['anchor_id']==sid and r['condition']==cond) for cond in cfg['conditions']]
    matrix.append(dict(anchor_id=sid,N1=rr[0],N2=rr[1],native_q_N2_minus_N1=rr[1]['q_native']-rr[0]['q_native'],cross_q_N2_minus_N1=rr[1]['q_cross']-rr[0]['q_cross'],native_top1_changes=rr[0]['native_top1']!=rr[1]['native_top1'],cross_top1_changes=rr[0]['cross_top1']!=rr[1]['cross_top1']))
result=dict(status='passed',audio_pairs_validated=audio_count,ctc_cases_validated=len(ctc),cross_ctc_scores_validated=8*24,native_ctc_scores_validated=8*24,ctc_checks=ctc,source_locks_validated=len(d['sha256_locks']),row_geometry_noise_checks=8,rows=rows,noise_instability_matrix=matrix,no_new_model_forwards=True)
dump(HERE/'outputs/independent_validation.json',result)
environment=dict(executable=sys.executable,python=sys.version,platform=platform.platform(),torch=torch.__version__,cuda=torch.version.cuda,numpy=np.__version__,soundfile=sf.__version__,ninja=shutil.which('ninja'),device=torch.cuda.get_device_name(0),git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),git_status=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),driver_started_unix=(HERE/'outputs/COMPLETED.json').stat().st_mtime-d['wall_seconds'],wall_seconds=d['wall_seconds'],command='PATH=/data/coding/sgmse/env/bin:$PATH /data/coding/sgmse/env/bin/python -u experiments/p8_vcv_fixed_norm_condition_swap_20261019/run_swap.py')
dump(HERE/'outputs/execution_environment.json',environment)
print(json.dumps({k:result[k] for k in ['status','audio_pairs_validated','ctc_cases_validated','cross_ctc_scores_validated','native_ctc_scores_validated','source_locks_validated','row_geometry_noise_checks']}))
