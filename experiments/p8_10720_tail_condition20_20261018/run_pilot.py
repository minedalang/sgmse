"""Bounded oracle causal mechanism probe. Execution belongs to experiment_worker."""
from pathlib import Path
import csv, datetime, hashlib, json, os, subprocess, sys, time, traceback
import numpy as np
ROOT = Path('/data/coding/sgmse')
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / 'experiments/p8_10720_snr_control_20261018'
SR, LENGTH, A, B, FADE, SEED, CAP = 16000, 480000, 240800, 252000, 320, 2997368683, 20_000_000
RUNS = [('no_op', SEED, False, False), ('condition_tail_clean', SEED, False, True)]
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20), b''): h.update(block)
    return h.hexdigest()
def ah(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def dump(name, d):
    with (OUT/name).open('x') as f: json.dump(d, f, indent=2, default=str); f.write('\n')
def cap():
    total = sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    assert total < CAP, f'Output cap exceeded: {total}'
    return total

def make_mask(length=LENGTH, a=A, b=B, fade=FADE):
    assert 0 <= a < b <= length and 2*fade <= b-a and fade >= 2
    mask = np.zeros(length, np.float32); mask[a:b] = 1
    ramp = (0.5-0.5*np.cos(np.linspace(0, np.pi, fade))).astype(np.float32)
    mask[a:a+fade] = ramp; mask[b-fade:b] = ramp[::-1]
    return mask

def intervene(original, clean, mask):
    # Assignment only on mask support guarantees bitwise identity elsewhere.
    result = original.clone()
    selected = mask > 0
    result[selected] = original[selected] + mask[selected] * (clean[selected] - original[selected])
    assert result.shape == original.shape and result.dtype == original.dtype
    assert bool((result[~selected] == original[~selected]).all())
    return result

def rms(x): return float(np.sqrt(np.mean(np.asarray(x, np.float64)**2)))
def db(x): return float(20*np.log10(max(float(x), 1e-12)))
def relative_l2(x, ref): return float(np.linalg.norm(np.asarray(x,np.float64)-ref)/max(np.linalg.norm(ref),1e-30))
def highband(x):
    """Mean-square 4–7.5kHz energy via unwindowed one-sided Parseval FFT."""
    x=np.asarray(x,np.float64); spectrum=np.fft.rfft(x); f=np.fft.rfftfreq(len(x),1/SR)
    weight=np.full(len(spectrum),2.); weight[0]=1
    if len(x)%2==0: weight[-1]=1
    return float(np.sum(weight[(f>=4000)&(f<=7500)]*abs(spectrum[(f>=4000)&(f<=7500)])**2)/len(x)**2)
def summary(x, base, clean, active):
    outside=np.ones(LENGTH,bool); outside[A:B]=False
    return {'tail_rms': rms(x[A:B]), 'tail_rms_dbfs': db(rms(x[A:B])), 'tail_highband_mean_square_4k_7p5k': highband(x[A:B]),
            'tail_delta_from_no_op_rms':rms(x[A:B]-base[A:B]), 'outside_window_relative_l2_vs_no_op':relative_l2(x[outside],base[outside]),
            'full_clean_speech_active_relative_l2_vs_no_op':relative_l2(x[active],base[active]),
            'full_clean_speech_active_relative_l2_vs_clean':relative_l2(x[active],clean[active]),
            'full_clean_speech_active_rmse_vs_clean':rms(x[active]-clean[active])}

def synthetic_checks():
    import torch
    m=make_mask(100,20,80,5)
    assert np.array_equal(m,make_mask(100,20,80,5)) and np.count_nonzero(m[:20])==0 and np.count_nonzero(m[80:])==0
    assert m[20]==m[79]==0 and m[24]==m[75]==1 and np.array_equal(m[20:25],m[75:80][::-1])
    x=torch.linspace(-1,1,100); c=torch.zeros(100); mask=torch.from_numpy(m)
    z=intervene(x,c,mask)
    assert torch.equal(z,intervene(x,c,mask)) and torch.equal(x[m==0],z[m==0]) and torch.equal(z[25:75],c[25:75])
    assert torch.equal(intervene(x,x,mask),x)
    sine=np.sin(2*np.pi*5000*np.arange(16000)/SR)
    assert abs(highband(sine)-.5)<1e-12
    return {'status':'passed','inference_executed':False,'checks':['mask deterministic symmetric internal raised cosine','strict unchanged outside support','tensor shape dtype and deterministic interpolation','no-op identity','Parseval highband calibration']}

def main():
    allowed={'run_pilot.py','config.yaml','engineer_validation.json','engineer_handoff.md','__pycache__','run.log'}
    assert not [p.name for p in OUT.iterdir() if p.name not in allowed], 'Existing/partial artifacts: archive and diagnose before any retry'
    dump('run_started.json',{'started_at_utc':utc()})
    d={'experiment_id':OUT.name,'status':'running','started_at_utc':utc(),'failures':[], 'runs':{},
       'limits':'Oracle clean-tail intervention is an OOD mechanism probe, not a deployable method. This is within-case/SNR replication, not independent generalization or a trained method. Prior P8-M 5243 active masking negative is preserved and is a distinct experiment.'}
    t0=time.monotonic()
    try:
        import soundfile as sf
        import torch
        import yaml
        os.chdir(ROOT);sys.path.insert(0,str(ROOT))
        import joint_enhance_by_dir as mod
        cfg=yaml.safe_load((OUT/'config.yaml').read_text())
        assert cfg['source_variant']=='noise_minus20dB' and cfg['output_cap_bytes']==CAP
        assert cfg['N']==30 and cfg['sampling_snr']==.5
        assert cfg['window_samples']==[A,B] and cfg['fade_samples']==FADE and cfg['seed']==SEED and cfg['runs']==[r[0] for r in RUNS]
        source=json.loads((SOURCE/'metrics.json').read_text()); assert source['status']=='completed'
        v=source['variants']['noise_minus20dB']; common=source['enhance_kwargs']
        assert common['N']==30 and common['snr']==.5 and common['corrector']=='none'
        d['enhance_kwargs']={**common,'rrsd':False,'timeit':True}
        d['source_metrics_sha256']=sha(SOURCE/'metrics.json'); d['source_driver_sha256']=sha(SOURCE/'run_control.py')
        d['code_sha256']={p:sha(ROOT/p) for p in source['code_sha256']}
        assert d['code_sha256']==source['code_sha256'],'Code changed since successful control'
        ckpt=Path(source['checkpoint']['path']); assert sha(ckpt)==source['checkpoint']['sha256']==cfg['checkpoint_sha256']
        d['checkpoint']=source['checkpoint']; d['driver_sha256']=sha(__file__); d['config_sha256']=sha(OUT/'config.yaml')
        d['git_head']=subprocess.getoutput('git rev-parse HEAD');d['git_status']=subprocess.getoutput('git status --short')
        d['environment']={'python':sys.version,'executable':sys.executable,'torch':torch.__version__,'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),'device':torch.cuda.get_device_name(0),'PATH':os.environ.get('PATH'),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark}
        paths={k:Path(v['arms'][k]) for k in ('clean','noisy','off','y_pre')}
        expected={'clean':source['original_sources']['clean']['sha256'],'noisy':v['input_sha256'],'off':v['off']['sha256'],'y_pre':v['frontend']['output_sha256']}
        d['inputs']={}
        for k,p in paths.items():
            assert sha(p)==expected[k]
            x,sr=sf.read(p,dtype='float32'); assert sr==SR and x.shape==(LENGTH,) and np.isfinite(x).all()
            d['inputs'][k]={'path':str(p),'sha256':sha(p),'float32_sha256':ah(x),'subtype':sf.info(p).subtype}
        assert sf.info(paths['noisy']).subtype=='FLOAT'
        clean,sr=sf.read(paths['clean'],dtype='float32')
        assert rms(clean[A:B])<=3/32768,'Clean tail no longer near quantization floor'
        d['clean_tail']={'rms':rms(clean[A:B]),'rms_dbfs':db(rms(clean[A:B])),'floor_limit_pcm16_steps':3}
        frames=clean.reshape(-1,320); active_frames=np.sqrt(np.mean(frames.astype(np.float64)**2,axis=1))>10**(-70/20)
        active=np.repeat(active_frames,320); assert active.any()
        d['speech_activity']={'reference':'full clean 16kHz waveform','frame_samples':320,'hop_samples':320,'threshold_dbfs':-70,'comparison':'>','active_frames':int(active_frames.sum()),'total_frames':len(active_frames)}
        dump('launch_provenance.json',d)
        model=mod.JointSemambaSgmseModel.load_from_checkpoint(str(ckpt),map_location='cpu',strict=True,data_module_cls=mod.JointWaveDataModule,base_dir='.',batch_size=1,num_workers=0,dummy=True,weights_only=False)
        model.eval(no_ema=False).to('cuda:0')
        d['loaded_model']={'train_type':model.train_type,'frontend':str(getattr(model.hparams,'frontend',None)),'residual_refine_enable':model.residual_refine_enable}
        assert d['loaded_model']==source['loaded_model']
        raw_baseline=None; pcm_baseline=None
        with torch.inference_mode():
            torch.manual_seed(SEED)
            noisy=mod.load_mono(str(paths['noisy']),SR).to('cuda:0')[None,None,:]
            cpu_rng=torch.get_rng_state().clone();gpu_rng=torch.cuda.get_rng_state().clone()
            norm=noisy.abs().amax(dim=-1,keepdim=True).clamp_min(1e-12)
            pre=model._frontend_forward(noisy/norm)*norm
            frontend=mod._match_length(pre,noisy).squeeze(); noisy=noisy.squeeze()
            torch.cuda.synchronize()
            assert torch.equal(cpu_rng,torch.get_rng_state()) and torch.equal(gpu_rng,torch.cuda.get_rng_state())
            front_np=frontend.cpu().numpy()
            assert ah(front_np)==v['frontend']['float32_sha256'],'Float frontend mismatch: STOP before intervention'
            d['frontend']={'float32_sha256':ah(front_np),'rng_unchanged':True,'recomputed_count':1}
            np.save(OUT/'frontend_raw.npy',front_np,allow_pickle=False)
            mask_np=make_mask(); np.save(OUT/'intervention_mask.npy',mask_np,allow_pickle=False)
            mask=torch.from_numpy(mask_np).to(noisy); clean_t=torch.from_numpy(clean).to(noisy)
            outside=torch.ones(LENGTH,dtype=torch.bool,device=noisy.device);outside[A:B]=False
            with (OUT/'frames.csv').open('x',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=['run','start_sample','end_sample_exclusive','clean_rms','clean_active','raw_rms','raw_highband_mean_square','raw_delta_no_op_rms','raw_clean_error_rms']);writer.writeheader()
                for name,seed,edit_anchor,edit_condition in RUNS:
                    assert not edit_anchor
                    anchor=frontend
                    assert torch.equal(anchor,frontend), 'Original float frontend must be held unchanged globally'
                    condition=intervene(noisy,clean_t,mask) if edit_condition else noisy
                    assert torch.equal(anchor[outside],frontend[outside]) and torch.equal(condition[outside],noisy[outside])
                    assert torch.equal(condition.abs().max(),noisy.abs().max()),'Global normalization peak changed'
                    r={'seed':seed,'anchor_tail_clean':edit_anchor,'condition_tail_clean':edit_condition,'outside_input_max_abs_difference':0.,'normalization_peak':float(condition.abs().max()),'anchor_float32_sha256':ah(anchor.cpu().numpy()),'condition_float32_sha256':ah(condition.cpu().numpy())}
                    r['input_perturbation']={'anchor_tail_delta_rms':rms((anchor-frontend).cpu().numpy()[A:B]),'condition_tail_delta_rms':rms((condition-noisy).cpu().numpy()[A:B]),'anchor_outside_max_abs':float((anchor[outside]-frontend[outside]).abs().max()),'condition_outside_max_abs':float((condition[outside]-noisy[outside]).abs().max()),'normalization_peak_unchanged':True}
                    if edit_condition and name=='condition_tail_clean':np.save(OUT/'condition_tail_clean_input_raw.npy',condition.cpu().numpy(),allow_pickle=False)
                    d['runs'][name]=r
                    torch.manual_seed(seed)
                    r['rng_before']={'cpu':ah(torch.get_rng_state().numpy()),'cuda':ah(torch.cuda.get_rng_state().cpu().numpy())}
                    tt=time.monotonic();out,nfe,rtf=model.enhance(anchor,y_cond=condition,**common,rrsd=False,timeit=True);torch.cuda.synchronize()
                    out=np.asarray(out);assert out.shape==(LENGTH,) and np.isfinite(out).all() and int(nfe)==30
                    np.save(OUT/f'{name}_raw.npy',out,allow_pickle=False)
                    pcm_path=OUT/f'{name}.wav';sf.write(pcm_path,np.clip(out,-1,1),SR,subtype='PCM_16')
                    pcm,_=sf.read(pcm_path,dtype='float32')
                    r.update({'nfe':int(nfe),'wall_seconds':time.monotonic()-tt,'rtf':float(rtf),'raw_dtype':str(out.dtype),'raw_sha256':ah(out),'raw_file_sha256':sha(OUT/f'{name}_raw.npy'),'pcm16_sha256':sha(pcm_path),'clipped_samples':int(np.count_nonzero(abs(out)>1))})
                    if name=='no_op':
                        r['exact_saved_pcm16_byte_match']=sha(pcm_path)==v['off']['sha256']
                        assert r['exact_saved_pcm16_byte_match'],'No-op not exact PCM16 match: STOP before causal runs'
                        raw_baseline=out.copy();pcm_baseline=pcm.copy()
                    r['raw_metrics']=summary(out,raw_baseline,clean,active);r['pcm16_reference_metrics']=summary(pcm,pcm_baseline,clean,active)
                    for i in range(len(active_frames)):
                        a=i*320;b=a+320
                        writer.writerow({'run':name,'start_sample':a,'end_sample_exclusive':b,'clean_rms':rms(clean[a:b]),'clean_active':int(active_frames[i]),'raw_rms':rms(out[a:b]),'raw_highband_mean_square':highband(out[a:b]),'raw_delta_no_op_rms':rms(out[a:b]-raw_baseline[a:b]),'raw_clean_error_rms':rms(out[a:b]-clean[a:b])})
                    f.flush();cap()
        assert all(sha(p)==expected[k] for k,p in paths.items()) and sha(ckpt)==cfg['checkpoint_sha256']
        d['sources_and_checkpoint_unchanged']=True;d['output_bytes']=cap();d['status']='completed'
    except Exception as exc:
        d['status']='failed';d['failures'].append({'error':repr(exc),'traceback':traceback.format_exc()});traceback.print_exc()
        archive=OUT/'failure_archive';archive.mkdir(exist_ok=False)
        with (archive/'failure_snapshot.json').open('x') as f:json.dump({'failure':d['failures'][-1],'preserved_artifacts':[str(p.relative_to(OUT)) for p in OUT.rglob('*') if p.is_file()],'action':'Stop; preserve all partial artifacts in place. No retry without diagnosis and a new attempt directory.'},f,indent=2)
    d['completed_at_utc']=utc();d['wall_seconds']=time.monotonic()-t0
    dump('metrics.json',d)
    report='# 10720 condition-tail 20dB replication\n\nStatus: '+d['status']+'\n\n'+d['limits']+'\n\nWindow: samples240800:252000, 15.05–15.75s, internal20ms raised cosine fades. Model loaded once; original FLOAT frontend recomputed once and hash checked. Both runs are backend off, NFE30. Full speech activity uses clean20ms disjoint frames above−70dBFS. Raw metrics compare with new no_op raw; PCM16 values are reference only. frames.csv retains raw fixed frame values. No clean/noisy corpus changes or downloads.\n\n'
    if d['status']=='completed':
        report+='|Run|Tail RMS dBFS|4–7.5kHz mean square|Outside relative L2|Active relative L2 vs no_op|\n|---|---:|---:|---:|---:|\n'
        for name,r in d['runs'].items():
            m=r['raw_metrics'];report+=f"|{name}|{m['tail_rms_dbfs']:.5f}|{m['tail_highband_mean_square_4k_7p5k']:.8g}|{m['outside_window_relative_l2_vs_no_op']:.8g}|{m['full_clean_speech_active_relative_l2_vs_no_op']:.8g}|\n"
    else:report+='Failure recorded with traceback. Preserve this directory; archive and diagnose before a separately approved attempt. No automatic retries.\n'
    with (OUT/'experiment_report.md').open('x') as f:f.write(report)
    print(json.dumps({'status':d['status'],'failures':d['failures'],'wall_seconds':d['wall_seconds']},indent=2))
    if d['status']!='completed':raise SystemExit(1)
if __name__=='__main__':
    if sys.argv[1:]==['--synthetic-checks']:print(json.dumps(synthetic_checks(),indent=2))
    elif not sys.argv[1:]:main()
    else:raise SystemExit('Only --synthetic-checks or no arguments accepted')
