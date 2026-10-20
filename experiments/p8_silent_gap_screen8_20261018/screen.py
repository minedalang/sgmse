#!/usr/bin/env python3
"""Offline fixed-threshold screen; no model imports or source writes."""
import csv
import hashlib
import io
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def db(rms):
    # No epsilon or floor changes the RMS or an undefined logarithm.
    return float(20*np.log10(rms)) if rms > 0 else None


def rms(x):
    return float(np.sqrt(np.mean(x*x))) if x.size else None


def gain_db(a, b):
    return float(20*np.log10(a/b)) if a > 0 and b > 0 else None


def corr(a,b):
    a,b = a-a.mean(),b-b.mean()
    den = np.linalg.norm(a)*np.linalg.norm(b)
    return float(np.dot(a,b)/den) if den > 0 else None


def runs(mask,min_frames):
    edges = np.diff(np.r_[False,mask,False].astype(int))
    return [(int(a),int(b)) for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)) if b-a>=min_frames]


def features(wave,fs,n):
    z=wave.reshape(-1,n)
    power=np.mean(z*z,axis=1)
    spec=np.abs(np.fft.rfft(z,axis=1))**2
    spec[:,1:-1]*=2
    freq=np.fft.rfftfreq(n,1/fs)
    high=spec[:,(freq>=4000)&(freq<=7500)].sum(axis=1)/(n*n)
    return {'frames':z,'power':power,'rms':np.sqrt(power),'high_power':high}


def relative_l2(x,c):
    den=np.linalg.norm(c)
    value=float(np.linalg.norm(x-c)/den) if den>0 else None
    return {'relative_l2':value,'relative_l2_db':db(value) if value is not None else None,
            'db_status':'undefined_zero_error' if value==0 else ('defined' if value is not None else 'undefined_zero_clean')}


def main():
    names=['metrics.json','experiment_report.md','gaps.csv','frame_masks.csv','provenance.json']
    for name in names:
        if (ROOT/name).exists():
            raise FileExistsError(f'Refusing overwrite: {name}')
    with (ROOT/'run_started.json').open('x') as f:
        json.dump({'utc':datetime.now(timezone.utc).isoformat()},f)
    try:
        run()
    except Exception as exc:
        with (ROOT/'failure.json').open('x') as f:
            json.dump({'type':type(exc).__name__,'message':str(exc)},f)
        raise


def run():
    cfg=yaml.safe_load((ROOT/'config.yaml').read_text())
    source_manifest=(ROOT/cfg['manifest']).resolve()
    snapshot=json.loads((ROOT/'source_snapshot.json').read_text())
    assert sha(source_manifest)==snapshot['manifest_sha256']
    records=list(csv.DictReader(source_manifest.open()))
    assert len(records)==8 and len({r['fileid'] for r in records})==8
    fs=cfg['sample_rate']; n=round(fs*cfg['frame_ms']/1000)
    assert n==320 and cfg['hop_ms']==cfg['frame_ms']
    cut=round(cfg['boundary_exclusion_seconds']*fs/n)
    minrun=round(cfg['minimum_gap_seconds']*fs/n)
    active_amp=10**(cfg['clean_active_min_dbfs']/20)
    all_gaps=[]; frame_rows=[]; utterances=[]; source_evidence=[]
    for rec in records:
        fileid=rec['fileid']; mapped=snapshot['files'][fileid]
        assert mapped['index']==int(rec['index']) and mapped['noisy_basename']==rec['noisy_basename']
        waves={}; fts={}
        for arm,s in mapped['sources'].items():
            path=Path(s['path']); assert sha(path)==s['sha256']
            assert re.search(r'fileid_'+re.escape(fileid)+r'\.wav$',path.name)
            wave,rate=sf.read(path,dtype='float64')
            assert rate==fs and wave.ndim==1 and len(wave)==cfg['samples'] and np.all(np.isfinite(wave))
            if arm in ['noisy','clean']:
                assert s['sha256']==rec[arm+'_sha256']
            waves[arm]=wave; fts[arm]=features(wave,fs,n)
            source_evidence.append(dict(fileid=fileid,arm=arm,**s,samples=len(wave),sample_rate=rate,
                decoded_float64_sha256=hashlib.sha256(wave.tobytes()).hexdigest()))
        clean=fts['clean']; count=len(clean['rms'])
        interior=np.zeros(count,dtype=bool);interior[cut:count-cut]=True
        quiet=(clean['rms']<=cfg['clean_quiet_max_rms']) & interior
        active=(clean['rms']>=active_amp) & interior
        unclassified=interior & ~quiet & ~active
        gap_runs=runs(quiet,minrun)
        eligible=np.zeros(count,dtype=bool)
        label='discovery_10720' if fileid==str(cfg['discovery_fileid']) else 'exploratory_other7'
        summary={'fileid':fileid,'index':int(rec['index']),'speaker':rec['speaker'],'group':label,
            'interior_frames':int(interior.sum()),'boundary_excluded_frames':int((~interior).sum()),
            'clean_quiet_frames':int(quiet.sum()),'clean_active_frames':int(active.sum()),
            'unclassified_frames':int(unclassified.sum()),'quiet_duration_seconds':float(quiet.sum()*n/fs),
            'qualifying_gap_count':len(gap_runs),'qualifying_gap_seconds':float(sum(b-a for a,b in gap_runs)*n/fs),
            'backend':{},'active_complement':{}}
        for g,(a,b) in enumerate(gap_runs):
            eligible[a:b]=True
            row={'fileid':fileid,'index':int(rec['index']),'speaker':rec['speaker'],'group':label,
                 'gap_id':f'{fileid}_{g:03d}','start_frame':a,'end_frame_exclusive':b,
                 'start_sample':a*n,'end_sample_exclusive':b*n,
                 'start_seconds':a*n/fs,'end_seconds':b*n/fs,'duration_seconds':(b-a)*n/fs,
                 'all_clean_frames_quiet':bool(np.all(quiet[a:b]))}
            for arm,f in fts.items():
                value=rms(waves[arm][a*n:b*n])
                high=float(np.mean(f['high_power'][a:b]))
                row[arm+'_rms']=value;row[arm+'_dbfs']=db(value)
                row[arm+'_highband_mean_square']=high
                row[arm+'_highband_fraction']=high/(value*value) if value>0 else None
            for arm in ['off','router']:
                gain=gain_db(row[arm+'_rms'],row['y_pre_rms'])
                row[arm+'_gain_vs_ypre_db']=gain
                row[arm+'_gain_status']='defined' if gain is not None else 'undefined_zero_rms'
                row[arm+'_rank_flag']=bool(row[arm+'_dbfs'] is not None and row[arm+'_dbfs']>cfg['backend_rank_min_dbfs_exclusive'] and gain is not None and gain>=cfg['backend_rank_min_gain_vs_ypre_db'])
                row[arm+'_noise_wave_pearson_r']=corr(waves[arm][a*n:b*n],waves['noise'][a*n:b*n])
                row[arm+'_noise_rms_envelope_pearson_r']=corr(fts[arm]['rms'][a:b],fts['noise']['rms'][a:b])
                # Signed output-minus-Y_pre energy change and waveform-difference energy
                # are both recorded: the latter also includes phase changes.
                delta=waves[arm][a*n:b*n]-waves['y_pre'][a*n:b*n]
                row[arm+'_delta_vs_ypre_energy_sample_sum']=float(np.sum(delta*delta))
                row[arm+'_signed_energy_change_sample_sum']=float(np.sum(waves[arm][a*n:b*n]**2-waves['y_pre'][a*n:b*n]**2))
            all_gaps.append(row)
        for arm in ['y_pre','off','router']:
            summary['active_complement'][arm]=relative_l2(fts[arm]['frames'][active].ravel(),clean['frames'][active].ravel())
        for arm in ['off','router']:
            rows=[r for r in all_gaps if r['fileid']==fileid]
            summary['backend'][arm]={'rank_flagged_gap_count':sum(r[arm+'_rank_flag'] for r in rows),
                'rank_flagged_duration_seconds':sum(r['duration_seconds'] for r in rows if r[arm+'_rank_flag']),
                'gap_delta_vs_ypre_energy_sample_sum':sum(r[arm+'_delta_vs_ypre_energy_sample_sum'] for r in rows),
                'gap_delta_vs_ypre_energy_seconds':sum(r[arm+'_delta_vs_ypre_energy_sample_sum'] for r in rows)/fs,
                'gap_signed_energy_change_sample_sum':sum(r[arm+'_signed_energy_change_sample_sum'] for r in rows),
                'all_quiet_delta_vs_ypre_energy_sample_sum':float(np.sum((fts[arm]['frames'][quiet]-fts['y_pre']['frames'][quiet])**2))}
        for i in range(count):
            frame_rows.append({'fileid':fileid,'frame':i,'start_sample':i*n,
                'boundary_excluded':bool(not interior[i]),'clean_quiet':bool(quiet[i]),
                'clean_active':bool(active[i]),'unclassified':bool(unclassified[i]),
                'in_qualifying_gap':bool(eligible[i]),'clean_rms':float(clean['rms'][i])})
        utterances.append(summary)
    for s in source_evidence:
        assert sha(s['path'])==s['sha256']
    grouped={}
    for group in ['discovery_10720','exploratory_other7']:
        u=[r for r in utterances if r['group']==group]
        grouped[group]={'utterances':len(u),'qualifying_gap_count':sum(r['qualifying_gap_count'] for r in u),
            'qualifying_gap_seconds':sum(r['qualifying_gap_seconds'] for r in u),
            'quiet_duration_seconds':sum(r['quiet_duration_seconds'] for r in u),
            'backend':{arm:{'rank_flagged_gap_count':sum(r['backend'][arm]['rank_flagged_gap_count'] for r in u),
                'utterances_with_rank_flag':sum(r['backend'][arm]['rank_flagged_gap_count']>0 for r in u),
                'gap_delta_vs_ypre_energy_seconds':sum(r['backend'][arm]['gap_delta_vs_ypre_energy_seconds'] for r in u)} for arm in ['off','router']}}
    metrics={'experiment_id':cfg['experiment_id'],'status':'completed','interpretation':cfg['interpretation'],
        'threshold_note':cfg['threshold_note'],'noise_note':'Exact archived paired noise WAV; not noisy minus clean proxy.',
        'db_note':'No RMS floor. Zero RMS or zero reference gives null/undefined dB or ratio; never silently floored.',
        'highband_note':'Rectangular 20ms frame FFT, one-sided Parseval power, inclusive 4–7.5kHz bins.',
        'addition_note':'Delta waveform energy versus Y_pre includes phase changes; signed output energy change also retained. Neither identifies speech.',
        'utterances':utterances,'groups':grouped,'all_sources_unchanged':True,'new_model_inference':False}
    provenance={'script_sha256':sha(__file__),'config_sha256':sha(ROOT/'config.yaml'),
        'source_snapshot_sha256':sha(ROOT/'source_snapshot.json'),'manifest_sha256':sha(source_manifest),
        'python':platform.python_version(),'numpy':np.__version__,'soundfile':sf.__version__,
        'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        'git_status':subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),
        'sources':source_evidence,'completed_utc':datetime.now(timezone.utc).isoformat()}
    def csv_text(rows,fallback):
        f=io.StringIO();w=csv.DictWriter(f,fieldnames=list(rows[0]) if rows else fallback)
        w.writeheader();w.writerows(rows);return f.getvalue()
    report=['# Exploratory P8 silent-gap screen','',cfg['interpretation'],'',cfg['threshold_note'],'',
        'All qualifying gaps appear in gaps.csv. frame_masks.csv includes every frame, boundaries, quiet/active/unclassified masks and exact clean RMS. Gap rank flags require both fixed backend thresholds. Undefined gain never passes the ranking rule.',
        '', 'The prior local probe used 9–21 s clean-active 0.5 s windows, 32 ms overlapping Hann frames and adaptive clean activity. This screen uses full 30 s sources, disjoint 20 ms frames, fixed quiet runs and excluded boundary frames. It does not repeat the prior configuration.',
        '', '| Fileid | Group | Gaps | Quiet seconds | Off flags | Router flags |', '|---|---|---:|---:|---:|---:|']
    for u in utterances:
        report.append(f"| {u['fileid']} | {u['group']} | {u['qualifying_gap_count']} | {u['quiet_duration_seconds']:.2f} | {u['backend']['off']['rank_flagged_gap_count']} | {u['backend']['router']['rank_flagged_gap_count']} |")
    report+=['','Clean-active complementary relative L2 metrics use disjoint frames at or above -70 dBFS. Frames above the quiet threshold but below the active threshold are unclassified. Boundary frames are excluded from all aggregate masks. No speech, word, or hallucination label is produced.']
    payload={'metrics.json':json.dumps(metrics,indent=2,allow_nan=False)+'\n',
        'provenance.json':json.dumps(provenance,indent=2,allow_nan=False)+'\n',
        'gaps.csv':csv_text(all_gaps,['fileid','gap_id','start_seconds','end_seconds']),
        'frame_masks.csv':csv_text(frame_rows,[]),'experiment_report.md':'\n'.join(report)+'\n'}
    total=sum(len(t.encode()) for t in payload.values())+sum(p.stat().st_size for p in ROOT.iterdir() if p.is_file())
    assert total<cfg['max_total_bytes'],f'Output budget exceeded: {total}'
    for name,text in payload.items():
        with (ROOT/name).open('x') as out:out.write(text)


if __name__=='__main__':main()
