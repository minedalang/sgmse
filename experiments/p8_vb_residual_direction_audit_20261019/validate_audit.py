"""Independent direct PCM scalar reconstruction; does not import run_audit."""
from pathlib import Path
import math,json,wave,struct,hashlib,time,signal,os
B=Path(__file__).resolve().parent

def validate():
 signal.alarm(60);start=time.monotonic();m=json.loads((B/'metrics.json').read_text());assert m['status']=='completed';locks=json.loads((B/'source_pcm_locks.json').read_text());assert len(locks)==72;cache={};decodes=0
 sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
 for p,x in locks.items():
  assert sha(p)==x['sha256'] and Path(p).stat().st_size==x['bytes']
  with wave.open(p,'rb') as w:
   assert x['header']==dict(channels=w.getnchannels(),width=w.getsampwidth(),rate=w.getframerate(),frames=w.getnframes(),compression=w.getcomptype());raw=w.readframes(w.getnframes())
  cache[p]=[a[0]/32768. for a in struct.iter_unpack('<h',raw)];decodes+=1
 close=lambda a,b: a==b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-13)
 avg=lambda xs:math.fsum(xs)/len(xs)
 direct={}
 for row in m['cases']:
  s=cache[row['paths']['S']];f=cache[row['paths']['F']];o=cache[row['paths']['B']];L=len(s);assert L==len(f)==len(o)==row['samples'];p=math.fsum(v*v for v in s)/L;pf=math.fsum(v*v for v in f)/L
  errors=[a-b for a,b in zip(f,s)];changes=[b-a for a,b in zip(f,o)]
  # Direct in-memory scalar residuals at endpoints and recorded oracle, rather than quadratic evaluation.
  mse=lambda a:math.fsum((ff+ a*(oo-ff)-ss)**2 for ss,ff,oo in zip(s,f,o))/L
  en=math.fsum(v*v for v in changes)/L;dot=math.fsum(e*d for e,d in zip(errors,changes))/L;a=max(0.,min(1.,-dot/en)) if en else 0.;u=-dot/en if en else None
  assert close(row['alpha_unconstrained'],u) and close(row['alpha_oracle'],a)
  assert row['alpha_class']==('no_change' if en==0 else 'zero' if a==0 else 'one' if a==1 else 'interior')
  for k,v in {'P':p,'PF':pf,'E0':mse(0),'E1':mse(1),'A':dot,'D':en,'E_oracle':mse(a),'E1_minus_E0':mse(1)-mse(0),'oracle_minus_frontend':mse(a)-mse(0),'oracle_minus_OFF':mse(a)-mse(1)}.items():assert close(row[k],v),(row['basename'],k)
  for k,v in row['normalized_by_P'].items():assert close(v,row[k]/p)
  assert close(row['identity_residual'],mse(1)-(mse(0)+2*dot+en)) and mse(a)<=min(mse(0),mse(1))+1e-13
  b=math.fsum((oo-ff)*ff for ff,oo in zip(f,o))/(L*pf) if pf>0 else None;gain=[b*ff if b is not None else 0. for ff in f];rest=[oo-ff-gg for ff,oo,gg in zip(f,o,gain)]
  genergy=math.fsum(gg*gg for gg in gain)/L;renergy=math.fsum(rr*rr for rr in rest)/L
  # Changes in direct residual energy, not recycled part-contribution formula.
  gc=math.fsum(((e+g)**2-e**2) for e,g in zip(errors,gain))/L;rc=math.fsum(((e+r)**2-e**2) for e,r in zip(errors,rest))/L;cross=math.fsum(g*r for g,r in zip(gain,rest))/L
  for k,v in {'gain_beta':b,'gain_energy':genergy,'rest_energy':renergy,'gain_rest_inner':cross,'gain_MSE_contribution':gc,'rest_MSE_contribution':rc,'part_contribution_residual':mse(1)-mse(0)-gc-rc,'rest_energy_fraction':renergy/en if en else None}.items():assert close(row[k],v),(row['basename'],k)
  assert abs(cross)<=1e-12*max(1.,en)
  direct[row['basename']]={'row':row,'s':s,'f':f,'b':o,'P':p,'A':dot,'D':en,'mse':mse(a)}
 # Independently rebuild21 index lists without driver grouping helper.
 sets={'all':list(direct)}
 for field,prefix,values in [('noise','noise_',['N1','N2']),('snr_db','snr_',[-18,0]),('speaker','speaker_',sorted({r['speaker'] for r in m['cases']})),('id','utterance_',sorted({r['id'] for r in m['cases']}))]:
  for value in values:sets[prefix+str(value)]=[r['basename'] for r in m['cases'] if r[field]==value]
 for n in ['N1','N2']:
  for q in [-18,0]:sets[n+'_'+str(q)]=[r['basename'] for r in m['cases'] if r['noise']==n and r['snr_db']==q]
 assert set(sets)==set(m['strata']) and len(sets)==21
 for name,ids in sets.items():
  rows=[direct[i] for i in ids];out=m['strata'][name];num=math.fsum(r['A']/r['P'] for r in rows);den=math.fsum(r['D']/r['P'] for r in rows);u=-num/den if den else None;a=max(0.,min(1.,u)) if u is not None else 0.
  assert out['cases']==len(rows) and close(out['common_alpha_unconstrained'],u) and close(out['common_alpha_oracle'],a)
  objectives={}
  for label,alpha in [('mean_frontend_normalized_MSE',0.),('mean_OFF_normalized_MSE',1.),('mean_common_alpha_normalized_MSE',a)]:
   vals=[]
   for r in rows:vals.append(math.fsum((ff+alpha*(oo-ff)-ss)**2 for ss,ff,oo in zip(r['s'],r['f'],r['b']))/len(r['s'])/r['P'])
   objectives[label]=avg(vals)
  objectives['mean_case_oracle_normalized_MSE']=avg([r['mse']/r['P'] for r in rows])
  for k,v in objectives.items():assert close(out[k],v),(name,k)
  assert out['alpha_classes']=={c:sum(r['row']['alpha_class']==c for r in rows) for c in ['no_change','zero','interior','one']}
  for key,field in [('mean_gain_MSE_contribution_normalized','gain_MSE_contribution'),('mean_rest_MSE_contribution_normalized','rest_MSE_contribution')]:assert close(out[key],avg([r['row'][field]/r['P'] for r in rows]))
  rf=[r['row']['rest_energy_fraction'] for r in rows if r['row']['rest_energy_fraction'] is not None];assert close(out['mean_rest_energy_fraction'],avg(rf) if rf else None)
  si=[r['row']['saved_original_OFF_minus_frontend_si_sdr_db'] for r in rows if r['row']['saved_original_OFF_minus_frontend_si_sdr_db'] is not None];assert close(out['mean_saved_OFF_minus_frontend_si_sdr_db'],avg(si))
 source_m=json.loads(Path('/data/coding/sgmse/experiments/p8_vb_prospective_headroom_20261019/model_stage/metrics.json').read_text());saved={r['basename']:r['off_minus_frontend']['si_sdr'] for r in source_m['quality']['per_case']}
 assert all(saved[r['basename']]==r['saved_original_OFF_minus_frontend_si_sdr_db'] for r in m['cases'])
 before=json.loads((B/'source_preservation_pre.json').read_text());after=json.loads((B/'source_preservation_post.json').read_text());assert before==after
 for p,h in before.items():assert sha(p)==h
 result={'status':'passed','cases':32,'strata':21,'additional_original_PCM_decodes':decodes,'main_PCM_decodes':m['source_pcm_decodes'],'total_real_PCM_decodes':decodes+m['source_pcm_decodes'],'direct_residual_and_projection_reconstruction':True,'seconds':time.monotonic()-start,'metrics_sha256':sha(B/'metrics.json'),'validator_sha256':sha(__file__),'preserved_files':len(before),'models':0,'ASR':0,'quality_recomputations':0,'audio_outputs':0,'network':0}
 with (B/'independent_validation.json').open('x') as f:json.dump(result,f,indent=2)
 signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':validate()
