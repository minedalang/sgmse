"""Independent scalar pairwise finite-ensemble validation; no business-function import."""
from pathlib import Path
import json,math,hashlib,struct,wave,time,signal,itertools
D=Path(__file__).resolve().parent

def main():
 signal.alarm(120);t=time.monotonic();read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest();m=read(D/'metrics.json');assert m['status']=='completed';locks=read(D/'source_PCM_locks.json');sources=read(D/'source_registry.json')['source_rows'];assert len(locks)==80
 ndecode=0
 def decode(p):
  nonlocal ndecode
  x=locks[p];assert sha(p)==x['sha256'] and Path(p).stat().st_size==x['bytes']
  with wave.open(p,'rb') as w:
   assert x['header']==dict(rate=w.getframerate(),channels=w.getnchannels(),width=w.getsampwidth(),frames=w.getnframes(),compression=w.getcomptype());b=w.readframes(w.getnframes())
  assert len(b)==x['header']['frames']*2;ndecode+=1;return [v[0]/32768. for v in struct.iter_unpack('<h',b)]
 close=lambda a,b:a==b if a is None or b is None else math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-13)
 rebuilt=[]
 for src,row in zip(sources,m['files']):
  assert src['filename']==row['filename'] and src['speaker']==row['speaker'];s=decode(src['clean']);bs=[decode(v['wav']) for v in src['trajectories']];L=len(s);assert row['samples']==L and all(len(b)==L for b in bs)
  p=math.fsum(x*x for x in s)/L;ends={str(v['seed']):math.fsum((x-y)**2 for x,y in zip(b,s))/L for b,v in zip(bs,src['trajectories'])};assert set(ends)=={'2026','2027','2028','2029'}
  # Distinct V implementation: sum six pairwise distances, not deviations from sample mean.
  vp=math.fsum(math.fsum((x-y)**2 for x,y in zip(bs[i],bs[j]))/L for i,j in itertools.combinations(range(4),2))/16
  # Direct residual of four-way mean in temporary scalar expression; no sample array output.
  em=math.fsum(((a+b+c+d)/4-z)**2 for a,b,c,d,z in zip(*bs,s))/L;ea=math.fsum(ends.values())/4;vu=vp*4/3;be=em-vu/4
  v={'E_average':ea,'E_mean':em,'V_population4':vp,'V_unbiased_under_IID':vu,'finite4_bias2_est_raw_under_IID':be};assert close(p,row['P'])
  for key,value in v.items():assert close(row[key],value) and close(row['normalized_by_P'][key],value/p),(row['filename'],key)
  for seed,e in ends.items():assert close(row['seed_endpoint_MSEs'][seed],e)
  assert close(row['identity_residual'],ea-em-vp) and close(ea,em+vp);assert close(row['population_variance_fraction'],vp/ea if ea else None);assert row['negative_bias2_est']==(be<0)
  rebuilt.append({'speaker':src['speaker'],'v':{k:z/p for k,z in v.items()},'fraction':vp/ea if ea else None,'negative':be<0})
 assert len(rebuilt)==16 and ndecode==80
 group={'all':rebuilt}
 for s in sorted({r['speaker'] for r in rebuilt}):group['speaker_'+s]=[r for r in rebuilt if r['speaker']==s]
 assert len(group)==9 and set(group)==set(m['strata'])
 for label,rs in group.items():
  d=m['strata'][label];assert d['files']==len(rs);means={k:math.fsum(x['v'][k] for x in rs)/len(rs) for k in rebuilt[0]['v']}
  for k,v in means.items():assert close(d['mean_normalized'][k],v),(label,k)
  ratios=[r['fraction'] for r in rs if r['fraction'] is not None];assert close(d['mean_file_variance_fraction'],math.fsum(ratios)/len(ratios) if ratios else None)
  assert close(d['pooled_normalized_variance_fraction'],means['V_population4']/means['E_average'] if means['E_average'] else None);assert d['negative_bias2_est_count']==sum(r['negative'] for r in rs);assert close(d['mean_identity_residual'],means['E_average']-means['E_mean']-means['V_population4'])
 before=read(D/'source_metadata_pre.json');assert before==read(D/'source_metadata_post.json')
 for p,h in before.items():assert sha(p)==h
 assert m['audio_decodes']==80 and m['historical_original_byte_identity_proven'] is False
 result={'status':'passed','files':16,'strata':9,'unique_original_PCM':80,'main_decodes':80,'additional_validator_decodes':ndecode,'total_actual_decodes':160,'independent_pairwise_population_variance':True,'direct_mean_residual':True,'seconds':time.monotonic()-t,'metrics_sha256':sha(D/'metrics.json'),'validator_sha256':sha(__file__),'historical_original_WAV_hash_gap_unresolved':True,'models':0,'ASR':0,'quality_recomputations':0,'audio_outputs':0}
 with (D/'independent_validation.json').open('x') as f:json.dump(result,f,indent=2)
 signal.alarm(0);print(json.dumps(result))
if __name__=='__main__':main()
