from pathlib import Path
import json,re,hashlib,collections,time
D=Path(__file__).resolve().parent

def main():
 t=time.monotonic();cfg=json.loads((D/'config.yaml').read_text());fetch=json.loads((D/'fetch_result.json').read_text());raw=(D/'log_trainset_28spk.txt').read_bytes();text=raw.decode('ascii');lines=text.splitlines();rows=[];by={}
 for i,line in enumerate(lines,1):
  fields=line.split();assert len(fields)==3 and re.fullmatch(r'p\d+_\d+',fields[0]) and re.fullmatch(r'[A-Za-z0-9_\-]+',fields[1]) and re.fullmatch(r'[-+]?\d+(?:\.\d+)?',fields[2]),(i,line)
  assert fields[0] not in by,'duplicate ID';r={'id':fields[0],'literal_noise_token':fields[1],'column3_raw':fields[2],'column3_numeric':float(fields[2]),'source_line_number':i,'source_line_raw':line};by[r['id']]=r;rows.append(r)
 reg=json.loads(Path(cfg['registry']).read_text());selected=[]
 for r in reg['records']:
  x=by[r['id']];selected.append({**x,'speaker':r['speaker'],'nominal_SNR_dB':None,'column3_semantics':'numeric field; likely nominal SNR but header/column definition not present in opened train entry, not certified','noise_recording_id':'unknown','noise_identity_interpretation':'literal category/scenario token only; no unique recording or offset','offset':None,'channel':None,'gain':None,'segment_length':None,'noise_source_sha256':None,'noise_disjoint_ready':False,'scientific_gate_ready':False,'split_frozen':False})
 remaining={r['id'] for r in selected};components=[]
 while remaining:
  component={min(remaining)}
  while True:
   speakers={r['speaker'] for r in selected if r['id'] in component};tokens={r['literal_noise_token'] for r in selected if r['id'] in component};grown=component|{r['id'] for r in selected if r['speaker'] in speakers or r['literal_noise_token'] in tokens}
   if grown==component:break
   component=grown
  components.append(sorted(component));remaining-=component
 result={'status':'completed_metadata_only','encoding':'ASCII','schema':'three whitespace columns: utteranceID/literal noise token/numeric column3; no header','train_line_count':len(lines),'unique_IDs':len(by),'duplicate_ID_count':0,'column_count':3,'column3_unique_numeric_values':sorted({x['column3_numeric'] for x in rows}),'literal_noise_tokens_all_training':sorted({x['literal_noise_token'] for x in rows}),'selected_count':12,'selected':selected,'components_speaker_or_same_literal_token':components,'graph_is_not_recording_identity_or_independent_split':True,'noise_disjoint_ready':False,'scientific_gate_ready':False,'split_frozen':False,'archive_and_training':fetch,'offline_parse_seconds':time.monotonic()-t,'network_requests':3,'network_body_bytes':fetch['network_body_bytes'],'opened_payloads':[fetch['train_entry']],'test_56_payload_open_count':0,'audio_decodes':0,'models':0,'ASR':0,'training':0,'retries':0,'source_locks_pre':cfg['source_locks']}
 for p,x in cfg['source_locks'].items():assert Path(p).stat().st_size==x['bytes'] and hashlib.sha256(Path(p).read_bytes()).hexdigest()==x['sha256']
 result['source_locks_post_equal']=True;(D/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'train_count':len(rows),'join12':[(r['id'],r['literal_noise_token'],r['column3_raw']) for r in selected],'components':components}))
if __name__=='__main__':main()
