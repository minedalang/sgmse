"""Independent standard-library CPU validator; no audio decoding or model imports."""
from pathlib import Path
import hashlib, json, re, zipfile
P=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def normal(text):
    t=text.replace('\u2019',"'").replace('\u2018',"'").lower()
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)*",t),re.findall('[a-z0-9]',t)
def independent_align(ref,hyp):
    matrix=[[i+j for j in range(len(hyp)+1)] for i in range(len(ref)+1)]
    for i,r in enumerate(ref,1):
        for j,h in enumerate(hyp,1):matrix[i][j]=min(matrix[i-1][j-1]+int(r!=h),matrix[i-1][j]+1,matrix[i][j-1]+1)
    i=len(ref);j=len(hyp);steps=[];counts=dict(H=0,S=0,D=0,I=0)
    while i+j:
        choices=[]
        if i and j:choices.append((matrix[i-1][j-1]+int(ref[i-1]!=hyp[j-1]),0,'H' if ref[i-1]==hyp[j-1] else 'S',i-1,j-1))
        if i:choices.append((matrix[i-1][j]+1,1,'D',i-1,j))
        if j:choices.append((matrix[i][j-1]+1,2,'I',i,j-1))
        cost,_,op,ni,nj=min(choices);assert cost==matrix[i][j]
        ri=None if op=='I' else ni;hj=None if op=='D' else nj
        steps.append({'op':op,'reference_index':ri,'hypothesis_index':hj,'reference':None if ri is None else ref[ri],'hypothesis':None if hj is None else hyp[hj]});counts[op]+=1;i,j=ni,nj
    steps.reverse();n=len(ref);err=matrix[n][len(hyp)]
    assert counts['S']+counts['D']+counts['I']==err
    return dict(counts,N=n,N_hyp=len(hyp),edit_distance=err,rate=err/n if n else None,rate_status='defined' if n else 'undefined_empty_reference',alignment=steps)
def main():
    cfg=json.loads((P/'config.yaml').read_text());metrics=json.loads((P/'metrics.json').read_text());checks=[]
    assert metrics['status']=='completed';assert metrics['transcribe_api_calls']==metrics['audio_decodes']==4
    assert metrics['model_loads']==1 and not metrics['network_attempts'] and metrics['network_bytes']==metrics['enhancement_calls']==metrics['training_calls']==0
    for name,h in cfg['source_locks'].items():assert sha(name)==h
    for name,h in json.loads((P/'frozen_manifest.json').read_text()).items():assert sha(P/name)==h
    for name,entry in metrics['artifact_manifest'].items():assert (P/name).stat().st_size==entry['bytes'] and sha(P/name)==entry['sha256']
    rows={}
    for line in Path(cfg['text_manifest']).open():
        row=json.loads(line)
        if row['id'] in cfg['ids']:rows[row['id']]=row
    scores=[]
    with zipfile.ZipFile(cfg['text_zip']) as z:
        for sid in cfg['ids']:
            raw=json.loads((P/f'outputs/{sid}_raw.json').read_text());sc=json.loads((P/f'outputs/{sid}_scoring.json').read_text())
            t=rows[sid]['transcript'];text_bytes=z.read(t['archive_entry']);ref=text_bytes.decode('utf-8-sig');hyp=raw['raw_result']['text']
            assert hashlib.sha256(text_bytes).hexdigest()==t['raw_text_SHA256'] and text_bytes.hex()==raw['publisher_reference']['raw_text_hex']
            assert ref==sc['original_reference']==t['original_text'] and hyp==sc['original_hypothesis']
            rw,rc=normal(ref);hw,hc=normal(hyp)
            assert sc['reference_word_tokens']==rw and sc['hypothesis_word_tokens']==hw
            assert sc['reference_scoring_words']==' '.join(rw) and sc['hypothesis_scoring_words']==' '.join(hw)
            assert sc['reference_scoring_chars']==''.join(rc) and sc['hypothesis_scoring_chars']==''.join(hc)
            assert independent_align(rw,hw)==sc['word'];assert independent_align(rc,hc)==sc['character']
            assert sc==metrics['samples'][sid]['score']
            assert raw['input']['input_WAV_SHA256']==sha(next(s['audio'] for s in cfg['samples'] if s['id']==sid))
            for entry in raw['decode_calls']:assert entry['options']==metrics['effective_decoding_options_before_inference']
            checks.append({'id':sid,'word_and_character_alignment_exact':True,'publisher_raw_reference_exact':True});scores.append(sc)
    assert [s['word']['N'] for s in scores]==[12,9,9,9]
    for kind in ['word','character']:
        sums={key:sum(s[kind][key] for s in scores) for key in ['S','D','I','H','N','N_hyp','edit_distance']}
        sums.update(micro_rate=sums['edit_distance']/sums['N'],sentence_mean_rate=sum(s[kind]['rate'] for s in scores)/4)
        assert sums==metrics['aggregate'][kind]
    assert metrics['internal_decode_api_calls']==sum(metrics['samples'][sid]['decode_api_calls'] for sid in cfg['ids'])
    result={'status':'passed','checks':checks,'source_locks':len(cfg['source_locks']),'metrics_SHA256_unchanged':sha(P/'metrics.json'),'models':0,'network_calls':0,'audio_decodes':0,'independent_scoring':True}
    (P/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
