from pathlib import Path
import json, time, hashlib
import torch, whisper

PACK = Path('/data/coding/sgmse/experiments/p8_e1_candidate_listening_20261018')
ROOT = Path('/data/coding/sgmse/experiments/p8_frontend_backend_audit_20261020/listening_pack')
rows = json.loads((PACK/'manifest.json').read_text())
model = whisper.load_model('tiny.en', device='cuda' if torch.cuda.is_available() else 'cpu', download_root='/data/.rootcache/whisper')
out=[]
started=time.time()
for row in rows:
    fid = row['fileid']
    clean = next(ROOT.glob(f'*_clean_*fileid_{fid}_center12s.wav'))
    # Candidate times are in original 30 s audio; listening_pack starts at t=9 s.
    a,b = [float(x)-9.0 for x in row['candidate_window_s']]
    result = model.transcribe(str(clean), language='en', task='transcribe', temperature=0,
                              beam_size=5, best_of=5, condition_on_previous_text=False,
                              word_timestamps=True, fp16=torch.cuda.is_available(), verbose=False)
    words=[]
    for seg in result.get('segments',[]):
        for w in seg.get('words',[]):
            words.append({'word':w.get('word','').strip(),'start_s':w.get('start'),'end_s':w.get('end')})
    hit=[w for w in words if w['end_s'] is not None and w['start_s'] is not None and w['end_s']>=a and w['start_s']<=b]
    out.append({'fileid':fid,'candidate_original_s':row['candidate_window_s'],'candidate_center12_s':[a,b],
                'clean_audio':str(clean),'full_text':result.get('text','').strip(),'overlapping_words':hit})
(PACK/'clean_timestamp_prompt.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
lines=['# P8-E1 候选窗 clean 时间戳自动转写提示','','这是 Whisper tiny.en beam-5 的辅助提示，不是真实文本标注。候选窗原始时间减去 9 秒后映射到中央 12 秒 clean 音频。','']
for r in out:
    lines += [f"## file-id {r['fileid']}（原始 {r['candidate_original_s'][0]:.2f}–{r['candidate_original_s'][1]:.2f} s；中央片段 {r['candidate_center12_s'][0]:.2f}–{r['candidate_center12_s'][1]:.2f} s）",'',f"候选窗附近自动词：{' '.join(x['word'] for x in r['overlapping_words']) or '(无稳定词级输出，听 clean 全段)' }",'',f"中央 12 秒自动转写：{r['full_text']}",'']
(PACK/'clean_timestamp_prompt.md').write_text('\n'.join(lines))
meta={'experiment_id':'p8_e1_clean_timestamp_prompt_20261018','model':'whisper-tiny.en','device':'cuda' if torch.cuda.is_available() else 'cpu','beam_size':5,'word_timestamps':True,'rows':len(out),'runtime_seconds':time.time()-started,'dataset_modified':False,'checkpoint_modified':False,'interpretation':'automatic listening prompt only; not ground-truth transcript'}
(Path(__file__).resolve().parent/'metrics.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
