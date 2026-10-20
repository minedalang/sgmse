"""Build six existing aligned clips; no inference or source mutation."""
from pathlib import Path
import hashlib
import io
from html import escape
import json
from urllib.request import Request, urlopen
import numpy as np
import soundfile as sf

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parent
ALIGNED = ROOT / 'p8_e1_candidate_listening_aligned_20261020'
BASE = 'http://localhost:8766/' + EXP.name + '/listening/'
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
metrics = json.loads((EXP/'metrics.json').read_text())
selection = json.loads((EXP/'selection_frozen.json').read_text())
registry = json.loads((ALIGNED/'metrics.json').read_text())
record = next(r for r in registry['records'] if r['fileid'] == 5243)
feedback_path = HERE / 'user_feedback_20261019.json'
feedback = json.loads(feedback_path.read_text())
amendment_path = HERE / 'user_feedback_amendment_20261019.json'
amendment = json.loads(amendment_path.read_text())
assert record['context_window_s'] == [18.75,20.75]
assert selection['context_samples'] == [300000,332000]
assert [selection['target']['start_s'],selection['target']['end_s']] == [19.35,19.55]
manifest = {'status':'passed','url':BASE+'index.html','source_interval_s':[18.75,20.75],'source_interval_samples':[300000,332000],'amplitude':'original; no normalization or gain change','sources':[],'documents':{str(p):digest(p) for p in [EXP/'metrics.json',EXP/'selection_frozen.json',ALIGNED/'metrics.json']}}
labels = ['1 · 干净参考','2 · 前端 Y_pre','3 · 原后端','4 · 目标附近移除条件语音','5 · 前方位置对照（可选）','6 · 后方位置对照（可选）']
clips = []
for i, arm in enumerate(['clean','y_pre','native','target','left','right']):
    if arm in ['clean','y_pre']:
        source = ALIGNED/'clips'/('00_fileid_5243_19.50-20.00s_context_'+arm+'.wav')
        full = Path(record['source_paths'][arm])
        assert digest(full) == record['source_sha256'][arm]
        original, sr = sf.read(full,dtype='int16')
        raw = Path(selection['frozen_original_anchor']['path']) if arm=='y_pre' else full
        assert digest(raw) == selection['source_locks'][str(raw)]
        if arm=='y_pre':
            arr=np.load(raw).reshape(-1)
            assert hashlib.sha256(arr.tobytes()).hexdigest() == selection['frozen_original_anchor']['array_sha256']
            buffer=io.BytesIO();sf.write(buffer,np.clip(arr,-1,1),16000,format='WAV',subtype='PCM_16');buffer.seek(0);pcm,_=sf.read(buffer,dtype='int16')
            assert np.array_equal(pcm,original)
    else:
        source=EXP/'outputs'/(arm+'_context.wav')
        full=EXP/'outputs'/(arm+'.wav')
        raw=EXP/'outputs'/(arm+'_raw.npy')
        for path in [source,full,raw]:
            assert digest(path)==metrics['artifact_manifest'][str(path.relative_to(EXP))]['sha256']
        original,sr=sf.read(full,dtype='int16')
        arr=np.load(raw).reshape(-1)
        buffer=io.BytesIO();sf.write(buffer,np.clip(arr,-1,1),16000,format='WAV',subtype='PCM_16');buffer.seek(0);pcm,_=sf.read(buffer,dtype='int16')
        assert np.array_equal(pcm,original)
        assert metrics['variants'][arm]['anchor']['sha256']==selection['frozen_original_anchor']['array_sha256']
    audio,clip_sr=sf.read(source,dtype='int16')
    info=sf.info(source)
    assert sr==clip_sr==16000 and info.channels==1 and info.subtype=='PCM_16' and audio.shape==(32000,)
    assert np.array_equal(audio,original[300000:332000])
    filename=arm+'.wav';destination=HERE/filename;data=source.read_bytes()
    if destination.exists(): assert destination.read_bytes()==data
    else: destination.write_bytes(data)
    url=BASE+filename
    assert urlopen(url,timeout=10).read()==data
    head=urlopen(Request(url,method='HEAD'),timeout=10)
    assert head.status==200 and int(head.headers['Content-Length'])==len(data)
    clips.append({'file':filename,'name':labels[i]+' · 原录音18.75–20.75秒','label':labels[i]})
    manifest['sources'].append({'arm':arm,'source_wav':str(source),'source_full_wav':str(full),'raw_source':str(raw),'wav_sha256':digest(source),'full_wav_sha256':digest(full),'raw_sha256':digest(raw),'local_file':filename,'sample_rate':16000,'samples':32000,'channels':1,'subtype':'PCM_16','source_slice_pcm_exact':True,'raw_to_full_pcm_exact':True,'http_get_byte_equal':True,'http_head_passed':True})

page='''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>5243 · 局部声音对照试听</title>
<style>body{font:18px/1.65 system-ui,sans-serif;max-width:850px;margin:28px auto;padding:0 18px;color:#192734;background:#f6f8fa}section{background:white;padding:18px;margin:18px 0;border:1px solid #ccd4dc;border-radius:10px}button{font:inherit;padding:10px;margin:5px}audio{width:100%}#playerbox{position:sticky;top:0;background:#e8eff7;padding:12px;z-index:1}small{color:#465566}</style>
<h1>5243 · 试听记录（已完成）</h1>
<p><strong>本轮试听已完成，无需重新试听或提交。</strong>音频保留供日后自愿回听。</p><section><h2>最新修订（同一位听者）</h2><blockquote>__AMENDMENT__</blockquote><p><strong>当前判断：4 的 /d/ 比 3 原后端相对弱一点。</strong>这次未重新判断哪版更接近干净参考，也未确认声音删除或身份错误；5、6 的先前评估保持不变。修订来自同一位听者，不是额外一票。</p></section><section><h2>先前反馈（历史记录）</h2><blockquote>__FEEDBACK__</blockquote><p>以上为修订前的不确定判断，保留原话；3、4 的相对强弱以最新修订为准。</p></section>
<p>六版均为同一原录音 <strong>18.75–20.75 秒</strong>，完整 2 秒、原幅度、不逐条归一化。重点约在播放器 <strong>0.70 秒（原录音 19.45 秒）</strong>附近；这个位置只是识别器提供的定位提示，不是人工标注的声音边界。</p>
<div id="playerbox"><div id="current" role="status" aria-live="polite">当前片段：尚未选择</div><audio id="player" controls preload="none"></audio><button id="stop" type="button">停止</button></div>
<section><div id="clips"></div></section>
<section><h2>已完成的比较</h2><p>A 比较了 3 原后端与 4 目标附近移除条件语音后的中间短声音，以 1 干净参考为准；B 补充比较了 5 前方与 6 后方位置对照。最新修订只补充了 3、4 的相对强弱判断，未新增与干净参考接近程度的排序。</p></section>
<p><strong>原始 Y_pre 前端语音始终作为扩散锚点保留。</strong>4 仅在另一路 noisy 条件的原录音 19.35–19.55 秒附近移除已知语音成分，保留噪声；不是把整个输入或锚点语音删掉。5、6 在前方或后方位置作局部对照。这是机制诊断。</p>
<p><small>聊天反馈已记录；本页无表单，不需要再次提交。<a href="manifest_validation.json">音频来源与校验</a></small></p>
<script>
const clips = __CLIPS__;
const player = document.getElementById('player');
const current = document.getElementById('current');
let generation=0, request=null, objectURL=null;
function stopPlayback(){
 generation++;
 if(request) request.abort(); request=null;
 player.pause();player.removeAttribute('src');player.load();
 if(objectURL) URL.revokeObjectURL(objectURL);objectURL=null;
}
async function selectClip(clip){
 stopPlayback();const ticket=generation,controller=new AbortController();request=controller;
 current.textContent='正在载入：'+clip.name;
 try{
  const response=await fetch(clip.file,{signal:controller.signal});
  if(!response.ok) throw new Error('HTTP '+response.status);
  const blob=await response.blob();
  if(ticket!==generation||controller.signal.aborted)return;
  objectURL=URL.createObjectURL(blob);player.src=objectURL;
  current.textContent='当前片段：'+clip.name;await player.play();
 }catch(error){
  if(ticket!==generation||controller.signal.aborted)return;
  current.textContent='当前片段：'+clip.name+'（未自动播放，可按播放器播放；若未载入请重试）';
 }finally{if(ticket===generation)request=null;}
}
document.getElementById('stop').onclick=()=>{stopPlayback();current.textContent='已停止，请选择片段';};
clips.forEach(clip=>{const button=document.createElement('button');button.type='button';button.textContent=clip.label;button.onclick=()=>selectClip(clip);document.getElementById('clips').append(button);});
window.addEventListener('pagehide',stopPlayback);
</script></html>'''.replace('__CLIPS__',json.dumps(clips,ensure_ascii=False)).replace('__FEEDBACK__',escape(feedback['verbatim'])).replace('__AMENDMENT__',escape(amendment['verbatim']))
(HERE/'index.html').write_text(page)
assert page.count('<audio ')==1 and '<select' not in page and '<form' not in page
assert urlopen(BASE+'index.html',timeout=10).read()==page.encode()
head=urlopen(Request(BASE+'index.html',method='HEAD'),timeout=10)
assert head.status==200 and int(head.headers['Content-Length'])==len(page.encode())
assert sum((HERE/c['file']).stat().st_size for c in clips)<1_000_000
manifest['feedback_amendment_source']={'path':str(amendment_path),'sha256':digest(amendment_path),'task_status':amendment['task_status']}
manifest['feedback_source']={'path':str(feedback_path),'sha256':digest(feedback_path),'task_status':feedback['task_status']}
manifest.update({'single_audio_element':True,'no_forms_or_dropdowns':True,'page_http_byte_equal':True,'page_head_passed':True,'page_sha256':digest(HERE/'index.html'),'copied_audio_bytes':sum((HERE/c['file']).stat().st_size for c in clips)})
(HERE/'manifest_validation.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'status':'passed','url':manifest['url'],'clips':len(clips),'copied_audio_bytes':manifest['copied_audio_bytes']}))
