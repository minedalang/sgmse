"""Rebuild only optional listening artifacts from frozen existing WAVs; no inference."""
import ast
import hashlib
import io
import json
from pathlib import Path
import struct
from urllib.request import urlopen
import wave

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
REPO = EXP.parent.parent
BASE = 'http://localhost:8766/' + EXP.name + '/listening/'
IDS = ['original__N1_snr-6__m2ada1', 'fresh__N1_snr-6__f20aka1', 'original__N2_snr-6__m2aga1', 'original__N1_snr-6__f11ata1', 'fresh__N2_snr-6__f20aka1']
sha = lambda b: hashlib.sha256(b).hexdigest()
metrics = json.loads((EXP / 'metrics.json').read_text())
selection = json.loads((EXP / 'selection_frozen.json').read_text())
manual = json.loads((HERE / 'manual_listener_feedback_20261019.json').read_text())
completion_path = HERE / 'manual_listener_feedback_completion_20261019.json'
completion = json.loads(completion_path.read_text())
manual_responses = manual['responses'] + completion['responses']
manual_choices = {r['case_id']: r['choice_verbatim'] for r in manual_responses}
manual_comments = {r['case_id']: r.get('comment_verbatim', '') for r in manual_responses}
report = {'url': BASE + 'index.html', 'amplitude': 'original; byte copies; no per-clip normalization', 'cases': [], 'files': [], 'source_documents': {}}

def document(path):
    report['source_documents'][str(path)] = sha(path.read_bytes())

document(EXP / 'metrics.json')
document(EXP / 'selection_frozen.json')
for case_id in IDS:
    frozen = next(x for x in selection['cases'] if x['case_id'] == case_id)
    comp = next(x for x in metrics['comparisons'] if x['case_id'] == case_id)
    assert (comp['native_off_prediction'] != comp['ablated_prediction']) == (case_id in IDS[:3])
    src = REPO / frozen['source_experiment']
    manifest_path = src / 'outputs/audio_manifest.json'
    document(manifest_path)
    source_rows = json.loads(manifest_path.read_text())
    clean = next(x for x in source_rows if x['condition'] == frozen['condition'] and x['sample_id'] == frozen['sample_id'] and x['arm'] == 'clean')
    frontend = frozen['frontend_audio']
    assert any(x['raw_sha256'] == frontend['raw_sha256'] and x['wav_sha256'] == frontend['wav_sha256'] and x['wav_path'] == frontend['wav_path'] for x in source_rows)
    native = next(x for x in metrics['cases'] if x['case_id'] == case_id and not x['ablated'])
    ablated = next(x for x in metrics['cases'] if x['case_id'] == case_id and x['ablated'])
    assert native['raw_sha256'] == frozen['off_audio']['raw_sha256']
    clips = []
    for arm, root, row, label in [('clean', src, clean, '干净参考'), ('y_pre', src, frontend, '前端 Y_pre'), ('native', EXP, native, '原始后端 native'), ('ablated', EXP, ablated, '后端（移除 noisy 条件中的语音）')]:
        wav_path, raw_path = root / row['wav_path'], root / row['raw_path']
        data, raw = wav_path.read_bytes(), raw_path.read_bytes()
        assert sha(data) == row['wav_sha256']
        assert sha(raw) == row['raw_sha256']
        if str(wav_path) in metrics['sha256_locks']:
            assert sha(data) == metrics['sha256_locks'][str(wav_path)]
        if str(raw_path) in metrics['sha256_locks']:
            assert sha(raw) == metrics['sha256_locks'][str(raw_path)]
        assert raw[:6] == b'\x93NUMPY'
        major = raw[6]
        n = 2 if major == 1 else 4
        hlen = int.from_bytes(raw[8:8+n], 'little')
        header = ast.literal_eval(raw[8+n:8+n+hlen].decode().strip())
        assert header['shape'] in [(32000,), (1, 32000)] and header['descr'] == '<f4'
        assert sha(raw[8+n+hlen:]) == row['float32_sha256']
        with wave.open(io.BytesIO(data), 'rb') as w:
            assert (w.getframerate(), w.getnframes(), w.getnchannels(), w.getsampwidth()) == (16000, 32000, 1, 2)
        filename = case_id + '__' + arm + '.wav'
        if (HERE / filename).exists():
            assert (HERE / filename).read_bytes() == data
        else:
            (HERE / filename).write_bytes(data)
        fetched = urlopen(BASE + filename, timeout=10).read()
        assert fetched == data
        clips.append({'name': case_id + ' · ' + label, 'label': label, 'file': filename})
        report['files'].append({'case_id': case_id, 'arm': arm, 'source_wav': str(wav_path), 'source_raw': str(raw_path), 'wav_sha256': sha(data), 'raw_sha256': sha(raw), 'float32_sha256': row['float32_sha256'], 'local_file': filename, 'sample_rate': 16000, 'samples': 32000, 'http_byte_equal': True, 'raw_and_wav_hashes_valid': True})
    report['cases'].append({'case_id': case_id, 'manual_chat_choice': manual_choices.get(case_id), 'manual_chat_comment': manual_comments.get(case_id), 'proxy_native_label': comp['native_off_prediction'], 'proxy_ablated_label': comp['ablated_prediction'], 'clips': clips})

page = '''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>五例可选试听</title>
<style>body{font:17px/1.65 system-ui,sans-serif;max-width:880px;margin:32px auto;padding:0 18px;color:#172432;background:#f6f8fa}section{background:white;padding:18px;margin:18px 0;border:1px solid #ccd4dc;border-radius:10px}button,select{font:inherit;margin:5px;padding:8px 12px}audio{width:100%}#playerbox{position:sticky;top:0;background:#e8eff7;padding:12px;border-radius:8px;z-index:1}small{color:#4d5b69}</style>
<h1>五例可选试听</h1>
<p>本轮五例试听已完成并通过聊天确认。比较聚焦中间辅音是否接近干净参考，不以整体清晰度或噪声大小替代该判断；具体听到的尾音另列于备注。无需再次试听或提交。若日后自愿更新网页选项，按“保存选择”并收到服务器确认后，该次网页更新才算保存。</p>
<p><strong>仍保留原始 Y_pre 前端语音作为扩散锚点；只移除另一路 noisy 条件中的语音成分，保留原噪声。因此输出仍可有清晰语音，noise-only 不是输出内容。</strong>这是诊断消融，不是正常增强设置。模型类别只是代理判断，不等于听感真值。</p>
<p>全部为对齐的完整 2 秒，16 kHz；默认原幅度，无逐条归一化。请保持系统音量一致。可先听干净参考，再比较另外三版。</p>
<div id="playerbox"><div id="current" role="status" aria-live="polite">当前片段：尚未选择</div><audio id="player" controls preload="none"></audio><button id="stop" type="button">停止</button></div>
<p><strong>五例试听反馈均已按聊天回复单独记录，本轮试听已完成，无需重听或重新填写。页面保留已有音频与保存功能，便于日后自愿更新。</strong>本页共五例来自既有前后端类别正确性转换，仍是选定诊断样本，不代表随机抽样或总体收益。聊天记录与本页保存记录分别保留来源。</p>
<main id="cases"></main>
<section><h2>日后自愿更新（本轮无需操作）</h2><label for="feedback-note">备注（可选，最多 2000 字）</label><br><textarea id="feedback-note" maxlength="2000" rows="3" style="width:95%"></textarea><br><button id="save-feedback" type="button">保存选择</button><p id="feedback-status" role="status" aria-live="polite">网页更新尚未保存；已确认的聊天反馈不受影响。</p><small>选项变化会保留在本浏览器草稿中。只有显示“已保存到服务器”才表示已提交；失败时可重试。记录按提示后、非盲、单人试听反馈保存。<a href="validation.json">来源与校验记录</a></small></section>
<script>
const cases = __CASES__;
const player = document.getElementById('player');
const current = document.getElementById('current');
let generation = 0, request = null, objectURL = null;
function stopPlayback() {
  generation++;
  if (request) request.abort();
  request = null;
  player.pause();
  player.removeAttribute('src');
  player.load();
  if (objectURL) URL.revokeObjectURL(objectURL);
  objectURL = null;
}
async function selectClip(clip) {
  stopPlayback();
  const ticket = generation;
  const controller = new AbortController();
  request = controller;
  current.textContent = '正在载入：' + clip.name;
  try {
    const response = await fetch(clip.file, {signal: controller.signal});
    if (!response.ok) throw new Error('HTTP ' + response.status);
    const blob = await response.blob();
    if (ticket !== generation || controller.signal.aborted) return;
    objectURL = URL.createObjectURL(blob);
    player.src = objectURL;
    current.textContent = '当前片段：' + clip.name;
    await player.play();
  } catch (error) {
    if (ticket !== generation || controller.signal.aborted) return;
    current.textContent = '当前片段：' + clip.name + '（未自动播放，可按播放器播放；若未载入请重试）';
  } finally {
    if (ticket === generation) request = null;
  }
}
document.getElementById('stop').onclick = () => { stopPlayback(); current.textContent = '已停止，请选择片段'; };
cases.forEach((item, i) => {
  const section = document.createElement('section');
  const title = document.createElement('h2'); title.textContent = (i+1) + '. ' + item.case_id; section.append(title);
  if (item.manual_chat_choice) { const recorded = document.createElement('p'); recorded.textContent = '聊天记录确认：' + item.manual_chat_choice + '（已单独保存，无需重听或重复填写）'; section.append(recorded); }
  if (item.manual_chat_comment) { const comment = document.createElement('p'); comment.textContent = '聊天备注（原话）：' + item.manual_chat_comment; section.append(comment); }
  item.clips.forEach(clip => { const b = document.createElement('button'); b.type = 'button'; b.textContent = clip.label; b.onclick = () => selectClip(clip); section.append(b); });
  const question = document.createElement('p'); question.textContent = '中间辅音哪一版更接近干净参考（不比较噪声大小或整体清晰度）？'; section.append(question);
  const answer = document.createElement('select'); answer.setAttribute('aria-label', item.case_id + ' 试听判断'); answer.setAttribute('data-case-id', item.case_id);
  [['unrated','可选：暂不判断'],['native','原始后端更接近'],['ablated','移除条件语音的后端更接近'],['similar','两者差不多'],['uncertain','听不清／不确定']].forEach(([value,text]) => { const option = document.createElement('option'); option.value = value; option.textContent = value === 'unrated' && item.manual_chat_choice ? '聊天已记录，无需重复填写' : text; answer.append(option); }); section.append(answer);
  const details = document.createElement('details'); const summary = document.createElement('summary'); summary.textContent = '查看模型代理标签（可先听再展开）'; details.append(summary); const p = document.createElement('p'); p.textContent = '原始后端 ' + item.proxy_native_label + ' → 移除条件语音的后端 ' + item.proxy_ablated_label + '；仅模型代理，非听感结论。'; details.append(p); section.append(details);
  document.getElementById('cases').append(section);
});
window.addEventListener('pagehide', stopPlayback);
</script><script src="feedback.js"></script></html>'''.replace('__CASES__', json.dumps(report['cases'], ensure_ascii=False))
(HERE / 'index.html').write_text(page)
assert page.count('<audio ') == 1
assert urlopen(BASE + 'index.html', timeout=10).read() == (HERE / 'index.html').read_bytes()
report.update({'status': 'passed', 'wav_count': len(IDS) * 4, 'all_http_byte_equal': True, 'page_http_byte_equal': True, 'single_audio_element': True, 'alignment': 'untrimmed existing full2s carrier; original manifests; no resampling or amplitude changes'})
(HERE / 'validation.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
print(json.dumps({'status': report['status'], 'url': report['url'], 'wav_count': len(IDS) * 4}, ensure_ascii=False))
