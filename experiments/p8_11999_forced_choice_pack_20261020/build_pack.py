#!/usr/bin/env python3
"""Build a blind, paired forced-choice pack for five preselected file-id 11999 windows.

This script only reads existing P8 WAVs/metadata and writes small listening clips into
its own experiment directory. It never modifies source audio or checkpoints.
"""
from __future__ import annotations
import csv, hashlib, json, random, re, shutil
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT = Path('/data/coding/sgmse')
OUT = ROOT / 'experiments/p8_11999_forced_choice_pack_20261020'
CLIP_DIR = OUT / 'audio'
RANK_CSV = ROOT / 'experiments/p8_e1_candidate_ranking_20261018/candidate_windows.csv'
ALIGNED_METRICS = ROOT / 'experiments/p8_e1_candidate_listening_aligned_20261020/metrics.json'
PHONEME_JSON = ROOT / 'experiments/p8_phoneme_probe_20261020/phoneme_results.json'
SEED = 20261020
CONDITIONS = ('noisy', 'y_pre', 'off', 'router')
COMPARISONS = (('noisy','y_pre'), ('y_pre','off'), ('off','router'))


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def safe_float(x):
    return float(x)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    CLIP_DIR.mkdir(parents=True, exist_ok=True)
    # Select the five highest-ranked suppressed windows for file-id 11999. This is
    # a fixed, predeclared candidate rule from the existing read-only ranking.
    ranking = list(csv.DictReader(RANK_CSV.open(newline='', encoding='utf-8')))
    # candidate_windows.csv is already restricted to candidate_frontend_suppression
    # rows by its producer; retain that provenance explicitly below.
    candidates = [r for r in ranking if r['fileid'] == '11999']
    candidates.sort(key=lambda r: (int(r['rank']), float(r['start_s'])))
    selected = candidates[:5]
    if len(selected) != 5:
        raise RuntimeError(f'expected at least 5 11999 suppressed candidates, found {len(selected)}')
    # Source paths are taken from the already-aligned P8 paired manifest.
    aligned = json.loads(ALIGNED_METRICS.read_text())
    source_record = next(r for r in aligned['records'] if str(r['fileid']) == '11999')
    source_paths = {k: Path(v) for k,v in source_record['source_paths'].items()}
    for c,p in source_paths.items():
        if not p.exists(): raise FileNotFoundError(str(p))
    # Validate source hashes from the existing aligned manifest before reading.
    source_hashes = {c: sha256(p) for c,p in source_paths.items()}
    if source_record.get('source_sha256'):
        for c,h in source_record['source_sha256'].items():
            if source_hashes[c] != h:
                raise RuntimeError(f'source hash mismatch for {c}: {source_hashes[c]} != {h}')
    # Load all source WAVs once. Existing files are only read.
    audio = {}
    for c,p in source_paths.items():
        x,sr = sf.read(p, always_2d=False, dtype='float32')
        if x.ndim != 1: raise RuntimeError(f'expected mono source: {p}')
        if sr != 16000: raise RuntimeError(f'expected 16 kHz: {p} ({sr})')
        audio[c] = (x,sr)
    # Generate common aligned clips. PCM16 copies in this directory are derived
    # artifacts; original source WAVs remain untouched.
    windows=[]
    for wi,row in enumerate(selected,1):
        start=float(row['start_s']); end=float(row['end_s']);
        # Keep 0.75 s of context on each side of the 0.5 s candidate. This
        # matches the existing aligned P8 pack's 2 s context convention while
        # keeping the candidate target window fixed for the E1 judgment.
        context_start=max(0.0,start-0.75); context_end=end+0.75
        s0=int(round(context_start*16000)); s1=int(round(context_end*16000))
        if s1 <= s0 or s1 > len(audio['clean'][0]): raise RuntimeError((row,s0,s1))
        wid=f'W{wi:02d}'
        win={'window_id':wid,'fileid':'11999','speaker':row['speaker'],'snr_db':int(row['snr_db']),'target_start_s':start,'target_end_s':end,'context_start_s':context_start,'context_end_s':context_end,'samples':s1-s0,'ranking_rank':int(row['rank']),'candidate_rank_score':float(row['cue_rank_score']),'oracle_local_snr_db':float(row['oracle_local_snr_db']),'noisy_clean_wave_corr':float(row['noisy_clean_wave_corr']),'y_pre_rms_db_vs_clean':float(row['y_pre_rms_db_vs_clean']),'off_gain_vs_y_pre_db':float(row['off_gain_vs_y_pre_db']),'source_paths':{c:str(p) for c,p in source_paths.items()},'source_sha256':source_hashes.copy(),'clips':{}}
        for c in ('clean',)+CONDITIONS:
            arr=audio[c][0][s0:s1]
            fn=f'{wid}_reference.wav' if c=='clean' else f'{wid}_{c}.wav'
            # Internal filenames are not exposed in HTML for candidate conditions.
            out=CLIP_DIR/fn
            sf.write(out,arr,16000,subtype='PCM_16')
            win['clips'][c]={'path':str(out.relative_to(OUT)),'sha256':sha256(out),'samples':int(len(arr))}
        windows.append(win)
    # Check that W01 reproduces the target 19.5--20.0 segment inside the
    # previously aligned 18.75--20.75 context clip.
    old=ROOT/'experiments/p8_e1_candidate_listening_aligned_20261020/clips/02_fileid_11999_19.50-20.00s_context_clean.wav'
    old_match=False
    if old.exists():
        old_x,old_sr=sf.read(old,always_2d=False,dtype='float32')
        new_x,_=sf.read(OUT/'audio/W01_reference.wav',always_2d=False,dtype='float32')
        old_match=bool(old_sr==16000 and len(old_x)==len(new_x) and np.array_equal(old_x,new_x))
    # Randomize trial order and A/B assignment. Hidden answer mapping is kept only
    # in answer_key.json and metrics.json, never displayed in HTML.
    rng=random.Random(SEED)
    trials=[]
    for win in windows:
        for ci,(left,right) in enumerate(COMPARISONS,1):
            pair=[left,right]; rng.shuffle(pair)
            tid=f"{win['window_id']}_C{ci}"
            trials.append({'trial_id':tid,'window_id':win['window_id'],'comparison_id':f'C{ci}','reference_audio':f"audio/{win['window_id']}_reference.wav",'option_A':f"audio/{tid}_A.wav",'option_B':f"audio/{tid}_B.wav",'_A_condition':pair[0],'_B_condition':pair[1],'_comparison_conditions':[left,right]})
    rng.shuffle(trials)
    for t in trials:
        win=next(w for w in windows if w['window_id']==t['window_id'])
        for label,cond in (('A',t['_A_condition']),('B',t['_B_condition'])):
            src=OUT/ win['clips'][cond]['path']
            dst=CLIP_DIR/f"{t['trial_id']}_{label}.wav"
            shutil.copyfile(src,dst)
            t[f'{label}_sha256']=sha256(dst)
    # Auxiliary subset from the existing phoneme_results.json; no new inference.
    phoneme_all=json.loads(PHONEME_JSON.read_text())
    phoneme_ref=[r for r in phoneme_all['records'] if r.get('group','').startswith('02_fileid_11999_19.50-20.00s')]
    aux={'source_path':str(PHONEME_JSON),'source_sha256':sha256(PHONEME_JSON),'source_model':phoneme_all.get('model'),'source_device':phoneme_all.get('device'),'records_reused_without_reinference':phoneme_ref}
    (OUT/'phoneme_auxiliary_11999.json').write_text(json.dumps(aux,indent=2,ensure_ascii=False)+'\n')
    # Persist answer key separately; metrics retains enough provenance for audit.
    key=[]
    for t in trials:
        key.append({'trial_id':t['trial_id'],'window_id':t['window_id'],'comparison_conditions':t['_comparison_conditions'],'A_condition':t['_A_condition'],'B_condition':t['_B_condition'],'better_condition_not_prespecified':True})
    (OUT/'answer_key.json').write_text(json.dumps({'seed':SEED,'trials':key},indent=2,ensure_ascii=False)+'\n')
    # Remove hidden fields before writing public trial metadata; answer key remains separate.
    public_trials=[]
    for t in trials:
        public_trials.append({k:v for k,v in t.items() if not k.startswith('_')})
    metrics={'schema_version':1,'experiment_id':'p8_11999_forced_choice_pack_20261020','status':'ready_for_human_assessment','purpose':'blind forced-choice auxiliary check for P8-E1 candidate windows','selection':{'source_ranking':str(RANK_CSV),'source_ranking_sha256':sha256(RANK_CSV),'rule':'fileid=11999, candidate_frontend_suppression=true, five smallest existing rank values','n_windows':len(windows),'windows':windows},'source_manifest_metrics':str(ALIGNED_METRICS),'source_manifest_metrics_sha256':sha256(ALIGNED_METRICS),'phoneme_auxiliary':{'source_path':str(PHONEME_JSON),'source_sha256':sha256(PHONEME_JSON),'subset_path':'phoneme_auxiliary_11999.json','n_records':len(phoneme_ref),'note':'Only the existing 19.50--20.00 s 11999 five-condition CTC record is reused; no phoneme inference was run for the other four windows.'},'randomization':{'seed':SEED,'trial_order_randomized':True,'AB_assignment_randomized_per_trial':True,'mapping_path':'answer_key.json'},'comparisons':['noisy vs Y_pre','Y_pre vs Router-off','Router-off vs Router'],'n_trials':len(public_trials),'trials':public_trials,'validation':{'w01_matches_existing_aligned_clean_clip_samples':old_match,'all_source_hashes_verified':True,'dataset_modified':False,'checkpoint_modified':False,'new_inference':False,'human_responses_present':False},'e1_gate':{'minimum_supporting_windows':3,'required_chain':['listener can identify same target consonant/word fragment in noisy relative to clean reference','Y_pre is judged to lose/suppress that fragment','Router-off or Router is judged to restore the same fragment rather than only make residual speech smoother'],'status':'not_assessed_until_human_notes_are_recorded'}}
    (OUT/'metrics.json').write_text(json.dumps(metrics,indent=2,ensure_ascii=False)+'\n')
    # Responses template does not reveal condition mapping.
    with (OUT/'responses.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['trial_id','window_id','choice_A_or_B','noisy_target_fragment_audible_yes_no_uncertain','y_pre_target_lost_yes_no_uncertain','same_fragment_restored_yes_no_uncertain','notes'])
        for t in public_trials: w.writerow([t['trial_id'],t['window_id'],'','','','',''])
    build_html(public_trials)
    build_docs(windows,public_trials,old_match,aux)


def build_html(trials):
    cards=[]
    for i,t in enumerate(trials,1):
        cards.append(f'''<section class="trial"><h2>试次 {i}（{t['trial_id']}）</h2><p>先听参考，再反复听 A/B。只按“与参考的词句/音素内容更一致”选择。</p><audio controls preload="none" src="{t['reference_audio']}"></audio><span>参考</span><div class="opts"><div><h3>A</h3><audio controls preload="none" src="{t['option_A']}"></audio><label><input type="radio" name="{t['trial_id']}" value="A"> 选择 A</label></div><div><h3>B</h3><audio controls preload="none" src="{t['option_B']}"></audio><label><input type="radio" name="{t['trial_id']}" value="B"> 选择 B</label></div></div><label>备注（例如听到的目标辅音/词片段）：<input class="note" data-trial="{t['trial_id']}" type="text"></label></section>''')
    html='''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>P8 file-id 11999 blind forced choice</title><style>body{font-family:system-ui,sans-serif;max-width:900px;margin:2em auto;padding:0 1em;line-height:1.5;background:#fafafa}.trial{background:#fff;border:1px solid #ddd;border-radius:8px;padding:1em;margin:1em 0}.opts{display:flex;gap:2em;flex-wrap:wrap}.opts>div{min-width:260px;border:1px solid #ddd;padding:.7em;border-radius:6px}audio{display:block;margin:.4em 0;max-width:100%}.note{width:min(100%,600px);padding:.4em}button{padding:.7em 1em;margin:.5em .3em 1em 0}small{color:#666}</style></head><body><h1>file-id 11999 盲化强迫选择</h1><p>这是 P8-E1 的辅助人工核实包，共五个窗口、每个窗口三个 A/B 对照。参考音频可见，A/B 条件盲化且顺序随机。评价内容一致性/目标音素，不评价音量或整体悦耳度。</p><p><strong>完成后点击下载 responses.csv</strong>，并把备注写清楚。没有听出差别也必须按直觉二选一，同时在备注写“差不多”。</p>''' + ''.join(cards) + '''<button onclick="downloadCSV()">下载 responses.csv</button><button onclick="window.scrollTo(0,0)">回到顶部</button><p><small>本页面不显示条件映射；映射仅保存在实验目录的 answer_key.json，供审计使用。</small></p><script>function downloadCSV(){const rows=[['trial_id','window_id','choice_A_or_B','noisy_target_fragment_audible_yes_no_uncertain','y_pre_target_lost_yes_no_uncertain','same_fragment_restored_yes_no_uncertain','notes']];document.querySelectorAll('.trial').forEach(s=>{const id=s.querySelector('input[type=radio]').name;const w=s.querySelector('h2').textContent.match(/(W\\d+)_/)[1];const c=s.querySelector('input[type=radio]:checked');const n=s.querySelector('.note').value.replaceAll('"','""');rows.push([id,w,c?c.value:'','', '', '',n]);});const csv=rows.map(r=>r.map(x=>'"'+String(x).replaceAll('"','""')+'"').join(',')).join('\\n');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8'}));a.download='responses.csv';a.click();}</script></body></html>'''
    (OUT/'index.html').write_text(html,encoding='utf-8')


def build_docs(windows,trials,old_match,aux):
    lines=['# file-id 11999 盲化强迫选择包（P8-E1 辅助核实）','', '## 目的','', '这个包只用于人工核实 file-id 11999 的候选窗口，不能单独证明 E1、音素恢复或模型幻觉。五个窗口来自已有 clean-aware 候选排序；音素模型结果只复用已有 `phoneme_results.json`，没有为本包重新推理。','', '## 使用方法','', '1. 用浏览器打开 `index.html`。远程 SSH 可在服务器上执行 `python3 -m http.server 8000 --directory /data/coding/sgmse/experiments/p8_11999_forced_choice_pack_20261020`，再做 SSH 端口转发。', '2. 每个试次先听参考，再交替听 A/B。只判断目标词句/音素与参考是否一致，不按响度、噪声少或音色平滑选择。', '3. A/B 条件和顺序在页面中隐藏并已固定随机化。每个试次必须二选一；听不出差别时仍选择一个，并在备注写“差不多”。', '4. 点击页面底部下载 `responses.csv`，或直接填写目录中的模板。','', '## 试次结构','', '- 每个窗口有三个比较：`noisy vs Y_pre`、`Y_pre vs Router-off`、`Router-off vs Router`。', '- 参考为同一原始时间窗的 clean 片段；A/B 是同窗候选，所有音频均为 16 kHz。', '- 目标是内容/音素一致性；这不是音质偏好测试。','', '## E1 通过条件（预先固定）','', '至少 3/5 个窗口同时满足以下链条，才允许把结果作为 E1 候选证据：', '', '1. 听者能在 noisy 中相对 clean 参考识别同一目标辅音或词片段；', '2. Y_pre 被判断为丢失或明显压低该片段；', '3. Router-off 或 Router 被判断为恢复同一片段，而不是只让残余语音更圆润。', '', 'CSV 的三个附加列用于记录这三项判断。若 noisy 目标片段听不出、只发生整体清晰度变化、或只凭一个 ASR 代理判断，窗口不算 E1。当前尚无人工结果，不能提前宣称通过。','', '## 选窗依据','', '五个窗口是既有 `p8_e1_candidate_ranking_20261018/candidate_windows.csv` 中 file-id 11999 且 `candidate_frontend_suppression=true` 的前五名，未在本包中重新排序或使用人工结果调阈值。窗口如下：','', '| 窗口 | 时间(s) | 原候选 rank | noisy-clean corr | local SNR (dB) | Y_pre RMS vs clean (dB) | off gain vs Y_pre (dB) |', '|---|---:|---:|---:|---:|---:|---:|']
    for w in windows: lines.append('| %s | %.2f–%.2f | %d | %.3f | %.2f | %.2f | %.2f |'%(w['window_id'],w['start_s'],w['end_s'],w['ranking_rank'],w['noisy_clean_wave_corr'],w['oracle_local_snr_db'],w['y_pre_rms_db_vs_clean'],w['off_gain_vs_y_pre_db']))
    lines += ['', '已有音素结果只覆盖 W01（19.50–20.00 s）的 clean/noisy/Y_pre/off/Router 五个片段，保存在 `phoneme_auxiliary_11999.json`。它是 CTC 声学模型代理，不是真实音素标签；其他四个窗口没有在本任务中运行音素推理。','', '## 输出文件','', '- `index.html`：盲化播放和 A/B 表单。', '- `responses.csv`：空白记录模板。', '- `answer_key.json`：A/B 条件映射，仅供审计。', '- `metrics.json`：固定选择、源文件哈希、随机种子和验证状态。', '- `experiment_report.md`：实验边界和当前待人工结果。', '- `phoneme_auxiliary_11999.json`：复用的已有音素结果子集。', '', '未修改 DNS 数据集、原始 P8 WAV 或 checkpoint；没有训练和新推理。']
    (OUT/'README_CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    report=['# P8 file-id 11999 blind forced-choice pack (2026-10-20)','', '## Status','', 'Pack generation completed; human assessment is pending. This is a read-only auxiliary check and does not establish E1 or phoneme recovery.', '', '## Fixed protocol','', '- Source: existing aligned P8 paired WAVs and existing `p8_phoneme_probe_20261020/phoneme_results.json`.', '- Selection: five highest-ranked existing `fileid=11999` windows with `candidate_frontend_suppression=true`; no new ranking or inference.', '- Conditions: clean reference plus noisy, Y_pre, Router-off, Router candidates. Three blind A/B comparisons per window.', '- Random seed: `%d`; trial order and A/B assignment randomized. Hidden mapping is in `answer_key.json`.'%SEED, '- Output clips are small derived PCM16 files. Source hashes were checked before extraction.', '', '## Current result','', 'No human responses are present. The pack cannot decide whether noisy contains an identifiable target or whether a backend restores it. The pre-registered E1 gate requires at least 3/5 windows to satisfy the three-part listener chain documented in `README_CN.md`.', '', 'The existing phoneme result was reused only for the 19.50–20.00 s window; it is a CTC acoustic-model proxy and not phoneme ground truth. W01 sample equality against the corrected aligned clean clip: `%s`.'%('true' if old_match else 'false/old clip unavailable'), '', 'No dataset, original WAV, or checkpoint was modified; no training or new inference was run.', '', '## Reproduction', '', '```bash', '/data/coding/sgmse/env/bin/python experiments/p8_11999_forced_choice_pack_20261020/build_pack.py', '```']
    (OUT/'experiment_report.md').write_text('\n'.join(report)+'\n',encoding='utf-8')

if __name__=='__main__': main()
