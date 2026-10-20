"""Read-only artifact review. Writes inventories only; never deletes or moves."""
from pathlib import Path
import os,stat,json,csv,datetime,shutil
ROOT=Path('/data/coding/sgmse');OUT=Path(__file__).resolve().parent;JT=ROOT/'joint_train_logs';EL=ROOT/'experiment_logs';EX=ROOT/'experiments'
groups=[];files=[];seen=set()
def regulars(base,suffix):
    got=[]
    for d,dirs,names in os.walk(base,followlinks=False):
        for name in names:
            p=Path(d)/name;s=p.lstat()
            if stat.S_ISREG(s.st_mode) and p.suffix.lower()==suffix:got.append(p)
    return sorted(got)
def add(gid,level,title,paths,reason,cost,keep='',evidence=None):
    rows=[]
    for p in sorted(paths):
        p=Path(p);s=p.lstat();assert stat.S_ISREG(s.st_mode) and str(p).startswith('/data/') and str(p) not in seen
        seen.add(str(p));r=dict(group_id=gid,level=level,path=str(p),logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,
            device=s.st_dev,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns,decision='NOT_APPROVED_DO_NOT_DELETE')
        rows.append(r);files.append(r)
    groups.append(dict(id=gid,level=level,title=title,count=len(rows),logical_bytes=sum(r['logical_bytes'] for r in rows),
        allocated_bytes=sum(r['allocated_bytes'] for r in rows),reason=reason,deletion_cost=cost,keep=keep,evidence=evidence or []))
def extras(names,steps):
    return [JT/n/f for n,step in zip(names,steps) for f in (f'epoch=0-step={step}-last.ckpt','last.ckpt')]
speaker=['5l5cb2a0-rrsd_speaker_pair_masked1000','ftyjnqu1-rrsd_speaker_pair1000','kn1hgnni-rrsd_speaker_svproto_masked_sw05_1000','q2ilpruq-rrsd_speaker_proto01_cw1_1000','qpf32q6y-rrsd_speaker_proto01_1000']
common_reason='已结束的旧短试额外保存件；没有当前CPEL依赖。未核张量等价，不能称字节相同副本。'
common_cost='删除该保存件后不能用它精确恢复；即便保留同一步代表权重，也未证明二者恢复状态相同。'
def add_extra(gid,title,names,steps,evidence):
    add(gid,'A_优先审查',title,extras(names,steps),common_reason,common_cost,
        '各目录不带-last的epoch=0-step=<步数>.ckpt及全部config/log/CSV/report保留。',evidence)
add_extra('A01','rrsd_speaker额外10份last保存件',speaker,[1000]*5,['/data/research/archive.md',str(EX/'speaker_parameter_audit_20261017/experiment_report.md'),str(EX/'rrsd_cv_repeat32/experiment_report.md')])
add_extra('A02','SSL/WavLM旧小试额外保存件',['f6q35kgy-rrsd_ssl_rel_probe20','ipyj136o-rrsd_ssl_mse_smoke4_b2','p9lnw5px-rrsd_ssl_mse_1000','sgk1xb6q-rrsd_ssl_mse_w1_1000','woe2707y-rrsd_ssl_rel_m0_w1_1000_s2026'],[20,4,1000,1000,1000],['/data/research/archive.md'])
add_extra('A03','旧conflict/gradient probe额外保存件',['7zj73436-conflict1000','d1gyntzs-gradprobe_conflict_w0002'],[1000,20],['/data/research/archive.md'])
add_extra('A04','旧projection smoke/gate额外保存件',['cf69f1wr-rrsd_gate_projection_smoke4','d1z5ovza-rrsd_projection_gate_1000'],[4,1000],['/data/research/archive.md'])
add_extra('A05','旧k-step/debug额外保存件',['39f0668w-rrsd_kstep_debug1','mdkln6oq-rrsd_kstep_k4_1000_s2026','mp2dgxin-rrsd_kstep_k4_probe20_s2026'],[1,1000,20],['/data/research/archive.md'])
add_extra('A06','旧control1000额外保存件',['acg99gpj-control1000'],[1000],['/data/research/archive.md'])
a=JT/'anchor_v2_controller100/0hduhemo-anchor_v2_controller100'
add('A07','A_优先审查','Anchor V2旧小试额外保存件',[a/'epoch=0-step=100-last.ckpt',a/'last.ckpt'],common_reason,common_cost,str(a/'epoch=0-step=100.ckpt')+'保留。',['/data/research/archive.md'])
b=EX/'speaker_backend_only_matched_500_20261018/logs/None-speaker_backend_only_matched500_control'
add('A08','A_优先审查','旧speaker backend-only对照额外last',[b/'last.ckpt'],'报告引用另一epoch文件；既有SHA记录表明两件不同，不能称相同副本。',common_cost,str(b/'epoch=0-step=500-last.ckpt')+'及checkpoint_sha256.txt保留。',[str(EX/'speaker_backend_only_matched_500_20261018/experiment_report.md')])
wav_names=['rrsd_speaker_eval','rrsd_speaker_eval_svproto_masked_sw05_1000','rrsd_ssl_mse_eval','rrsd_ssl_rel_eval','rrsd_kstep_k4_eval','rrsd_tcut005_scale075_screen_20261017','rrsd_tcut005_scale05_screen_20261017','rrsd_tcut005_scale025_screen_20261017','rrsd_scale05_screen_20261017','rrsd_scale025_screen_20261017']
for i,name in enumerate(wav_names,1):
    add(f'W{i:02d}','A_旧验证WAV候选',name+'：仅WAV',regulars(EL/name,'.wav'),'旧loss/scale开发小试已经结束或暂停，不是当前CPEL训练源或正式final输出。','丢失直接试听、重算新指标和波形级追查能力；未证明可逐字节再生成。','只选清单中的.wav，保留目录内CSV/config/log/report及基座off/router输出。',['/data/research/archive.md'])
add('T01','A_低收益临时文件','超时失败的不完整下载包',[EX/'p8_keyword_clean_readout_20261020/mini_speech_commands.zip'],'完整包下载超时，只保存11.53MB；实际64条后来通过独立Range取得，此partial不用于模型。','丢失失败下载的原始partial字节；失败报告、SHA、receipt及已取得64条保留。','保留同目录报告/指标/代码与transport_attempt02全部材料。',[str(EX/'p8_keyword_clean_readout_20261020/experiment_report.md')])
add('B01','B_放弃直接复评才考虑','rrsd_speaker每组保留的代表权重',[JT/n/'epoch=0-step=1000.ckpt' for n in speaker],'若用户决定不再直接复评这5条旧speaker路线，可单独审查。','若连A01也删除，则5组共15份权重全消失；旧路线不能直接模型复评或恢复。','建议本轮先保留；A01已足够释放长训所需空间。',['/data/research/archive.md'])
a=JT/'anchor_v2_controller100_retry/None-anchor_v2_controller100_retry'
add('B02','B_放弃直接复评才考虑','Anchor retry全部两份权重',[a/'epoch=0-step=100-last.ckpt',a/'last.ckpt'],'旧路线已退出，但这里只剩两件，不是保留代表后的额外件。','删除后此retry运行没有任何权重可供直接复评/恢复。','建议先保留。',['/data/research/archive.md'])
add('B03','B_放弃直接复评才考虑','speaker backend-only报告所用权重',[b/'epoch=0-step=500-last.ckpt'],'旧500步控制已结束，只有明确放弃直接复评时才考虑。','与A08一起删除会失去该完整控制的全部权重。','建议本轮先保留。',[str(EX/'speaker_backend_only_matched_500_20261018/experiment_report.md')])
optional=[('V01','speaker_backend_only_matched_500_20261018/control_eval_64/control','旧speaker no-loss控制64-speaker输出'),('V02','rollout_adapter_pilot_20261018/fresh_cv_audio','旧rollout pilot输出'),('V03','rollout_adapter_replication_20261018/audio','旧rollout复验输出'),('V04','reverb_cv_smoke_20261018','旧混响smoke临时生成及输出'),('V05','p3b_blind_listening_pilot_20261017/playback/audio','已完成P3b盲听匿名副本')]
for gid,rel,title in optional:
    if gid=='V04':paths=regulars(EX/rel/'inputs','.wav')+regulars(EX/rel/'output','.wav')
    else:paths=regulars(EX/rel,'.wav')
    add(gid,'B_可再生成但证据有损',title+'：仅WAV',paths,'已结束旧实验；V05原报告明确是从保留源输出复制的试听副本。','丢失对应直接试听/指标复评；V05会使历史盲听页面音频链接失效。','保留报告、答卷、answer_key、CSV、代码和源基线。',[str(EX/rel.split('/')[0]/'experiment_report.md')])
assert len([r for r in files if r['group_id'].startswith('A')])==39
assert len([r for r in files if r['group_id'].startswith('W')])==448
assert all(r['nlink']==1 for r in files)
inodes={(r['device'],r['inode']) for r in files};assert len(inodes)==len(files)
# Snapshot all project artifact weights, not all /data/datasets or model-cache weights.
weights=[]
for prefix in ('joint_train_logs','experiment_logs','experiments','lightning_logs'):
 for d,ds,fs in os.walk(ROOT/prefix,followlinks=False):
  for name in fs:
   p=Path(d)/name;s=p.lstat()
   if stat.S_ISREG(s.st_mode) and p.suffix.lower() in {'.ckpt','.pt','.pth','.safetensors'}:
    weights.append(dict(path=str(p),logical_bytes=s.st_size,allocated_bytes=s.st_blocks*512,nlink=s.st_nlink,candidate=str(p) in seen))
with (OUT/'candidate_files.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(files[0]));w.writeheader();w.writerows(files)
with (OUT/'weight_inventory.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(weights[0]));w.writeheader();w.writerows(weights)
free=shutil.disk_usage('/data').free
out=dict(status='REVIEW_ONLY_NO_DELETION_AUTHORIZED',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 scope='Only /data; metadata and existing reports, no weight/PCM load or file-content hashing',free_bytes=free,
 groups=groups,candidate_files=len(files),candidate_logical_bytes=sum(r['logical_bytes'] for r in files),candidate_allocated_bytes=sum(r['allocated_bytes'] for r in files),
 primary_A_allocated_bytes=sum(g['allocated_bytes'] for g in groups if g['level'].startswith('A')),
 secondary_B_allocated_bytes=sum(g['allocated_bytes'] for g in groups if g['level'].startswith('B')),
 caveats=['Not proved weight-state or byte duplicates','Allocated bytes are static st_blocks estimate; df may differ due to snapshots/COW/open handles','No directory-recursive deletion approval implied','No original data/current source/CPEL/final-test outputs included','User must approve exact groups/files before any deletion'])
(OUT/'summary.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items() if k!='groups'},ensure_ascii=False,indent=2))
for g in groups:print(g['id'],g['count'],g['allocated_bytes'],g['title'])
