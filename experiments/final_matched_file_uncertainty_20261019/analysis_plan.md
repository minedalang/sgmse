# 冻结最终指标的matched-file配对不确定性补充

论文原pooled区间未保留同file-id在no-reverb/with-reverb间的关联。本次只改变重采样单位，保持原有两条件等权pooled均值、方法和原始数据不变；不重跑增强、评价器、ASR、训练或bootstrap旧实现，也不以区间结果选择方法/阈值。既有final_metrics_repro_audit只核均值，没有这项配对重采样，不是其重复。

只读final_eval_full_20261018/metrics下三arm×两split的六CSV，以及该实验metrics.json。按utt中的唯一fileid整数对齐，两split必须有相同150个ID，三arm同ID；任何重复/缺失/非有限值失败，不删行。七指标固定PESQ_e/STOI_e/SI-SDR_e/DNSMOS_SIG_e/DNSMOS_OVRL_e/DNSMOS_P808_e/DNSMOS_BAK_e。

对Router−off及tcut005−off分别计算每file-id的两条件差平均，再对150配对单位等权平均。该均值必须与原pooled差在2e−10内一致。使用NumPy PCG64/default_rng、seed20261019、10000次有放回抽150ID；每次同一抽样同时用于所有arm/metric并保留两条件，原95%percentile区间用linear quantile。保存抽样index与14列bootstrap均值NPZ（非音频、非训练数据）。主用index gather平均，独立复核用每draw的bincount加权和核全部draw、均值及端点；不import主业务。

记录原历史区间和本次配对区间并列，不因更宽/更窄择用；不补造speaker信息，也不把matched file-id当独立房间、speaker或noise cluster。仍非多重比较校正、跨训练seed或跨数据集确认，不修复最终test域失配/内容证据缺口。即便区间不跨零也不宣布全指标或内容恢复成功。

主/独立各≤60s，CPU单线程、目录≤8MiB；只有6CSV/原metrics/新代码配置可读，无原音频/数据集/checkpoint/模型/外网/服务操作。Engineer仅最小脚本和少量合成权重/配对算术检查，Root审后Worker一次主和一次复核。旧结果SHA前后不变；三工件config.yaml/metrics.json/experiment_report.md齐全。失败留存，不调draw/seed/统计法重试找区间。
