# CPEL正常训练预算准备：尚未授权执行

用户说明此前创新通常100k步起才显效。500步只回答短程筛查，2000步准备不启动；旧结果和旧门完整保留，不外推为一般CPEL无效。

本目录仅准备正式长程匹配对照。拟两组各100000次更新，从相同source EMA开始、相同完整训练数据/随机流/学习率/编辑/1%原始梯度预算/.999 EMA；区别只是否应用成对梯度。仍冻结前端和原生Fourier参数、停用Router/额外共同点clean回归。原普通组与成对组同4次Score/两次梯度遍历，额外计算成本另披露。扩大数据与步数后，不能将相对旧500步差异纯归因步数。

## 当前允许的准备

1. 只读现有元数据，登记完整eligible训练/CV、旧64趋势面板和按metadata预选的新固定验证面板，不读音频或冻结测试。
2. 独立断点恢复模块与极小CPU合成测试；不读真实权重、源checkpoint或WAV，不实例化SGMSE，不用CUDA/网络。
3. 设计与审查。当前进入隔离训练driver软件准备；实际source/CUDA技术检查和100k运行仍需Root另签授权。

## 数据准备规则

原metadata来源：experiments/p8_training_coverage_audit_20261020/{tr_synthetic_records,cv_synthetic_records}.jsonl。
沿旧CPEL receipt排speaker10658与8个探索file IDs；预计train10797、CV1186。不是新音频状态审计，不声称noise原录音/类别隔离或CV预训练未见。

新趋势验证面板按tag `cpel-longrun-v1|validation|` 加utt的SHA256字典序，各旧priority_bin取前64，共256；从eligible CV排除旧64的utt，不按模型分数或音频选样。该面板仍为开发验证，不是独立确认。旧64另作对照趋势面板，保留原顺序。最终全部eligible CV的评价到底采用固定短裁窗还是完整语句，必须在执行前冻结；不能人数覆盖冒充整句部署。

## 长程安全要求

所有输出仅在/data；不使用根盘、/tmp或/dev/shm存放训练产物。用户审查清理清单后才可删除明确批准文件，当前无删除权限。

checkpoint需raw后端、EMA、Adam、已完成步数、全局/局部RNG、剩余shuffle order、完整来源/config/code/registry身份、冻结摘要、日志commit offset/SHA。只在完整optimizer→EMA→日志提交后保存，无梯度累积故不存gradient buffer。恢复时RNG最后恢复。不把旧EMA-only文件称exact resume。

训练日志append-only，不每10步重写100k完整历史。mutable latest原子替换须先写同目录临时文件、flush/fsync、校验，然后替换并fsync目录；旧latest在新件确认前保留。不覆盖任何历史权重或固定里程碑。失败partial保留，恢复必须显式核查，不按超时自动从头重启。

拟固定10k/25k/50k/75k/100k里程碑，最终100k EMA为主，不挑最好、不以早期proxy未过10%杀正常长训。100k不是充分收敛保证。实际public、质量/内容/身份/效率评价以及数据限制必须预先写清；仅pair变好不算最终成功。未经独立对照，冻结source不代替同预算无编辑第三组。

按短程吞吐外推两组训练约22GPU小时，正常I/O/checkpoint及评价额外；不是已测长程耗时。存储初估新增峰值约2.8GB并另留1GiB，此前/data约1.3GBfree的状态已过时：用户扩容并仅授权两archive清理后，可用约156.7GB，空间不再是准备限制。仍须冻结实际资源预算再执行。
