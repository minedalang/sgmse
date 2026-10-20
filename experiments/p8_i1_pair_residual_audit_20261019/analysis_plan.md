# P8-I / I1a：Clean–Noisy配对及残差风险审计（仅准备）

## 科学目的与范围

落实用户第一阶段“准备与验证”中的实际配对检查，不训练检测器、不验证CAEG/音素身份。全部I0固定12对/6speaker保留；noise identity仍unknown。只读取24份来源WAV并在内存解码做CPU描述，绝不写新WAV/NPY/残差音频，不编辑/混音/生成事件，不调用前端、扩散、ASR或质量模型，不改原始文件/旧记录。Root仅冻结设计并授权Engineer软件准备，真实Worker运行仍须最终复核及独立launch授权。

## 固定统计及完整保留

1. 严格I0路径/hash/header/字节锁；16k monoPCM16完整无截断。解码int16/32768为float64。全句不裁剪、不peak归一化、不去DC。记录每信号样本数、min/max、abspeak、DC、RMS、精确零与PCM端点(-32768/32767)计数；端点不是剪切因果证明。
2. lag l在[-160,160]每样本321点；固定S索引[160,L-160)（闭区间[160,L-161]、L-320点），对应X[n+l]。正lag表示X相对S延后。每点输出centered Pearson、uncenteredOLS gain、拟合残差MSE、S/X功率、均值；固定共同支持零lag与best作比较，全句零lag另列不可混比。Pearson分母任一中心化能量为0记null及理由；OLS分母S能量0同理。不加epsilon造有限值。best在有定义corr中取最大（不abs），max-1e-12内数值tie按(|l|,l)，全无定义则null；报告边界命中，不扩大范围。
3. 主残差是全句Nraw=X-S（gain1/lag0），记录功率/RMS/mean/abspeak、与S的未中心化内积及中心化Pearson。核X1=X及S+Nraw重构最大绝对误差，强调只是代数/实现检查，不是纯noise证明。另列全句zero-lag OLS残差、共同区间zero/best OLS敏感性，不将任一fit用于生成对齐/增益后的数据。
4. 20ms窗/10ms hop=320/160样本，从每个start in range(0,L,160)保留[min(start+320,L)]尾段，记录实际n，不补零。全部帧CSV：S/X/Nraw的mean/power/RMS、uncenteredOLS gain、Nraw与S内积及Pearson、frame-fitted残差power、正交内积。帧gain波动与残差关联只描述风险，不能解释成真实语音泄漏率/局部真实SNR；零分母保留null与理由，不能丢帧。
5. 输出12对完整摘要、321×12全lag曲线与全帧行。等pair min/max/median等描述须标明有效n；无筛例、阈值、置信区间或speaker/噪声独立性能主张。noise identity unknown、noise_disjoint/split/scientific_ready保持false。

## 核心解释边界

- raw residual恰好重构X不等于近似加性模型或纯noise成立。S来自本地16k，先前官方48k转换门失败仍保留，不倒改为来源一致。
- OLS强制同一拟合支持上的未中心化残差内积近0，既不强制centeredPearson0，也不保证各帧正交；不能从fit后的低相关证明无语音泄漏。
- 从321个lag中挑最高corr的改善具有选择效应，不自动是显著时延或错配证明。低SNR/相位/噪声相关可能造成不同bestlag；不为找一致调整范围。
- 配对身份、纯噪声、实际音素内容及noise来源没有单靠本审计通过的阈值。技术完整/独立数字一致与科学gate分开。后续需要怎样保留raw、或另外取得有来源noise，是Lead收到结果后决定，不是本driver自动“修复”。

## 实现与验证（Engineer当前仅本地合成）

代码只写本新目录。允许标准库wave/array与NumPyCPU；禁止torch/soundfile/librosa/ASR/质量库、网络及未知子进程。准备时不读真实WAV/header或运行审计main。8MiB总目录含准备及将来记录；准备建议<=256KiB，不复制I0大preflight或音频。manifest锁定config/本设计/脚本/全部I0来源记录和24音频SHA，不把继承SHA冒充本轮已重读。主与validator各硬120秒、单线程CPU、独占STARTED/验证marker、无自动重试；预留收尾与真实失败记录，非仅结束后检查耗时。

主可用向量化矩/相关计算；独立validator不得import业务统计函数，独立int16解码，按直接中心化点积/残差重建全部12摘要、321lag曲线、所有帧。每次源PCM解码24，主+验证48，分别计sample/file/byte；不得验证中偷偷调用前端。pre/post来源和原metrics保持。

合成测试覆盖正/负已知lag、gain非1+DC、恒定/全零/negativecorr、tie、边界lag、非整尾帧、零分母及nullJSON、OLS内积vsPearson区别、错hash/路径/截断header、已有marker、超时/产物限额与网络/模型保护。统计correctness测试仅使用合成数组/临时WAV，禁止真实试跑smoke。

## 未来一次Worker命令与记录

Engineer交prepare报告、driver/validator/test源码与合成回执、preparation_manifest；Root复核后另给唯一精确命令。运行需标准config.yaml/metrics.json/experiment_report.md与独立validation，原metrics保持不可变。失败停且保留，不扩阈值重跑。当前无I1结果或新的波形/模型许可。
