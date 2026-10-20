# 现有N/D权重的public部署迁移补充评价

## 新事实引出缺失评价，不改旧门

接口审计7例partial显示raw谱anchor与真实public入口不天然等价，旧CPEL正常评价是raw接口。旧审计在第7数值界失败后终止、第8未测，保留failed，不改tol或补跑。旧M/P及N/D raw/precision10%门失败永久保留，本次不以新指标替代它，不选择新权重、训练或优化接口。

唯一问题：**既有N/D终末权重相对普通DSM对照的客观变化，是否在真实public部署入口仍存在？** 三臂source/N/D且全64既有dev，是事后部署迁移补充，非独立确认。只报public D−N及N−source/D−source、三臂均值和逐例完整覆盖；原raw结果另列，不跨接口择优部署/写论文。

## 真正部署入口与权重身份

source checkpoint严格加载原拓扑，EMA仅物化一次，全部参数冻结。source dnn完整state在CPU clone内存，N/D已保存的terminal dnn-only各读一次；核schema/arm/源checkpoint身份、每个键/shape/dtype/finite与source拓扑匹配。每臂严格注入对应state后，实际模型canonical tensor摘要须等于CPUstate摘要。F完整state和原生Fourier W必须与旧冻结摘要及source固定W一致；不能调用会把source EMA复制回来的model.eval()，只维持子模块eval状态。

从相同64dev顺序读取全部S/X，固定center32640/原Noisy peak，前端每例只预测一次并经真实inverse恢复物理前端wave。缓存physical front/noisy和同标度clean于内存，无中间文件。source/N/D共享完全相同的输入张量/输入hash，逐例seed固定20261019+100000+原index。3臂均调用原model.enhance，输入[1,L]，mode2/public原Noisy共同norm、rrsd/其它校准禁用、PC reverse_diffusion/none/N30。

实例sampler/profile观察必须不改变调用：实际anchor和Noisy谱hash跨3臂相同、norm同r、输出长度/finite及每次实际30Score。此run直接测public，不声明它等于手写expectedRoundTrip，所以不使用旧接口数值容差门；旧failure没有改判。所有hook在finally恢复。

## 评价与完整性

public返回物理波形后全部除同一原Noisy峰值，与同样归一clean计算MSE/SI-SDR/STOI/PESQ。沿N/D既有精确定义与标准metric内部处理，无独立峰值、外部clip或响度处理。silent/异常metric留null+error、全部64不换样本，配对差按有效同file/crop/seed覆盖明确给分母。生成/几何/权重不符是技术失败；个别指标undefined不偷偷删除病例。

本次不做controlled编辑、音素识别、ASR或人工评价，不宣称补回辅音/防幻觉；source第三臂用于保留相对基座代价，不用D对N好掩盖source更好。即使某项客观差正向，也不反转旧pair门、不自动扩大训练。若无迁移、整体变差或仅source改善，同样完整接受，不换样本/seed/模型/接口救结果。

## 预算与许可

只1次source模型加载、N/D2次CPUstate载入，128PCM/64前端/192后端×30=5760Score，600秒/新常规产物4MiB/始终留1GiB。0训练/梯度、网络/安装、冻结测试集、ASR/DNSMOS、权重写入或复制、波形/谱输出。原权重/数据/旧metrics/用户反馈/服务保留。单独Root授权绑定config/code/registry/3权重/core/旧N/Dmetrics；未授权不读实际权重或PCM，真实失败partial收口、不自动重试，不加新预检链。
