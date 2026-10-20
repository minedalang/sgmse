# P8 checkpoint基础权重身份：静态只读比较

## 科学问题与用途

P8现有frontend/native-OFF结果使用80k Router checkpoint，不能仅据文件名或`rrsd_freeze_base=True`推断其等于原300k StoRM的EMA。当前源码冻结非Router参数、新训练EMA仅追踪可训练参数；但历史初始化是否物化原EMA尚未由本次证据证实。本审计只回答**当前两份checkpoint的保存权重字节对应关系**，不重建历史训练过程或性能。

既有`checkpoint_data_condition_audit_20261018`及`rrsd_router_mechanism_audit_20261020`覆盖配置/机制，未提供这里需要的跨checkpoint命名tensor字节比较；因此不是重复增强或重测旧数据。

## 唯一比较与解释

- A：`joint_train_logs/sb61i0gj_RRSD/epoch=118-step=80000.ckpt`；P8运行所用文件，既有SHA256为`ab50510ca7fd80798b70f3f0b5365b17448a5f98c75d38755b82212ae2a119c9`。
- B：`joint_train_logs/5ul7u03z/epoch=444-step=300000.ckpt`；只核当前文件，不由路径认证训练来源。
- 主比较：A中`dnn.*`、`frontend.*`与B中同名RAW tensor的dtype/shape/布局、有效存储字节；逐项保留equal/different/unknown，不因不等而更换checkpoint。
- EMA仅作显式辅助描述：保存条数与内容匹配。若缺可信的参数名—EMA位置映射，必须记语义身份unknown；形状一致或无名内容匹配不自动证明命名EMA模型相等。
- 若RAW同名全部相同，只能说当前保存的对应RAW权重一致；不能自动说历史训练预算公平、EMA相同、运行配置/输出相同。
- 若不等，明确差异范围；不把差异自动归因为训练不足或Router损坏。若布局/序列化不支持，如实unknown/失败，不追加torch加载修补。

## 边界与成本

标准库静态ZIP读取，禁止原pickle GLOBAL执行、torch/numpy导入、模型实例化、波形/header读取、数据集/最终测试访问、网络、训练、GPU及checkpoint修改。不解压出storage副本。代码只允许受限解析器将四种已观察GLOBAL转换成惰性描述；未知内容fail closed。

执行前后hash两源checkpoint；记录data.pkl SHA、storage读取字节、有效tensor比较与源文件稳定性。主运行一次≤120秒；目录产物≤2MiB；不足或超限停止，不自动重试/换方式。Engineer只作软件准备与小型合成安全测试；Root复核后才委派Worker执行唯一静态比较。

完成后保存`metrics.json`与`experiment_report.md`，解释回到现有P8论文讨论；不新建增强实验、不因身份比较自动批准重跑300k或训练。独立听者/播放授权仍未确认，这项审计不取代内容评价。
