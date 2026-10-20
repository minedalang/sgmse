# 唯一公开 TextGrid 示例：原始字节与结构审计

Research Lead 批准一次小型**公开标注元数据**取回/审计，不是语音增强实验。入口 [官方主页](https://psi.engr.tamu.edu/l2-arctic-corpus/) Annotations 段明确公开链接到 LXC/arctic_a0018 TextGrid 示例。Root 已用 web 浏览该链接的文本呈现；呈现不提供原始字节 SHA，故本步骤只做一次原始字节归档，不重复科研模型实验。

唯一允许取回的 URL：`https://psi.engr.tamu.edu/wp-content/uploads/2018/03/lxc_arctic_a0018.txt`。不访问同页 WAV、其他样本、完整语料链接、表单、镜像或登录。公开示例不构成获取完整注册语料的授权，不接受/提交许可或个人资料。

## 固定边界

- 1次 curl GET，HTTPS、无重定向、无自动重试；代理 `http://127.0.0.1:7890`；请求 identity encoding。
- curl 30s总时间、10s连接时间、最大正文65,536字节；目录最终<=2,000,000字节。若长度缺失仍需事后硬核验；超限/非200/HTML登录或错误页/内容格式不符即失败并停止，不换传输或域名重试。
- 原正文/响应头/退出码/UTC/实际命令/已观测正文大小及SHA全部保留；不编造协议/线上传输总字节数。不给网页呈现重构出的文本伪造原字节哈希。
- 0音频下载/解码、0重采样、0模型/ASR/训练/波形输出。标准库离线结构审计，不安装依赖。不得执行语料文字或修改既有代码/实验。

## 原始结构审计

保留全部tier名称、IntervalTier索引、声明数量和逐interval索引/start/end/raw label；明确按tier保留，不把 words/IPA层当作phones。可仅支持此示例的长文本格式，不实现通用TextGrid parser，不静默丢空标签。

使用Decimal检查有序、非负、start<=end、tier全局范围与相邻连续关系及声明计数；报告失败，不自动修时间。按原始phones标签统计空值/无逗号/三字段s/a/d/未知形式，保留所有例子。`sp`语义未由本轮来源证明，不能自动解释为speech或某种确定音素；空值也不是silence真值。人工标签身份只是来源声明，未访问音频复核。

Root已浏览到该示例含 `sp` 和空phones interval，且有三个 substitution triples；这些是取回前已知来源检查，不可称独立确认或据此选一批“通过样本”。对实际正文独立清点，若不同完整报告不调预期找一致。不将该单例无a/d标记推断为整个语料无a/d。

## 交付与保护

Worker独占 `STARTED` marker，已存在则停止。不重做已成功原始取回。先读协议/计划、核对provider与bridge、git和已有产物。只写本目录。

交付 `config.yaml`（Root提供，保持）、`metrics.json`、`experiment_report.md`、raw文本/headers/log、结构JSON及独立标准库核验脚本/结果。核验至少检查源字节hash、独立tier/interval数量、时间和raw label保留；若业务解析与独立核验共用同一解析函数，必须披露，不能称独立。

冻结保护：标签契约目录 `file_hashes.json`所列6文件及manifest本身；既有VoiceBank attempt02 metrics SHA e0ed9b10c646cc19d28cc8c6c6cd0225fd87c1b19b77bac108b1bb19a17066e0；B opening metrics SHA a99373256417f29d01ee6f06758586ca06d6806335c52a53c3a096655211bbd7。pre/post检查并保留dirty树/桥/试听服务；不得修复v0契约，Root另行决定软件适配。

成功只代表公开单例原始来源/结构已验证；不是完整数据导入、音素真值确认、P8-V校准/训练或科学假设通过。
