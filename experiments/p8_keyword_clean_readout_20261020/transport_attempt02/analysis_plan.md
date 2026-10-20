# 同一固定64Clean读出：唯一一次取得方式修正

父目录唯一运行于120.018秒下载期限终止，仅保存11,534,336字节不完整archive，0成员读取/0ASR。父metrics/report/partial保持原样。它是取得预算技术失败，不能记0/64或评价识别器适配性。Source元数据运行的超时与4MiB预算失败也仍保留。

Root与独立工程静态审查判定：不必为约2MiB的64个目标WAV取得整个182MB包。允许一次新的明确运输方案，不是续传、重跑父driver或按识别结果救场。固定样本、原始字节、ASR全部参数、词规则、58/64且每词6/8的门均与父配置完全相同；0既有ASR结果可供选择。父配置selection/audio/asr/scoring/screen逐字段相同。

只按被冻结CD的64条offset/ASCII name/compressed_size作64个Range，每段长度30+文件名字节数+65535（最大local extra）+compressed_size，尾端不超archive。预期总响应body5,535,648字节，64原WAV总2,050,816字节；仅为存储metadata预估，不是已取得。每请求206/精确Content-Range和Length/固定ETag/identity，无redirect/retry/mirror。拒绝返回整个包的200，不读取全body。

核local PK0304签名、filename、flags8和deflate8，读取实际extra长度后只对精确压缩段有界raw-deflate，核原大小/EOF/未消费尾/CRC；bit3可使local CRC与size为0，以冻结CD为准，不误把它当冲突。邻近过取压缩字节仅原样保存，不解析或解码其他成员。拒绝未知加密/ZIP64/方法，不做兼容fallback。没有完整archive SHA，不能声称验证了完整包；记录每段与各原WAV SHA，原字节CRC与冻结官方CD相容。

所有64原成员取得并header通过后，才按父固定tiny.en执行64Clean读出。不得提取/解码reserved testing成员；承认过取可能接触其压缩字节。单独Root许可绑定新code/config及原64manifest/receipt/官方metadata/ASR来源锁。只允许一次新driver，不增加预检、源包扫描或通用remotezip平台。

成本：最多64HTTP/8MiB累计响应body，取得阶段180秒、总300秒、产物16MiB/留1GiB；保留64range原字节用于复核。内部ASR decode调用实计、64transcribe上限，仍0加噪/增强/CPEL权重/训练。若再次取得失败或Clean不适用，停止该运输/读出支线，不追加第三种抓取策略或改prompt/识别器。不宣称低SNR、自然弱辅音恢复、FPR或人耳可靠性已验证。
