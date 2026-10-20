# 两个指定压缩包删除回执

用户授权原话：“ok 这两个zip删了”。本次仅删除以下两条精确路径，未触碰早先752项候选或执行科研任务。

- `/data/aliyunpan-v0.4.0-linux-amd64/Downloads/86e8209211ee43c18d73dccdf862dfb9/dataset/dns_ins20/clean_testandtrain.gz`：已删除，复核路径不存在。
- `/data/coding/sgmse/experiments/p8_keyword_clean_readout_20261020/mini_speech_commands.zip`：已删除，复核路径不存在。

删除前逐项核对最终确认manifest SHA、regular file、resolve、nlink=1、device/inode/size/mtime和allocated bytes，均一致；扫描/proc文件描述符未发现占用。未读取压缩包内容或解码。

预期释放allocated bytes：50060414976（约46.622GiB）；记录df前可用：106642829312，df后可用：156703240192，实际df差值：50060410880 bytes（约46.622GiB）。随后复核可用空间为156703170560 bytes。最初statvfs在回收到账前采样，未即时反映释放；原观测与后续df均保留。并发文件系统写入可能造成小幅差异。

原始clean包外部备份由用户确认，未独立核验。失败partial下载的metrics/report仍保留，原审批manifest与候选CSV未改写。所有权重、WAV、数据集、旧指标、答卷和运行环境均未纳入本次删除。Root另报告goal已active；本次未改变goal状态或启动任何研究/训练。
