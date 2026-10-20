# 训练侧梯度尺度标定：固定8记录，零更新

## 为什么需要这次测量

M/P首次真实集成已通，但旧临时参数集合的单crop显示lambda=.1时pair/M梯度比例仅约1.06e−5；不能由此决定全局权重，也不值得不看尺度就跑500步。原生固定Fourier W现已恢复冻结，本次使用同一正确的可学习参数集合测量两项梯度。它不是追加原生N30预检、不是训练或增强性能评价。

## 预先固定的样本和测量

从已冻结256条train registry的四个名义SNR组中，各取原顺序前2，共8条（4279/8404、2307/5789、339/5756、7621/3938）。包括此前4279，不因已看到其比例而剔除。只读这8条S/X共16WAV，不读noise、CV或最终测试；固定中心crop32640，同Noisy归一因子及已有Hann反事实编辑。

每记录2个固定seed draw，各按原生DSM t∈[.03,1]与aux t∈[.3,.7]分布取样，parent0/1随机，同一forward图计算M=DSM+逐点aux和不含外层lambda的pair梯度。固定CPU/CUDA generator、相同原生可训参数集合，64次Score样本前向、32次autograd.grad遍历；没有正常sampler、optimizer、EMA或参数更新。

## 固定聚合，不看开发结果挑权重

逐probe r_i=||grad(pair)||/||grad(M)||，记录梯度夹角及各loss/target覆盖。inactive和zero pair梯度按0保留；任何M梯度范数非正/非有限则整体标定undefined，不补样或选有限子集。8记录×2draw的16个数相关，仅为描述性测量。

固定使用median(r_i)，不是median norms之比。若16项均定义且中位数>0，lambda_raw=.01/median(r_i)，建议值clip[.1,100]；否则无建议。报告raw/clipped与16个实际effective ratio分布及clip原因。1%是这些probe的中位梯度范数预算，不是Adam更新比例、总体保证或声学证据可信度。**脚本只建议，不回写父配置、不自动开训。**

## 安全、复现及局限

仅单次专用Root许可后执行；native PATH含既有env/bin。source严格SCP DataModule、EMA物化一次、F与dnn eval但可学习Score参数保留梯度，Fourier W固定。父helper/配置、源checkpoint、16WAV前后hash保护；加载后的所有参数/buffer digest不变。≤300s、产物≤2MiB、磁盘保留1GiB；不下载、安装、音频/权重导出或自动重试。Engineer只准备，Worker实际执行，Root解释结果。

不调整任何margin/时间/编辑、也不根据结果切换公式来获得好看的比例。当前float编辑与原PCM可能有量化格点不对称；窗口heldout不能排除此捷径。本次没有训练，不能验证或否定这一混杂，不能将尺度建议当成自然弱音素恢复或拒幻觉收益。
