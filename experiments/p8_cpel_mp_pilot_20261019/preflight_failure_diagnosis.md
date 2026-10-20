# 无更新真实预检：两次技术失败与下一修复

## 已发生，而非科学负结果

1. 初次预检进程36540/session10351，6.19s后exit1。`sgmse`导入触发Torch扩展加载，因child PATH没有项目env/bin，找不到Ninja。未到checkpoint模型构造/PCM/Score/反向/更新。来源保护字节哈希不变。完整记录在`preflight/experiment_report.md`和`preflight_result.json`。
2. Root实际执行既有env/bin/ninja --version成功，并核旧成功原生基线使用同一个env/bin前缀PATH。明确授权新attempt02；四份代码、配置及清单复制后逐字节相同，旧失败保持不变，只改child PATH，不安装或替代算子。
3. attempt02进程37385/session93436，10.52s后exit1。已越过Ninja/import阶段，在`load_from_checkpoint`构造期因`JointWaveDataModule`缺少必需`base_dir`失败。checkpoint加载调用发生，但模型构造/严格state应用未完成；PCM、Score、反向和参数更新均0。参数digest、加性、梯度尺度、内存/时间的真实模型结果均未知。记录在`preflight_attempt02/preflight/experiment_report.md`。

这两次都没有进入M/P训练或测到条件响应，不能称为“CPEL两次无改进”，也不能声称真实API已通过。缺依赖的初始报错经检查实为PATH遗漏，不应安装一个已经存在的Ninja。

## 有针对性的静态修复，当前不继续试跑

Root要求Engineer一次核完数据模块及相关API，而不是逐个运行碰错：

- `joint_train.py:163`针对joint_scp选择`JointWaveScpDataModule`；文件含重复类定义，应核最终导出签名。
- 原生推理旧入口虽然传`JointWaveDataModule`，还需核其附带的base_dir构造参数；不能只复制类名，也不添加假base_dir掩盖问题。
- 核checkpoint已有metadata位置、谱变换/裁窗属性、STFT window设备、EMA物化和原生sampler接口。修复不得改变神经拓扑、strict加载、EMA、样本、loss或采样过程。
- parent准备代码修复前留带SHA备份；两个失败attempt的工件不覆盖。修改后只做静态/少量合成接口检查并交Root，未授权第三次真实调用或M/P。

后续若再批准真实检查，应是明确修复后的新attempt，而非删除STARTED、覆盖失败、换样、降级dtype/strict或自动重启。实际训练仍需单独Root许可，当前不存在`root_execution_authorization.json`。

## 后续attempt03：接口通过，但不是训练放行

Root实核最小diff后明确批准新attempt03，保留原两次失败。17.19s内完成2native＋2aux＋30正常Score、2次反向、0optimizer/EMA/参数更新；参数和buffer digest前后相同，GPU峰值约9.1GiB。完整报告在`preflight_attempt03/preflight/experiment_report.md`。

这个结果真正改变了下一行动：API可运行，但本crop有效pair/M梯度比仅1.059e−5；gain=.188已在当前包络内，lower/upper为零、orth/outside非零。因此不能从“梯度存在”推出当前pair强度足以检验学习假设，亦不能由单crop判定全体。M/P的500步尚未授权。

train加性relative RMS .00019356通过原门槛；dev .003355588未通过原.001门槛，不能给该dev noise-only完整有效解释。随后独立CPU诊断只重读同dev三源：均PCM16，raw residual RMS2.1433e−5、max一个量化步长3.0517578125e−5，32640点都未越过“仅最终nearest rounding”条件下的1.5LSB上界。它与舍入解释相容，**不证明noise纯度/编码舍入模式，也不撤销旧failed gate**；见`additivity_diagnosis/experiment_report.md`。后续若采用量化感知容差，须先明确编码/舍入前提并对所有样本同规则，不能看到失败就把.001提高到能通过的数。

Root又核出全量解冻dnn会误解冻原生GaussianFourierProjection.W（源码原requires_grad=False）。准备代码已最小修复，optimizer/grad probe/EMA仅作用原生可学习参数；旧代码/配置有备份，所有attempt结果不改。attempt03的271项梯度与比例属于旧临时集合，不能宣称新子集已实测。当前配置保留原生固定W，不改变前向拓扑或任一已有权重。

当前停止真实预检，不做第四次或M/P；下一步为训练侧loss尺度标定设计和量化/来源前提规则，而不是继续软件检查数量或盲跑500步。原生参数范围修复见`intrinsic_fixed_parameter_fix.md`。
