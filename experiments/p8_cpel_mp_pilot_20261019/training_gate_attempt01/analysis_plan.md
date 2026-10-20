# CPEL M/P版本2：固定一次动态梯度配比小试（未执行）

## 1. 修订依据与不变目标

训练侧8记录×2draw、0更新标定已完成，正确保留原生Fourier W固定。未加外lambda的pair/M梯度范数比中位数.0007206013、最大.2042272。原固定规则给global lambda13.8773，虽中位effective比例为1%，最大却为2.834；该建议未应用。不用一个中位数掩盖长尾，也不遍历lambda择开发结果。

同16个保存ratio离线计算动态规则，得到effective比例范围.002668935–.01、中位.01，两项触及系数上限100。这是当前训练探针上的代数检查，不是新模型运行或总体保证。标定记录及原建议保持不改。

v2仅在真实M/P学习前作以下明确修订：动态梯度配比代替静态外lambda，编码相容性与旧strict结果并列，加一个精度匹配反证packet。数据256/64、目标损失四分量、margin、时间、编辑、原生DSM/逐点aux、初始化、优化器、模型/采样器及部署NFE不变。版本1准备/失败/预检工件和代码快照均保留，**从未进行过M/P optimizer更新**。

## 2. 更新规则，不伪称静态loss等价

每步同4份Score前向，分别求gM=grad(DSM+aux)和gp=grad(pair)，两次反向遍历；pair不含外lambda。固定rho=.01、eta_max=100：

\[
\eta=\begin{cases}
0,&\text{inactive或任一梯度范数为零},\\
\min(100,\ .01\|g_M\|/\|g_p\|),&\text{有限且非零}.
\end{cases}
\]

非finite停止记录失败，不能悄悄设零继续。P将返回的一阶梯度设置为gM+stopgrad(eta)gp，M计算相同两项、范数和候选eta，但只应用gM。没有eta下限。unused/None处理一致：两者None保持None；当前pair计算图是逐点aux图的子图，若gp非None却gM为None应报错，不偷偷改变M的Adam语义。

随后双方同global clip1、一次Adam、一次backend EMA；原生固定W及其余模块不更新。此时有

\[
\|\eta g_p\|\le .01\|g_M\|.
\]

这仅限当前指定可学习参数集合、clip/Adam之前的原始向量范数；不限制Adam更新差、逐参数影响、累计轨迹或语音内容风险。它一般不是某个固定全局标量loss的完整导数，不反传eta本身、不需要二阶梯度。日志可记当前detached系数下的surrogate值，但必须标明更新算法。每step分别实计Score、两次grad遍历、一次optimizer/EMA及实际eta/effective ratio/夹角/零覆盖，不把两次反向算作两个训练更新。

这是工程稳定化训练配置，不作为独立新颖性贡献；科学问题仍是成对响应是否超出相同受控逐点学习并迁移到正常内容，而非证明梯度预算公式成立。

## 3. 数据一致性按编码假设显式分层

独立CPU检查64固定dev，原strict .001为43pass/21fail，全部64都与nearest/general两种量化界相容。所有源为PCM16，无crop码端点，但处理链未验证。不能将后者改写为原21条“本来通过”或noise纯度证明。

v2每个noise实例控制同时记录strict、nearest(.5sum(q))、general(sum(q))、端点/格式和processing_chain_verified=False。工程eligible采用已知整数码、无crop端点、general逐sample界相容；未知/不相容不评、不换例。结果同时呈现全固定清单、工程组和strict子组覆盖，不只挑43个较易例。

允许描述“记录的noise实例扰动响应”，不称真实无语音负类、FPR、幻觉率或已验证纯noise。margin .001及所有历史失败仍原样保存。新增界由编码误差模型而非M/P增强分数确定；不拟合gain/lag。

## 4. 新增一个精度匹配heldout packet

原PCM与float编辑两臂可能有量化格点差异，换Tukey窗不足以排除。训练编辑不变，正常未编辑推理不变；只加一次预定开发反证：

- 同一heldout Tukey事件中心；对S1/S0/X1/X0四端在physical float64坐标执行相同PCM16 nearest ties-to-even操作，q=1/32768。
- 超出整数码范围则记饱和/unknown，不clip后冒充有效。再按同原X1峰值归一化。
- 用实际量化后clean两端各自变换之差作target，实际Noisy差作条件变化；不得沿用raw Delta。保存target相对改变量及两端残差变化，承认量化后不是严格同noise。
- 与raw Tukey共用t=.5、双parent和相同Gaussian seed，所有M/P记录完全配对。额外控制每dev最多4次Score；不增加实际部署网络或NFE。

若raw编辑收益在精度匹配packet消失，不宣称真实证据利用；不能更换bit depth、dither或窗型救本轮。这个packet也不是自然音素金标，不替代原生未编辑输出和未来身份/内容确认。

## 5. 一次小试、评价与退出

M/P同一个明确EMA物化基座、原生可学习参数集合、seed/顺序/随机crop、4前向/2grad、500更新上限及终止EMA，前端与固定参数不动；实际相同成本结构另记录GPU秒/显存。只这一个v2配置，不进行lambda/margin/时间网格、选checkpoint或重新挑样。

正常64 dev的wave MSE/SI-SDR/STOI/PESQ不变；所有失败/undefined保留。开发阶段的候选信号要求raw Tukey与precision-matched packet都具匹配覆盖、M组件均值>1e-12且P≤.9M，不能由零响应或静音解释；noise-instance工程响应不越过预定损伤界，正常P−M不越过原PESQ/STOI/SI-SDR工程损伤界。仍不是正式独立确认，DNS dummy文本不能提供音素恢复率。

原每臂≤3600秒、总≤7200GPU秒、产物≤512MiB/留1GiB磁盘、无网络/安装/源数据或原权重写入的边界不变。若不能完整达到配对500更新，记录budget/OOM等实际终止，不拿不同步数或缺覆盖宣称通过，不自动重启。

当前只准备`config_v2_draft.yaml`与隔离代码；Root还需审核最终SHA，才决定是否授权唯一M/P执行。**不再增加一轮无更新预检或标定；接口与两次grad流程已有实际证据。** 成功只支持进入普通增强A/B与独立内容确认，不等于长期goal已完成；失败关闭本配置，不因准备成本继续救场。
