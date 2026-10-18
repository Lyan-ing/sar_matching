from common import *
def main():
 a=json.loads(sorted(ROOT.glob('analysis_*.json'))[-1].read_text());ts=stamp();figts=a['figure_stamp'];table=a['comparison'];T=a['training'];hist=json.loads(output('historical_training_audit').read_text())['rows'];diag=json.loads(output('orb_diagnostic').read_text())['summaries'];real=json.loads(output('real512').read_text())['rows'];cli=json.loads(sorted((ROOT/'cli_test').glob('joint_*.json'))[-1].read_text())
 def pct(x):return 'N/A' if x is None else f'{100*x:.2f}%'
 def tab(rows,columns):return '\n'.join(['|'+'|'.join(h for h,k,fmt in columns)+'|','|'+'|'.join('---' for _ in columns)+'|']+['|'+'|'.join(fmt(r.get(k)) for h,k,fmt in columns)+'|' for r in rows])
 cols=[('模型','method',str),('S@3','S3',pct),('S@5','S5',pct),('S@10','S10',pct),('重复性','repeatability',pct),('正确静态匹配','correct_matches',lambda x:f'{x:.2f}'),('实际点数左','keypoints0',lambda x:f'{x:.2f}'),('无重叠误接受','no_overlap_FPR',pct)]
 eq=tab([r for r in table if r['split']=='all' and r['protocol']=='equal'],cols);natural=tab([r for r in table if r['split']=='all' and r['protocol']=='natural'],cols)
 held=tab([r for r in table if r['split']=='holdout'],[('预算','protocol',str),('模型','method',str),('S@5','S5',pct),('正确静态匹配','correct_matches',lambda x:f'{x:.2f}'),('NN正确','NN_correct',lambda x:f'{x:.2f}')])
 htable=tab(hist,[('模型','model',str),('实际优化步','global_step',str),('有效配对调用','accepted_pair_calls',str),('尝试调用','attempted_pair_calls',str),('不同源记录','unique_accepted_source_records',str),('更新范围','trainable_scope',str)])
 train=tab(T,[('模型','variant',str),('最终步/最佳步','best_step',lambda x:f'5000/{x}'),('best所见源记录','unique_sources_best',str),('实际生成配对','physical_pair_calls',str),('有效轨迹配对','logical_pair_calls_last',str),('训练分钟','elapsed_seconds',lambda x:f'{x/60:.2f}'),('峰值分配MiB','peak_MiB',lambda x:f'{x:.2f}'),('硬门槛','hard_gate_pass',str)])
 strat=tab([r for r in a['strata'] if r['protocol']=='equal' and r['method'] in ['m2','control','joint','frozen'] and r['field'] in ['category','low_texture']],[('字段','field',str),('分组','value',str),('模型','method',str),('正例','n_positive',str),('S@5','S5',pct),('正确匹配','correct_matches',lambda x:f'{x:.2f}')])
 text=f'''# SuperPoint-SAR候选质量联合适配：实际训练与受控验收

**结论：已经获得三套真实训练权重，但新主方法没有达到稳定优于M2、增加正确静态对应的预定成功标准，不建议替换正式M2。**没有继续搜索损失、门槛或新模块，也没有训练LightGlue。所有训练完成并退出；本报告不把合成结果解释为真实跨观测方位精度。

## 1. 历史状态核实

{htable}

历史源池是5111条SAR裁片记录、668个已知来源组件，不是5111次独立原始采集。上表不同源记录通过冻结源码的seed/采样/重试规则重放得到，明确属于重建统计；M1/M2还重放了几何有效点不足16时的重试。已从last.ckpt的AdamW计数核实实际优化步；早期数值失败并重新开始的分支不混进正式轨迹。逐源重放清单在`historical_training_audit_20261014_2031.json`。

M1仅256维残差适配器；M2仅convDa/convDb；M3仅convPa/convPb；A仅新增结构分支；B为结构分支+检测头，描述头仍是M2；旧control只继续描述头。六种历史模型的共享编码器16个参数张量与官方原始权重逐张量完全相等，说明此前未真正开展SAR共享骨干微调。LightGlue此前均冻结，未接受SAR梯度。本轮首次对conv4a/conv4b进行有限SAR联合微调。

旧正式900关系仅读已有CSV。离岸350正例：ORB273/350=78.00%，M2=217/350=62.00%；左图关键点中位数639.5对60，正确静态几何匹配中位数77.5对16，M2失败中75条too_few_matches。该历史成绩不与新验证集相减，不重跑900或据其调阈值。

## 2. 一次有界ORB诊断及决定

预定58对=54正+4无重叠，覆盖近岸/离岸/背景/SSDD；ORB多尺度、ORB单尺度、M0/M2/B共580条预算记录。相同图像、GT和几何门控。严格预算为每侧五方法自然候选数与128上限的共同最小值，每方法均按自身响应排序，实际左图均值53.67；原生自然预算单列。

|方法|等实际点数S@5|等点数正确匹配|自然S@5|自然左图点数|
|---|---:|---:|---:|---:|
|M0|50.00%|16.93|53.70%|98.69|
|M2|59.26%|18.67|62.96%|98.69|
|B|70.37%|22.13|92.59%|586.04|
|ORB 8层|22.22%|6.52|75.93%|488.65|
|ORB 1层|24.07%|6.96|75.93%|575.85|

自然预算ORB检测重复性56.13%、M2为51.74%；等点数ORB47.95%、M2为48.91%，条件描述检索R@1约70%以下对M2约92%。因此数量/覆盖和完整匹配机制都是历史差距的重要因素，不能声称ORB每个描述子更好，也不能把二值NN与SP+LG的差异都归于检测器。去掉ORB金字塔没有降低这批样本的自然S@5，但改变了候选、方向/尺度描述和覆盖，不能叫纯检测单因素消融。样本有限，结论仅是本轮没有增加金字塔模块的证据。

值得借鉴的是覆盖与可重复候选的实际质量约束，不是照搬FAST阈值、增加1024点或直接拼接ORB/Scharr。ORB单尺度实验不用于后续选模；完整诊断逐对CSV、原生匹配缓存均保留。

## 3. 新方法：概率质量校准的几何候选重排

结构完全保留SuperPoint：Conv1–3冻结；joint/control解冻conv4a/b、convPa/Pb、convDa/Db，共968001个训练参数；frozen只训练两个头672833参数。没有新增推理参数、额外结构分支、DCN、Transformer或多尺度网络。M2最佳step3300初始化，原256维归一化描述子、8倍采样坐标、NMS4、阈值0.0005、1024上限与冻结LG接口不变。

![架构及梯度边界](figures/architecture_{figts}.png)

训练机制有三组明确作用：

1. **实际检测位置描述监督**：两图实际NMS点在解析H下建立≤3px静态对应，双方共同可见且排除舰船bbox+5px；双向InfoNCE，温度0.1、最多128正对应、8px内邻近替代正例不作负样本。梯度经双线性描述采样传到描述头及允许的conv4，不经NMS坐标反传。
2. **候选质量目标**：以几何重复性r和正描述相似度相对错误候选的margin构造分离梯度的q；只重排共同静态区域的实际候选。以M2原65类概率为基准，对候选非dustbin概率乘`0.25+1.5q`，再使每个8×8单元的64类总概率恢复到教师总量μ，dustbin保持教师值。KL除`max(μ,0.01)`，避免背景稀释监督。它通过学生logits更新检测头/共享层，不假定TopK可微；没有oracle点直接作为检测结果。
3. **保持与空样本校准**：检测器的教师KL按同样单元质量归一化，描述子在实际对应点蒸馏权重0.1；质量目标额外含随机均匀噪声和交替黑/灰dustbin负例。质量族外权重0.2，空样本内权重0.25。空白/噪声是无对应校准探针，不能视为有真实地理对应的SAR负标签。

普通control使用同数据/步骤/解冻范围/描述监督/教师保持，仅把新质量族替换为已知几何对应上的普通检测响应MSE；frozen保持新质量机制但冻结conv4。control与新方法还同时相差显式噪声校准，故只能评价整个新监督族，不能将全部差异单归因于q公式。本轮没有声称完成每个子项的独立因果归因。

初始裸相对排序及对数响应一致性烟测都出现快速密度膨胀，正式训练前拒绝并保留在`preflight_rejected_*`。校准版30步固定Train过拟合的描述损失下降，点数保持约257→254/257/255，冻结张量、LG读取、保存重载和优化器/RNG恢复检查通过。恢复最大误差门槛1e-7；CUDA grid_sample原子累加不宣称整个训练字节级确定性。

## 4. 数据、实际训练规模与选模

固定原有来源索引Train5111/Val865；Train/Val/Test已知组件交集均0，SSDD原始采集身份仍不完全可证明。未采用其他会话尚未交付的新训练池，训练中没有替换数据。验证404对=372正+32负，112个Val组件；79组件选择、33组件内部留出。全部组件已被历史研发观察，因此是开发验证，不称全新独立确认集。

正例为近岸30、离岸45、背景6、SSDD场景未知291；港口没有可信独立合成标签，不自动把近岸/未知写成港口。留出111正例：近岸9、离岸12、背景3、SSDD87。**32负例全部来自P0010且都在选择集；留出无无重叠样本，留出FPR必须标N/A，不能把它报告为0。**这限制无重叠泛化结论，保留已锁定数据而不事后换负例。

复用旧几何生成器，在新实验中按真实原图尺度裁剪，128/256/384输入；小SSDD不补成大512来伪装分辨率，原始来源最短边190。静态监督排除舰船，独立gamma/对比度/Gamma散斑增强；low和明确none保留。合成旋转、尺度、平移和适度透视不是雷达观测角。验证GT均满足`H_AB=H_B@inverse(H_A)`，明确none交集0，没有被标正例但实际零重叠的验证项。

每项5000步、每步2个在线配对，先无放回遍历5111源，再按近岸/离岸/SSDD/背景0.4/0.4/0.15/0.05采样；不是把增强当独立原采集。分层AdamW学习率检测3e-6、描述1e-5、conv4=1e-6；固定cosine、seed26101434、FP32/TF32关闭，没有根据测试反馈修改。

{train}

三项last均真实覆盖5111条源/668已知组件；best的覆盖范围见表，不把control step1000说成看过全部源。各10000配对/20000视图为保留的优化轨迹，真实生成10008配对/20016视图包含一次故障回放。三任务在step1003遇到两个微批均无静态对应时，描述头无梯度的合法情形被原检查拒绝；在last step1000后通过独立`train_empty_guard.py`恢复，只允许这个明确空样本分支跳过描述头更新，原入口/日志/冻结身份均保留。物理执行了5003次更新、保留轨迹5000步；描述头实际AdamW计数4988或4989，检测/共享为5000，没有编造每步都有描述监督。

GPU3/4/5对应joint/control/frozen，其他任务未终止。表中显存是PyTorch分配峰值，不包含驱动context/nvidia-smi全占用；小输入、单配对微批与仅后部反传使显存很小，不能据此判断未训练。所有选择只读79组件，以硬门槛后category/component S@5、正确匹配和重复性排序；control所选best仍未过噪声门槛，明确作为失败控制保留。

![训练与选模曲线](figures/learning_curves_{figts}.png)

## 5. 统一验证与消融

七方法同图、原生坐标、Homography RANSAC3px/5000/.995、8匹配/8内点、比例0.3和原full gate。ORB为Hamming双向ratio0.8；SP为冻结LG，模型预测不读取GT/静态mask。GT只用于后验计分。严格共同实际预算取七方法每侧自然点数与128上限的最小值，左图平均74.37点；不是让某模型补点。自然预算同为上限1024。

### 严格共同实际点数：372正例、32负例

{eq}

### 自然检测预算：同一验证图像

{natural}

joint自然S@5相对M2+1.61个百分点，但与同范围control完全相同；严格预算joint为65.59%，低于M2的66.67%。严格正确静态匹配21.19→21.01、重复性50.59%→50.66%，没有实现所需的正确对应增量。条件描述R@1略升91.06%→91.95%，但这一变化没有转化为更可靠配准。B同点数75.27%和28.33个正确匹配仍较强，不能把B全部收益归为数量；其自然663.53点、已知噪声1010点与密度验收失败仍然存在，不能仅按自然92.47%升级正式模型。

### 内部留出：111正例、33组件，负例缺失

{held}

joint等点数相对M2新增4/损失3，净增1/111；组件平均Δ=1.01个百分点，探索性95%区间[-3.54,5.56]。自然净增4/111，组件Δ=4.04，区间[-0.52,9.09]。单训练seed、已观察开发来源、组件不等于独立采集，不能声称稳定显著改善。冻结骨干在留出优于主方法，却在全体验证仅与M2接近；没有证明解冻后部骨干是有价值的增量。

![内部留出预算对照](figures/held_comparison_{figts}.png)

### 海事与低纹理重点分层（严格等点数）

{strat}

新方法近岸S@5提高6.67个百分点，但control同幅提高；离岸没有提高，低纹理反而下降，且正确匹配均值未改善。近岸/离岸组件数量有限，不能只选近岸结果宣称完成海事目标。完整几何难度、尺寸、动态、低纹理分层见strata CSV；旋转列为解析H线性项的二维角度代理，不是雷达观测方位。

## 6. 冻结LightGlue是否成了主要瓶颈

同实际检测点和描述子下，全体严格预算joint的NN正确匹配均值16.78、LG为21.01；M2为16.33、21.19。自然joint为28.58、41.36，LG仍有明显净正确匹配增益。检测重复性约51%–52%基本未变，正确候选覆盖仍弱。局部个例可能有LG丢弃，但没有稳定证据支持把LG作为当前首要限制，因此本轮没有微调LG。

本方法的有效部分是完成真实联合梯度、覆盖更多源并避免密度崩坏；无效部分是保持教师单元质量过于保守、只在现存候选中重排，几乎没有改善重复结构的实际覆盖。后一点是与数据一致的机制解释，未被独立子项消融完全证明。不能用Train loss下降或描述R@1小升宣布算法成功。后续若继续SP主框架，优先研究可产生更多**可靠重复结构**而非更多点的监督，当前证据不支持扩大LG或再次盲加骨干模块。

## 7. 真实512纯图像诊断

固定C001/C002/C003原生DN图，七方法21次推理；不使用经纬度、方位、裁片偏移、NoData/船舶mask或人工点作为预测/门控先验。C001约36.37°、C002约45.98°、C003约18.88°仅在推理后说明来源，不是输入。三组均为已观察开发候选，只有两个组件，尚未覆盖目标60°–120°；数据Codex后续独立数据另行冻结评价。

|候选|M0|M2|B|control|joint|frozen|ORB|
|---|---|---|---|---|---|---|---|
|C001|37/11拒绝|16/6拒绝|15/6拒绝|50/8拒绝|14/5拒绝|18/5拒绝|3/0无输出|
|C002|25/6拒绝|7/0无输出|149/42拒绝|38/7拒绝|8/5拒绝|14/5拒绝|15/8工程接受|
|C003|154/62接受|133/66接受|282/130接受|134/56接受|133/70接受|134/60接受|22/14接受|

表为匹配数/自拟合内点，不是真实正确数。joint在C003内点66→70不能证明物理精度改善；B在C002149个匹配仍因内点比例拒绝，ORB仅8内点的接受也可能是错误物理对应。合格独立GT仍为0，**不计算真实Success@3/5/10或真实正确匹配率**。

![C003主方法原生匹配，自拟合支持不等于真对应](real512/C003_joint_{TAG}.png)

21张匹配/拒绝图、全部关键点/匹配/候选H与支持保存于real512目录。真实旧8192/Native与本次512窗口/DN输入不同，不把改善归于模型，不与旧报告直接相减。

## 8. 运行、权重与证据

统一验证平均每对耗时：M2等点数43.34ms、joint43.86ms，自然43.48/44.00ms，包含图像特征、匹配和几何，排除GT计分及权重加载。不同输入128–384混合，不能当固定512速度。验证进程记录的GPU峰值为所有驻留模型的累计进程峰值，不冒称各模型独占显存。独立joint双图512 CLI实际运行退出0，耗时{cli['elapsed_ms']:.2f}ms、单进程PyTorch峰值{cli['GPU_peak_MiB']:.2f}MiB，含首次推理暖机，与批量均时口径分开。

|模型|best权重|SHA256|
|---|---|---|
|joint|[runs/joint/best.ckpt](runs/joint/best.ckpt)|{T[0]['best_sha']}|
|control|[runs/control/best.ckpt](runs/control/best.ckpt)|{T[1]['best_sha']}|
|frozen|[runs/frozen/best.ckpt](runs/frozen/best.ckpt)|{T[2]['best_sha']}|

best、last、每1000步checkpoint、优化器/scheduler/AMP scaler（禁用）、RNG、数据/代码身份及逐源CSV均保留。checkpoint被实际加载并接入冻结LG。环境沿用match：torch2.5.1+cu121、numpy2.2.6、opencv-python-headless4.12.0.88，没有升级依赖。固定核心源码与空样本检查修订分别在code_freeze记录，不把修订当网络/超参数改变。LightGlue SHA与旧权重保持一致。

入口及完整命令见README。关键逐对记录：`final_{TAG}.csv`（5656行）、`orb_diagnostic_{TAG}.csv`（580行）；分层/组件bootstrap/训练覆盖CSV带本次时间戳；历史900从未重跑。结果冻结后不再选另一checkpoint，不继续训练以追求通过。
'''
 report=ROOT/f'research_report_{ts}.md';write_new(ROOT/f'report_identity_{ts}.json',dict(report=str(report),analysis_stamp=figts,code_sha={str(p):sha(p) for p in (ROOT/'code').glob('*.py')},models=T,no_new_weights_chosen_from_holdout_or_real=True))
 with report.open('x') as f:f.write(text)
 readme=f'''# SuperPoint-SAR联合适配实验（已完成）

正式结论与限制见[报告]({report.name})。不重启已完成训练；主方法没有稳定超过M2，不自动部署替换。

## 双图纯图像推理

```bash
cd /home/yl/matching/sar_matching
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n match python -B \\
  SAR_SP_LG_Matchability_20261014_2031/code/infer_pair.py \\
  --variant joint --gpu 4 \\
  --checkpoint SAR_SP_LG_Matchability_20261014_2031/runs/joint/best.ckpt \\
  --image0 /path/to/reference.png --image1 /path/to/moving.png \\
  --output /path/to/new_output_directory
```

每侧不超过512，原生坐标；允许拒绝。GPU仅为命令示例，执行前先检查空闲资源。需要项目现有M2及官方SP/LG缓存，不自动联网下载。只能输入两图，禁止给预测函数传GT/地理先验。成功CLI证据在cli_test及logs。

## 已执行训练与恢复命令

```bash
CUDA_VISIBLE_DEVICES=3 conda run --no-capture-output -n match python -B SAR_SP_LG_Matchability_20261014_2031/code/train_empty_guard.py --variant joint --gpu 3 --resume
CUDA_VISIBLE_DEVICES=4 conda run --no-capture-output -n match python -B SAR_SP_LG_Matchability_20261014_2031/code/train_empty_guard.py --variant control --gpu 4 --resume
CUDA_VISIBLE_DEVICES=5 conda run --no-capture-output -n match python -B SAR_SP_LG_Matchability_20261014_2031/code/train_empty_guard.py --variant frozen --gpu 5 --resume
bash SAR_SP_LG_Matchability_20261014_2031/check_training.sh
bash SAR_SP_LG_Matchability_20261014_2031/stop_training.sh joint
```

上述恢复命令只适用于未完成状态；当前三个selection_locked都已存在，命令会拒绝重训，不能删除锁。独立复现实验应复制源码到新的实验目录、生成新run身份并重新冻结，保持相同seed/数据schedule/配置；不要在本目录重建数据或覆盖权重。原train.py的零对应梯度检查失败入口保留，实际恢复使用train_empty_guard.py。

## 数据/协议与输出

- 固定配置/manifest：config、data_identity、source_schedule、validation_plan、code_freeze均以{TAG}尾缀保存；空样本修订身份另存。
- 训练：runs/joint、control、frozen；best/last可加载及恢复，sampling与source_coverage可核查。
- 验证：404对、112个Val组件；79选择/33留出，全部已观察开发来源；留出没有无重叠负例。
- 完整新统一预测：predictions/final；新真实512预测：real512；旧900保持原样。
- 同图/同H/同门控；实际等点数与自然预算分别报告；不将ORB与LG差异全归因于检测器。
'''
 with (ROOT/'README.md').open('x') as f:f.write(readme)
 print(report)
if __name__=='__main__':main()
