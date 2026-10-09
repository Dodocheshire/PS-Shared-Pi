# PS-CBM 的 Pi 噪声研究

基线为 **Partially Shared Concept Bottleneck Models（AAAI 2026）**，官方实现来自 [7494zdl/PS-CBM](https://github.com/7494zdl/PS-CBM)，论文见 [AAAI](https://ojs.aaai.org/index.php/AAAI/article/view/38312)。此处的 PS 表示 Partially Shared，与主项目 `Essay/PS` 的 Post-hoc Stochastic CBM 不同。

`official/` 保持官方提交 `749c6e3b044606c18d60ce07e3b32171f25ef563` 的 78 个文件逐字一致；运行时按 `UPSTREAM.json` 核对 SHA256。新增实现放在 `ps_pi/`。此仓库作为 [Pi-CBM](https://github.com/Dodocheshire/Pi-CBM) 的 `PS-Shared-Pi` submodule 管理。

首轮复现采用官方文件名划分，将 CIFAR 原始批次的文件名映射到 torchvision 图像索引，逐项核对类别。CIFAR-100 文件名按类别区分（原数据有跨类别重名），CUB 的数值目录标签重新映射为官方类别的字母顺序。CIFAR-10 使用完整的 45,000 / 5,000 / 10,000 划分。概念筛选、合并和标准化只使用训练集；准备与基线训练均不打开测试图像特征。教师为 CLIP ViT-B/16，目标为 CLIP RN50 的 2048 维倒数第二层浮点特征，不使用 1024 维投影特征替代。编码批量默认为 256，而官方默认 512；记录此运行差异。

官方脚本将 `state_dict()` 的可变引用当作最佳验证权重保存，最终实际保留末轮参数。因此基线同时保存 **官方末轮状态** 与 **克隆保存的最佳验证状态**，分别拟合同预算官方稀疏 SAGA 头（头训练独立重设相同随机种子以配对对照），避免把该修正算作噪声增益。CBL 使用官方配置的 20,000 步（CUB 为 30,000）；SAGA 上限 10,000 epoch，保留官方收敛终止条件。运行配置见 `configs/`。

```bash
cd PS-Shared-Pi
python -m ps_pi.prepare --dataset cifar10 --data-root /path/to/datasets
python -m ps_pi.baseline --dataset cifar10 --seed 42
```

需要 CUDA、PyTorch、torchvision、OpenAI CLIP，以及官方依赖。当前机器使用 `/home/pengcheng/miniconda3/envs/pi-cbm/bin/python`。`python -m ps_pi.launch` 串行运行首次 CIFAR-10 准备与基线训练，状态与日志保存在 `artifacts/logs/first_baseline/`。`--smoke` 使用独立输出目录并明确标注调试预算，其指标不得当作正式结果。

后续目标是用部分共享概念的结构约束 Pi 噪声，先以验证集筛选方案，再锁定方案进行独立五种子实验。必须保留仅均值、同容量确定性适配、固定随机噪声，以及同一个 Pi 模型关闭尺度的对照。只有随机分量优于这些对照时，才将增益归因于噪声。解释、校准和概念效率同时检查。当前尚无新的正式性能提升结论。

训练特征、模型权重和生成运行在 `.gitignore` 中，仅保留本地。简明主要结果持续整理到主项目 `Essay/PS-Shared/report/`。

## 验证集上的噪声迭代

`ps_pi.noise` 根据预测概念生成均值、对角尺度与共享因子尺度。共享因子的支持由训练阶段合并的概念类别集合确定，仅关联同一共享组的概念，协方差保持半正定。推理不接收真实类别。固定 CBL、标准化与稀疏分类头；仅训练噪声生成器。配套仅均值、同参数数确定性适配、对角噪声和固定高斯对照，并评估同一 Pi 模型关闭尺度与全部关闭扰动的结果。

```bash
python -m ps_pi.screen --prepare-remaining
python -m ps_pi.screen_mixup --dataset cifar10
python -m ps_pi.screen_predictive --dataset cifar10
python -m ps_pi.export_validation
python -m pytest tests -q
```

首轮 `validation_v1` 使用分类预测概率的 Monte Carlo 边际似然、期望概念 BCE 和扰动能量，已完成九组完整验证对照。90.94% 的最高验证准确率与同容量适配相同，关闭随机尺度后准确率相同，且 NLL/ECE 变差；因此当前没有可归因于随机分量的可靠增益。

第二轮 `validation_mixup_v2` 使用概念空间 mixup、独立非线性确定性分支、预先记录的校准/概念约束选模。第三轮 `validation_predictive_v3` 进一步将概念目标改为预测概念概率的边际 Bernoulli 似然，并同时保留期望概念 BCE 作诊断：期望每次采样的 BCE 与平均概率对应的 BCE 含义不同，前者会因凸性惩罚噪声。每个版本保留独立源码、协议、断点与输出目录，旧结果保持原样。

这些结果均来自验证集，尚未进行最终测试或独立五种子验证。研究状态与图表位于主项目 `Essay/PS-Shared/report/`。软件和上游来源由各协议与 `UPSTREAM.json` 记录；新增模型、特征和逐样本预测仍保存在本地忽略目录。

## 第五轮：联合训练与人工属性

前四轮 42 组对照已完成。第五轮在 CBL 学习期间注入语义方向的特征噪声。三个数据集的方向只用原训练集计算；CUB 为 509 个概念 / 200 个共享类别方向，CIFAR-100 为 583 / 96，CIFAR-10 为 61 / 9。噪声协方差为方向加权外积之和，保持半正定；经过非线性 CBL 后不保证同样的零相关结构。

八类对照匹配初始化、数据顺序、分类头、优化步数和每图 CBL 前向次数。共享关闭采样对照保留相同生成器、尺度先验与能量项，仅关闭采样。两套预登记配置各训练 40 epoch。

CUB 的 16 组已完成并冻结所选权重。用两组独立随机数、各 256 次采样复核，共享噪声验证准确率为 68.75% / 68.80%，没有超过关闭训练采样对照的 68.95%。干净联合微调为 68.85% / 68.80%。人工属性保守映射覆盖 68/312 属性，只在训练集筛选属性与设阈值，验证排除 certainty < 3。原基线属性 BA / AUROC 为 67.43% / 72.52%；共享噪声 BA 略降，不能宣称解释性能提升。此检查只评估整图属性预测，没有定位或因果干预结论。CIFAR-10/100 的同配置 16 组对照及独立 MC 复核均已完成。

```bash
python -m ps_pi.directions --dataset cub
python -m ps_pi.joint_screen --dataset cub
python -m ps_pi.native_attributes --annotations /path/to/native/annotations
python -m ps_pi.joint_audit --dataset cub --samples 256
python -m ps_pi.export_joint
```

精简图表和来源摘要见 [主要进展](reports/validation_v1/主要实验进展.md)，登记方案见 [联合训练协议](reports/validation_joint_v5/registered_plan.json)。当前 44 项测试通过。Goal 保持 active；仍需可靠优于最强对照的收益、锁定方案后的独立五种子和最终测试证据。


## 残差方向、预测似然与稀疏头适配

第五轮三个数据集共 48 组；后续四个 CUB 队列各 16 组，分别检验训练类内残差方向、训练敏感度尺度校准、联合边际似然及原稀疏头适配。加上前四轮 42 组，目前主要报告汇总 202 条完整验证运行，均有预登记控制、固定选模和独立高采样复核。只有一个基线/适配训练种子，重复控制不是独立重复。

残差几何仅使用训练样本，移除训练类中心和概念均值方向空间。在相同输入能量下 CUB 训练子集的概念漂移从 0.105 降至 0.022；后续尺度 0.481 来自训练敏感度比值，不是验证搜索。CUB 同容量适配从基线 68.70% 达到 69.45%，最新共享噪声复核 69.30%，尚无独立随机收益。稀疏头更新仅允许原 663 个非零位置，所有对照采用相同支持与目标。原始属性 BA 略降，不能宣称解释提升。

```bash
python -m ps_pi.residual_directions --dataset cub
python -m ps_pi.residual_diagnostic --dataset cub
python -m ps_pi.joint_residual_screen
python -m ps_pi.joint_residual_calibrated_screen
python -m ps_pi.joint_residual_predictive_screen
python -m ps_pi.joint_sparse_head_screen
python -m ps_pi.joint_sparse_head_audit
python -m ps_pi.expected_concept_audit --input outputs/cub/validation_joint_sparse_head_v9
python -m ps_pi.export_major
```

`expected_concepts` 对官方 Linear–ReLU–Linear CBL 的条件高斯扰动解析计算概念期望，区分非线性均值变化与 MC 分类概率；不是重新训练的控制，也不能据此归因训练随机性。完整摘要和源代码哈希见 `reports/joint_followup/`，当前状态见 `reports/CURRENT_STATE.json`。

第十轮特征层 mixup 的 16 组训练及复核已完成：固定 alpha=0.2，将特征、生成器上下文、概念目标和类别软标签按同一权重混合，保留同预算的八类对照。独立混合随机数状态随断点保存；原始验证输入保持完整。完成前不进入主要结果，最终五种子及测试仍待可靠候选后进行。


特征 mixup 首种子共享模型独立 MC 为 69.40%，同配置关闭采样为 69.30%，对角噪声为 69.35%–69.40%。这只是很小的单种子随机训练差异，尚无共享结构优势，人工属性 BA 下降约 0.17 pp。第十一轮 `python -m ps_pi.feature_mixup_replication` 已登记并启动其余四个适配种子的四类配对控制，固定基线种子42、概念权重1 / CBL lr1e-5、alpha0.2；自动接续 MC 与解析均值复核。共用完整基线，不能宣称独立五基线证据或最终测试。


第十一轮已完成：五个适配种子相对关闭采样的平均差值 −0.02 pp，条件配对95% t区间 [−0.19,+0.15] pp，未确认随机优势。共享原始属性 BA 平均下降0.24 pp。新图 [条件种子复验](reports/validation_v1/05_conditional_seed_replication.png) 明确区分该区间与主队列的 MC 范围；它仍共用一个完整基线。

第十二轮 `python -m ps_pi.joint_robust_mixup_screen` 已启动完整16组，唯一目标变化是分类损失从边际似然改为期望逐次采样交叉熵；概念边际似然、均值保持、mixup、方向、固定头和预算相同。控制器完成后自动接续独立 MC / 属性 / 解析概念均值复核，未完成前不进入主要成绩。运行状态是带时间戳的快照，实际状态在本地 `artifacts/logs/cub_joint_robust_mixup_v12/status.json`。


第十二轮完整16组及三类审计均已完成：共享69.10% / 69.20%，未超过关闭采样69.30%。目前共202条注册探索运行，Goal仍active。第十三阶段已仅用3994张原训练图缓存两种弱变化特征（水平翻转、8像素反射填充平移），没有打开验证或测试图像。原特征用准备时的256编码批次重新核验，相对RMS为0；较小批次曾产生0.066%的CUDA数值差异，完整准备因此固定使用原批次。下一步才用这些实测变化构造新的噪声几何并训练控制，缓存不是验证性能结果。

```bash
python -m ps_pi.augmentation_pairs
python -m ps_pi.export_replication
python -m ps_pi.export_major
```
