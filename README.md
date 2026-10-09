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
