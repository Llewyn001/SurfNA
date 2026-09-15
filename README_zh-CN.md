# SurfNA v2

本仓库提供 Generator、冻结 MDN 特征提取与最终 W0 Scorer 的代码、配置和数据清单。模型权重通过 GitHub Releases 单独下载。

- [安装与推理](README.md)
- [PDB 实例清单](datasets/nucleic_acid_instances.csv)：801 个训练、89 个验证、128 个测试实例，共 699 个 PDB 条目。
- [数据处理说明](docs/DATA.md)：配体、受体链/残基、CCD 化学修正、表面计算、去重与划分。
- [Checkpoint 说明](docs/CHECKPOINTS.md)：默认模型及主实验另外两个训练重复。
- [论文复现说明](docs/REPRODUCIBILITY.md)：实验协议与 Figure 2–4 数值源数据。

读者自行从 PDB 下载结构并生成表面、图缓存。训练特征、候选构象库、处理后的完整数据集和所有中间 epoch 权重均不随本次轻量版本发布。保留了原始样本顺序、嵌套子集和相应方法代码。

本次发布整理没有新增模型运行、benchmark 或训练检查。历史脚本的路径约定和当前公开资产范围见复现说明。
