# SurfNA 最小推理版本

SurfNA 通过共享分子表面将蛋白质–配体相互作用知识迁移到核酸–小分子对接。
本版提供当前论文对应的 Generator、Scorer 及必要 MDN 特征模型，面向指定结合位点，
生成 40 个候选构象并排序。训练数据、测试集、结构文件和历史实验输出不在本次发布范围内。

完整安装、输入要求、表面生成及命令见 [英文 README](README.md)。
本机和服务器工作区已放置相应权重；下载包的校验值见 `MODEL_CARD.json`。

```bash
conda env create -f environment.yml
conda activate surfna-v2
python scripts/install_weights.py --archive /path/to/surfna-v2-minimal-weights.tar.gz
python scripts/precompute.py
python scripts/check_models.py
python scripts/dock.py --receptor query_pocket.pdb --ligand query_ligand.sdf \
  --surface query.ply --output outputs/query --device cuda
```

推理环境验证于 Linux/CUDA；将仓库放在 Windows 上不等于已经验证 Windows 原生推理。
已有兼容表面时可直接使用；否则需要单独安装 APBS、PDB2PQR、MSMS 和 PyMesh。
当前模型使用 full8 表面，不能与早期四通道表面或 EMA350 版本的权重混用。

完整 40 候选流程及压缩包重新安装检查已通过，详见 `VALIDATION.md`。
项目自身代码及权重的许可证保留仓库现有待定状态；上游许可见 `licenses/`。
