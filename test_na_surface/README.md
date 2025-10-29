# 核酸表面计算测试

本目录包含从NAdata中选择的5个核酸结构进行从头表面计算测试。

## 测试结构

1. **100d** - 核酸-配体复合物
2. **101d** - 核酸-配体复合物
3. **102d** - 核酸-配体复合物
4. **107d** - 核酸-配体复合物
5. **108d** - 核酸-配体复合物

## 文件结构

```
test_na_surface/
├── input/              # 输入文件
│   ├── 100d/
│   │   ├── 100d_protein.pdb
│   │   └── 100d_ligand.sdf
│   └── ...
├── output/             # 输出文件（表面计算结果）
└── test_surface_computation.py  # 测试脚本
```

## 运行测试

### 准备工作

确保已激活SurfDock conda环境：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate SurfDock
```

### 运行测试

```bash
cd /root/autodl-tmp/SurfNA/test_na_surface
python test_surface_computation.py
```

## 预期输出

测试脚本会：
1. 读取每个结构的PDB和SDF文件
2. 调用comp_surface模块计算表面
3. 生成PLY格式的表面文件
4. 输出测试总结

## 注意事项

1. **pymesh依赖**: 如果遇到GLIBC版本问题，可能需要重新编译pymesh或使用conda版本
2. **APBS工具路径**: 已配置为使用comp_surface/tools中的工具
3. **计算时间**: 每个结构可能需要几分钟，取决于复杂度

## 输出文件说明

- `*.ply` - PLY格式的表面网格文件
- `*_8A.pdb` - 8埃范围内的口袋结构
- 其他临时文件会在计算完成后自动清理

## 故障排除

如果遇到问题：

1. 检查APBS工具是否可执行：
   ```bash
   ls -la ../comp_surface/tools/transfer/APBS-3.4.1.Linux/bin/apbs
   ```

2. 检查pymesh是否正确安装：
   ```bash
   python -c "import pymesh; print(pymesh.__version__)"
   ```

3. 查看详细错误日志：`test_run.log`
