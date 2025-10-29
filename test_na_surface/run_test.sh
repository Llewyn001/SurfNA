#!/bin/bash
# 运行核酸表面计算测试（基于computesurf.ipynb方法）

set -e

echo "=========================================="
echo "核酸表面计算测试"
echo "=========================================="

# 激活环境
source /root/miniconda3/etc/profile.d/conda.sh
conda activate SurfDock

# 进入测试目录
cd "$(dirname "$0")"

# 运行测试
python test_surface_na.py

echo ""
echo "测试完成！查看输出目录: output/"
