#!/usr/bin/env python3
"""
测试核酸表面计算 - 使用compute_surface.py（基于computesurf.ipynb方法）
"""
import os
import sys

# 添加comp_surface到路径
comp_surface_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../comp_surface'))
sys.path.insert(0, comp_surface_dir)

print("=" * 60)
print("核酸表面计算测试")
print("=" * 60)
print("使用方法: 基于computesurf.ipynb中的方法")
print()

# 导入计算函数
try:
    from prepare_target.compute_surface import compute_surface_for_directory
    print("✓ compute_surface模块导入成功")
except ImportError as e:
    print(f"✗ 导入失败: {e}")
    sys.exit(1)

# 设置路径
input_dir = os.path.join(os.path.dirname(__file__), 'input')
output_dir = os.path.join(os.path.dirname(__file__), 'output')

print(f"\n输入目录: {input_dir}")
print(f"输出目录: {output_dir}")
print(f"\n测试结构数量: {len([d for d in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, d))])}")

# 运行计算
print("\n开始计算...")
print("-" * 60)

try:
    compute_surface_for_directory(
        data_dir=input_dir,
        out_dir=output_dir,
        surface_dist=8,  # 8A表面
        ligand_suffix='_ligand.sdf'
    )
    print("\n" + "=" * 60)
    print("测试完成！")
    print("=" * 60)
except Exception as e:
    print(f"\n错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
