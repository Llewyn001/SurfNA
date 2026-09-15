"""
注意力可视化工具 - 针对构象生成模型（Diffusion模型）
用于分析模型在处理RNA和Protein时关注的几何特征（曲率/疏水斑块）
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import os
from torch_geometric.utils import to_dense_batch
from torch_scatter import scatter_mean, scatter_add


class AttentionExtractor:
    """从构象生成模型（Diffusion模型）中提取注意力信息"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def extract_attention_weights(self, data, timestep=None, return_details=False):
        """
        提取注意力权重
        
        Args:
            data: 输入数据（包含ligand, receptor, surface）
            timestep: 扩散时间步（如果为None，使用data中的时间步）
            return_details: 是否返回详细信息
        
        Returns:
            attention_dict: 包含注意力权重的字典
        """
        with torch.no_grad():
            # 设置时间步
            if timestep is not None:
                data.complex_t = {
                    'tr': torch.tensor([timestep], device=self.device),
                    'rot': torch.tensor([timestep], device=self.device),
                    'tor': torch.tensor([timestep], device=self.device)
                }
            
            # 获取时间步相关的sigma
            tr_sigma, rot_sigma, tor_sigma = self.model.t_to_sigma(
                *[data.complex_t[noise_type] for noise_type in ['tr', 'rot', 'tor']]
            )
            
            # 构建ligand图
            lig_node_attr, lig_edge_index, lig_edge_attr, lig_edge_sh = self.model.build_lig_conv_graph(data)
            lig_src, lig_dst = lig_edge_index
            lig_node_attr = self.model.lig_node_embedding(lig_node_attr)
            if hasattr(self.model, 'lig_node_adapter') and self.model.lig_node_adapter is not None:
                lig_node_attr = self.model.lig_node_adapter(lig_node_attr, lig_node_attr)
            lig_edge_attr = self.model.lig_edge_embedding(lig_edge_attr)
            if hasattr(self.model, 'lig_edge_adapter') and self.model.lig_edge_adapter is not None:
                lig_edge_attr = self.model.lig_edge_adapter(lig_edge_attr, lig_edge_attr)
            
            # 构建receptor图
            rec_node_attr, rec_edge_index, rec_edge_attr, rec_edge_sh = self.model.build_rec_conv_graph(data)
            rec_src, rec_dst = rec_edge_index
            rec_node_attr = self.model.rec_node_embedding(rec_node_attr)
            if hasattr(self.model, 'use_nucleic_feat_fusion') and self.model.use_nucleic_feat_fusion:
                if hasattr(data['receptor'], 'nucleic_feat'):
                    try:
                        nuc_feat = data['receptor'].nucleic_feat
                        if torch.is_tensor(nuc_feat):
                            nuc_feat = nuc_feat.to(rec_node_attr.device).float()
                            rec_node_attr = rec_node_attr + self.model.nuc_feat_gate(nuc_feat) * self.model.nuc_feat_proj(nuc_feat)
                    except Exception:
                        pass
            rec_edge_attr = self.model.rec_edge_embedding(rec_edge_attr)
            
            # 构建surface图
            surface_node_attr, surface_edge_index, surface_edge_attr, surface_edge_sh, masked_input, mask = \
                self.model.build_surface_conv_graph(data, mode=False)
            surface_src, surface_dst = surface_edge_index
            surface_node_attr = self.model.surface_node_embedding(surface_node_attr)
            if hasattr(self.model, 'surface_node_adapter') and self.model.surface_node_adapter is not None:
                surface_node_attr = self.model.surface_node_adapter(surface_node_attr, surface_node_attr)
            surface_edge_attr = self.model.surface_edge_embedding(surface_edge_attr)
            if hasattr(self.model, 'surface_edge_adapter') and self.model.surface_edge_adapter is not None:
                surface_edge_attr = self.model.surface_edge_adapter(surface_edge_attr, surface_edge_attr)
            
            # 构建cross graph（ligand-surface交互）
            if self.model.dynamic_max_cross:
                cross_cutoff = (tr_sigma * 3 + 10).unsqueeze(1)
            else:
                cross_cutoff = self.model.cross_max_distance
            
            surface_cross_edge_index, surface_cross_edge_attr, surface_cross_edge_sh = \
                self.model.build_surface_cross_conv_graph(data, cross_cutoff)
            surface_cross_lig, surface_cross_rec = surface_cross_edge_index
            surface_cross_edge_attr = self.model.cross_edge_embedding(surface_cross_edge_attr)
            if hasattr(self.model, 'cross_edge_adapter') and self.model.cross_edge_adapter is not None:
                surface_cross_edge_attr = self.model.cross_edge_adapter(surface_cross_edge_attr, surface_cross_edge_attr)
            
            # 更新receptor和surface特征
            rec_edge_attr_ = torch.cat([rec_edge_attr, rec_node_attr[rec_src, :self.model.ns], 
                                       rec_node_attr[rec_dst, :self.model.ns]], -1)
            rec_intra_update = self.model.rec_conv_layers[0](rec_node_attr, rec_edge_index, 
                                                             rec_edge_attr_, rec_edge_sh)
            rec_node_attr = torch.nn.functional.pad(rec_node_attr, 
                                                    (0, rec_intra_update.shape[-1] - rec_node_attr.shape[-1]))
            rec_node_attr = rec_node_attr + rec_intra_update
            
            # Residue to surface
            surface_rec_cross_edge_index, surface_rec_cross_edge_attr, surface_rec_cross_edge_sh = \
                self.model.build_surface_rec_cross_conv_graph(data)
            surface_rec_cross_rec, surface_rec_cross_surface = surface_rec_cross_edge_index
            surface_rec_cross_edge_attr = self.model.surface_rec_cross_edge_embedding(surface_rec_cross_edge_attr)
            
            residue_to_surface_edge_attr_ = torch.cat([
                surface_rec_cross_edge_attr, 
                rec_node_attr[surface_rec_cross_rec, :self.model.ns], 
                surface_node_attr[surface_rec_cross_surface, :self.model.ns]
            ], -1)
            surface_inter_residue_update = self.model.residue_to_surface_conv_layers[0](
                rec_node_attr, torch.flip(surface_rec_cross_edge_index, dims=[0]), 
                residue_to_surface_edge_attr_, surface_rec_cross_edge_sh,
                out_nodes=surface_node_attr.shape[0])
            
            if hasattr(self.model, 'residue_to_surface_adapter') and self.model.residue_to_surface_adapter is not None:
                surface_node_attr_padded = torch.nn.functional.pad(surface_node_attr, 
                                                                   (0, surface_inter_residue_update.shape[-1] - surface_node_attr.shape[-1]))
                surface_inter_residue_update = self.model.residue_to_surface_adapter(surface_node_attr_padded, surface_inter_residue_update)
            
            surface_node_attr = torch.nn.functional.pad(surface_node_attr, 
                                                       (0, surface_inter_residue_update.shape[-1] - surface_node_attr.shape[-1]))
            surface_node_attr = surface_node_attr + surface_inter_residue_update
            
            # 存储每层的注意力权重
            attention_weights_per_layer = []
            
            # 逐层处理
            for l in range(len(self.model.lig_conv_layers)):
                # Ligand intra graph
                lig_edge_attr_ = torch.cat([lig_edge_attr, lig_node_attr[lig_src, :self.model.ns], 
                                           lig_node_attr[lig_dst, :self.model.ns]], -1)
                lig_intra_update = self.model.lig_conv_layers[l](lig_node_attr, lig_edge_index, 
                                                                 lig_edge_attr_, lig_edge_sh)
                
                # Surface to ligand cross attention（关键：这是主要的注意力机制）
                surface_to_lig_edge_attr_ = torch.cat([
                    surface_cross_edge_attr, 
                    lig_node_attr[surface_cross_lig, :self.model.ns], 
                    surface_node_attr[surface_cross_rec, :self.model.ns]
                ], -1)
                surface_lig_inter_update = self.model.surface_to_lig_conv_layers[l](
                    surface_node_attr, surface_cross_edge_index, surface_to_lig_edge_attr_, 
                    surface_cross_edge_sh, out_nodes=lig_node_attr.shape[0])
                
                # 提取注意力权重：使用edge features的norm作为注意力权重
                # 计算cross edge的重要性
                cross_edge_importance = torch.norm(surface_cross_edge_attr, dim=-1)
                
                # 聚合到surface节点
                surface_attention = scatter_add(
                    cross_edge_importance, 
                    surface_cross_rec, 
                    dim=0, 
                    dim_size=surface_node_attr.shape[0]
                )
                
                # 聚合到ligand节点
                ligand_attention = scatter_add(
                    cross_edge_importance,
                    surface_cross_lig,
                    dim=0,
                    dim_size=lig_node_attr.shape[0]
                )
                
                attention_weights_per_layer.append({
                    'surface_attention': surface_attention,
                    'ligand_attention': ligand_attention,
                    'cross_edge_importance': cross_edge_importance,
                    'cross_edge_index': surface_cross_edge_index,
                    'layer': l
                })
                
                lig_node_attr = torch.nn.functional.pad(lig_node_attr, 
                                                        (0, lig_intra_update.shape[-1] - lig_node_attr.shape[-1]))
                lig_node_attr = lig_node_attr + lig_intra_update + surface_lig_inter_update
                
                if l != len(self.model.lig_conv_layers) - 1:
                    # Surface intra graph
                    surface_edge_attr_ = torch.cat([surface_edge_attr, 
                                                   surface_node_attr[surface_src, :self.model.ns], 
                                                   surface_node_attr[surface_dst, :self.model.ns]], -1)
                    surface_intra_update = self.model.surface_conv_layers[l](surface_node_attr, 
                                                                            surface_edge_index, 
                                                                            surface_edge_attr_, surface_edge_sh)
                    
                    # Lig to surface cross attention
                    lig_to_surface_edge_attr_ = torch.cat([
                        surface_cross_edge_attr, 
                        lig_node_attr[surface_cross_lig, :self.model.ns],
                        surface_node_attr[surface_cross_rec, :self.model.ns]
                    ], -1)
                    surface_inter_update = self.model.lig_to_surface_conv_layers[l](
                        lig_node_attr, torch.flip(surface_cross_edge_index, dims=[0]), 
                        lig_to_surface_edge_attr_, surface_cross_edge_sh,
                        out_nodes=surface_node_attr.shape[0])
                    
                    surface_node_attr = torch.nn.functional.pad(surface_node_attr, 
                                                               (0, surface_intra_update.shape[-1] - surface_node_attr.shape[-1]))
                    surface_node_attr = surface_node_attr + surface_intra_update + surface_inter_update
            
            # 使用最后一层的注意力权重
            final_attention = attention_weights_per_layer[-1]
            
            # 转换为dense batch格式以便可视化
            h_surface_attn, surface_mask = to_dense_batch(
                final_attention['surface_attention'], 
                data['surface'].batch, 
                fill_value=0
            )
            h_ligand_attn, ligand_mask = to_dense_batch(
                final_attention['ligand_attention'],
                data['ligand'].batch,
                fill_value=0
            )
            h_surface_pos, _ = to_dense_batch(data['surface'].pos, data['surface'].batch, fill_value=0)
            h_ligand_pos, _ = to_dense_batch(data['ligand'].pos, data['ligand'].batch, fill_value=0)
            
            # 提取surface节点的原始几何特征
            surface_features = data['surface'].x.cpu().numpy()  # [N_surface, 3] - 曲率、疏水性等
            
            result = {
                'surface_attention': h_surface_attn.cpu().numpy(),  # [B, N_surface]
                'ligand_attention': h_ligand_attn.cpu().numpy(),  # [B, N_ligand]
                'surface_positions': h_surface_pos.cpu().numpy(),  # [B, N_surface, 3]
                'ligand_positions': h_ligand_pos.cpu().numpy(),  # [B, N_ligand, 3]
                'surface_features': surface_features,  # [N_surface, 3] - 曲率、疏水性等
                'timestep': tr_sigma.item() if torch.is_tensor(tr_sigma) else tr_sigma,
                'attention_per_layer': attention_weights_per_layer,
            }
            
            if return_details:
                result.update({
                    'ligand_features': lig_node_attr.cpu().numpy(),
                    'surface_features_emb': surface_node_attr.cpu().numpy(),
                    'cross_edge_importance': final_attention['cross_edge_importance'].cpu().numpy(),
                    'cross_edge_index': final_attention['cross_edge_index'].cpu().numpy(),
                })
            
            return result
    
    def extract_attention_trajectory(self, data, timesteps: List[float]):
        """
        提取不同时间步的注意力轨迹（用于可视化生成过程）
        
        Args:
            data: 输入数据
            timesteps: 时间步列表
        
        Returns:
            trajectory: 包含每个时间步注意力信息的列表
        """
        trajectory = []
        for t in timesteps:
            attention_data = self.extract_attention_weights(data, timestep=t)
            trajectory.append(attention_data)
        return trajectory


class AttentionVisualizer:
    """可视化注意力权重"""
    
    def __init__(self, save_dir='attention_visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def plot_attention_on_surface(self, attention_data: Dict, 
                                   surface_type: str = 'protein',
                                   save_path: Optional[str] = None,
                                   title: Optional[str] = None):
        """
        在3D表面上绘制注意力权重
        
        Args:
            attention_data: 从AttentionExtractor提取的数据
            surface_type: 'protein' 或 'rna'
            save_path: 保存路径
            title: 图标题
        """
        surface_attention = attention_data['surface_attention'][0]  # 取第一个batch
        surface_pos = attention_data['surface_positions'][0]  # [N_surface, 3]
        surface_features = attention_data['surface_features']  # [N_surface, 3]
        
        # 创建3D图
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 归一化注意力权重
        attention_norm = (surface_attention - surface_attention.min()) / \
                        (surface_attention.max() - surface_attention.min() + 1e-10)
        
        # 根据注意力权重设置颜色
        scatter = ax.scatter(surface_pos[:, 0], surface_pos[:, 1], surface_pos[:, 2],
                            c=attention_norm, cmap='hot', s=50, alpha=0.6,
                            vmin=0, vmax=1)
        
        # 添加配体位置
        if 'ligand_positions' in attention_data:
            lig_pos = attention_data['ligand_positions'][0]
            ax.scatter(lig_pos[:, 0], lig_pos[:, 1], lig_pos[:, 2],
                      c='blue', s=100, alpha=0.8, marker='o', label='Ligand')
        
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title(title or f'Attention Map - {surface_type.upper()} (t={attention_data["timestep"]:.2f})')
        plt.colorbar(scatter, ax=ax, label='Attention Weight')
        ax.legend()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'attention_3d_{surface_type}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_geometric_feature_correlation(self, attention_data: Dict,
                                           surface_type: str = 'protein',
                                           save_path: Optional[str] = None):
        """
        绘制注意力权重与几何特征（曲率、疏水性）的相关性
        
        Args:
            attention_data: 从AttentionExtractor提取的数据
            surface_type: 'protein' 或 'rna'
            save_path: 保存路径
        """
        surface_attention = attention_data['surface_attention'][0]
        surface_features = attention_data['surface_features']  # [N_surface, 3]
        
        # 如果维度不匹配，需要处理
        if surface_features.shape[0] != len(surface_attention):
            print(f"Warning: Feature dimension mismatch. Features: {surface_features.shape[0]}, Attention: {len(surface_attention)}")
            return
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        # 曲率 vs 注意力
        axes[0].scatter(surface_features[:, 0], surface_attention, alpha=0.6)
        axes[0].set_xlabel('Curvature')
        axes[0].set_ylabel('Attention Weight')
        axes[0].set_title('Attention vs Curvature')
        axes[0].grid(True, alpha=0.3)
        
        # 疏水性 vs 注意力
        axes[1].scatter(surface_features[:, 1], surface_attention, alpha=0.6, color='orange')
        axes[1].set_xlabel('Hydrophobicity')
        axes[1].set_ylabel('Attention Weight')
        axes[1].set_title('Attention vs Hydrophobicity')
        axes[1].grid(True, alpha=0.3)
        
        # 第三个特征 vs 注意力
        if surface_features.shape[1] >= 3:
            axes[2].scatter(surface_features[:, 2], surface_attention, alpha=0.6, color='green')
            axes[2].set_xlabel('Feature 3')
            axes[2].set_ylabel('Attention Weight')
            axes[2].set_title('Attention vs Feature 3')
            axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Geometric Feature Correlation - {surface_type.upper()}')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'feature_correlation_{surface_type}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def compare_rna_protein_attention(self, rna_attention: Dict, protein_attention: Dict,
                                     save_path: Optional[str] = None):
        """
        对比RNA和Protein的注意力模式
        
        Args:
            rna_attention: RNA的注意力数据
            protein_attention: Protein的注意力数据
            save_path: 保存路径
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # RNA注意力分布
        rna_attn = rna_attention['surface_attention'][0]
        axes[0, 0].hist(rna_attn, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[0, 0].set_xlabel('Attention Weight')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('RNA Attention Distribution')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Protein注意力分布
        protein_attn = protein_attention['surface_attention'][0]
        axes[0, 1].hist(protein_attn, bins=50, alpha=0.7, color='blue', edgecolor='black')
        axes[0, 1].set_xlabel('Attention Weight')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Protein Attention Distribution')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 几何特征相关性对比
        rna_features = rna_attention['surface_features']
        protein_features = protein_attention['surface_features']
        
        # 曲率相关性
        axes[1, 0].scatter(rna_features[:, 0], rna_attn, alpha=0.5, label='RNA', color='red', s=20)
        axes[1, 0].scatter(protein_features[:, 0], protein_attn, alpha=0.5, label='Protein', color='blue', s=20)
        axes[1, 0].set_xlabel('Curvature')
        axes[1, 0].set_ylabel('Attention Weight')
        axes[1, 0].set_title('Attention vs Curvature (Comparison)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 疏水性相关性
        axes[1, 1].scatter(rna_features[:, 1], rna_attn, alpha=0.5, label='RNA', color='red', s=20)
        axes[1, 1].scatter(protein_features[:, 1], protein_attn, alpha=0.5, label='Protein', color='blue', s=20)
        axes[1, 1].set_xlabel('Hydrophobicity')
        axes[1, 1].set_ylabel('Attention Weight')
        axes[1, 1].set_title('Attention vs Hydrophobicity (Comparison)')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('RNA vs Protein Attention Pattern Comparison', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'rna_protein_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_transfer_learning_visualization(self, rna_attention: Dict, protein_attention: Dict,
                                              save_path: Optional[str] = None):
        """
        创建Transfer Learning可视化，展示模型关注的几何特征在RNA和Protein上的一致性
        
        Args:
            rna_attention: RNA的注意力数据
            protein_attention: Protein的注意力数据
            save_path: 保存路径
        """
        fig = plt.figure(figsize=(20, 10))
        
        # 创建子图网格
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)
        
        # 第一行：RNA
        ax1 = fig.add_subplot(gs[0, 0], projection='3d')
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[0, 2])
        ax4 = fig.add_subplot(gs[0, 3])
        
        # 第二行：Protein
        ax5 = fig.add_subplot(gs[1, 0], projection='3d')
        ax6 = fig.add_subplot(gs[1, 1])
        ax7 = fig.add_subplot(gs[1, 2])
        ax8 = fig.add_subplot(gs[1, 3])
        
        # RNA 3D注意力图
        rna_attn = rna_attention['surface_attention'][0]
        rna_pos = rna_attention['surface_positions'][0]
        rna_attn_norm = (rna_attn - rna_attn.min()) / (rna_attn.max() - rna_attn.min() + 1e-10)
        ax1.scatter(rna_pos[:, 0], rna_pos[:, 1], rna_pos[:, 2], 
                   c=rna_attn_norm, cmap='hot', s=30, alpha=0.6)
        ax1.set_title('RNA: 3D Attention Map', fontsize=12)
        ax1.set_xlabel('X (Å)')
        ax1.set_ylabel('Y (Å)')
        ax1.set_zlabel('Z (Å)')
        
        # RNA 曲率相关性
        rna_features = rna_attention['surface_features']
        ax2.scatter(rna_features[:, 0], rna_attn, alpha=0.6, s=20, color='red')
        ax2.set_xlabel('Curvature')
        ax2.set_ylabel('Attention Weight')
        ax2.set_title('RNA: Attention vs Curvature')
        ax2.grid(True, alpha=0.3)
        
        # RNA 疏水性相关性
        ax3.scatter(rna_features[:, 1], rna_attn, alpha=0.6, s=20, color='orange')
        ax3.set_xlabel('Hydrophobicity')
        ax3.set_ylabel('Attention Weight')
        ax3.set_title('RNA: Attention vs Hydrophobicity')
        ax3.grid(True, alpha=0.3)
        
        # RNA 注意力分布
        ax4.hist(rna_attn, bins=50, alpha=0.7, color='red', edgecolor='black')
        ax4.set_xlabel('Attention Weight')
        ax4.set_ylabel('Frequency')
        ax4.set_title('RNA: Attention Distribution')
        ax4.grid(True, alpha=0.3)
        
        # Protein 3D注意力图
        protein_attn = protein_attention['surface_attention'][0]
        protein_pos = protein_attention['surface_positions'][0]
        protein_attn_norm = (protein_attn - protein_attn.min()) / (protein_attn.max() - protein_attn.min() + 1e-10)
        ax5.scatter(protein_pos[:, 0], protein_pos[:, 1], protein_pos[:, 2], 
                   c=protein_attn_norm, cmap='hot', s=30, alpha=0.6)
        ax5.set_title('Protein: 3D Attention Map', fontsize=12)
        ax5.set_xlabel('X (Å)')
        ax5.set_ylabel('Y (Å)')
        ax5.set_zlabel('Z (Å)')
        
        # Protein 曲率相关性
        protein_features = protein_attention['surface_features']
        ax6.scatter(protein_features[:, 0], protein_attn, alpha=0.6, s=20, color='blue')
        ax6.set_xlabel('Curvature')
        ax6.set_ylabel('Attention Weight')
        ax6.set_title('Protein: Attention vs Curvature')
        ax6.grid(True, alpha=0.3)
        
        # Protein 疏水性相关性
        ax7.scatter(protein_features[:, 1], protein_attn, alpha=0.6, s=20, color='cyan')
        ax7.set_xlabel('Hydrophobicity')
        ax7.set_ylabel('Attention Weight')
        ax7.set_title('Protein: Attention vs Hydrophobicity')
        ax7.grid(True, alpha=0.3)
        
        # Protein 注意力分布
        ax8.hist(protein_attn, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax8.set_xlabel('Attention Weight')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Protein: Attention Distribution')
        ax8.grid(True, alpha=0.3)
        
        plt.suptitle('Transfer Learning Visualization: Consistent Geometric Feature Attention\n(RNA vs Protein)', 
                    fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, 'transfer_learning_visualization.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_attention_trajectory(self, trajectory: List[Dict], surface_type: str = 'protein',
                                 save_path: Optional[str] = None):
        """
        可视化不同时间步的注意力变化轨迹（展示生成过程）
        
        Args:
            trajectory: 从extract_attention_trajectory获取的轨迹数据
            surface_type: 'protein' 或 'rna'
            save_path: 保存路径
        """
        n_timesteps = len(trajectory)
        fig, axes = plt.subplots(2, (n_timesteps + 1) // 2, figsize=(5 * ((n_timesteps + 1) // 2), 10))
        if n_timesteps == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        for idx, attention_data in enumerate(trajectory):
            ax = axes[idx]
            surface_attention = attention_data['surface_attention'][0]
            timestep = attention_data['timestep']
            
            ax.hist(surface_attention, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Attention Weight')
            ax.set_ylabel('Frequency')
            ax.set_title(f't={timestep:.2f}')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for idx in range(n_timesteps, len(axes)):
            axes[idx].axis('off')
        
        plt.suptitle(f'Attention Trajectory - {surface_type.upper()}', fontsize=16)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(os.path.join(self.save_dir, f'attention_trajectory_{surface_type}.png'), 
                       dpi=300, bbox_inches='tight')
        plt.close()


def visualize_attention_for_transfer_learning(model, rna_data, protein_data, 
                                              device, save_dir='attention_visualizations',
                                              timesteps: Optional[List[float]] = None):
    """
    完整的可视化流程
    
    Args:
        model: 训练好的构象生成模型
        rna_data: RNA数据
        protein_data: Protein数据
        device: 设备
        save_dir: 保存目录
        timesteps: 可选的时间步列表，用于轨迹可视化
    """
    extractor = AttentionExtractor(model, device)
    visualizer = AttentionVisualizer(save_dir)
    
    # 提取注意力（使用默认时间步或指定时间步）
    print("Extracting RNA attention...")
    rna_attention = extractor.extract_attention_weights(rna_data)
    
    print("Extracting Protein attention...")
    protein_attention = extractor.extract_attention_weights(protein_data)
    
    # 可视化
    print("Creating visualizations...")
    visualizer.plot_attention_on_surface(rna_attention, 'rna', 
                                         title='RNA: Attention Map on Surface')
    visualizer.plot_attention_on_surface(protein_attention, 'protein',
                                         title='Protein: Attention Map on Surface')
    
    visualizer.plot_geometric_feature_correlation(rna_attention, 'rna')
    visualizer.plot_geometric_feature_correlation(protein_attention, 'protein')
    
    visualizer.compare_rna_protein_attention(rna_attention, protein_attention)
    visualizer.create_transfer_learning_visualization(rna_attention, protein_attention)
    
    # 如果提供了时间步，创建轨迹可视化
    if timesteps is not None:
        print("Creating attention trajectories...")
        rna_trajectory = extractor.extract_attention_trajectory(rna_data, timesteps)
        protein_trajectory = extractor.extract_attention_trajectory(protein_data, timesteps)
        
        visualizer.plot_attention_trajectory(rna_trajectory, 'rna')
        visualizer.plot_attention_trajectory(protein_trajectory, 'protein')
    
    print(f"Visualizations saved to {save_dir}")