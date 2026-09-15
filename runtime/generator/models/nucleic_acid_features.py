"""
核酸特异性特征增强模块
用于 SurfDock 框架，增强对核酸结构的建模能力
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from torch_geometric.data import Data
from scipy.spatial.distance import cdist
from loguru import logger


class NucleicAcidFeatureExtractor:
    """
    提取核酸特异性特征
    包括：二级结构、配对关系、backbone 构象等
    """
    
    def __init__(self):
        # 二级结构类型
        self.ss_types = [
            'stem', 'loop', 'bulge', 'internal_loop', 
            'hairpin', 'junction', 'pseudoknot', 'other'
        ]
        
        # 配对类型
        self.pairing_types = [
            'watson_crick', 'hoogsteen', 'sugar_edge', 
            'wobble', 'reverse_hoogsteen', 'no_pair', 'other'
        ]
        
        # 糖环 pucker 状态
        self.pucker_states = ['C2-endo', 'C3-endo', 'other']
        
        # Groove 类型
        self.groove_types = ['major', 'minor', 'backbone', 'other']
    
    def extract_residue_features(self, 
                                 residue_graph: Data,
                                 ss_info: Optional[Dict] = None,
                                 pairing_info: Optional[Dict] = None,
                                 backbone_info: Optional[Dict] = None) -> torch.Tensor:
        """
        提取残基级别的核酸特异性特征
        
        Args:
            residue_graph: 残基图（C4'节点）
            ss_info: 二级结构信息 {res_idx: ss_type}
            pairing_info: 配对信息 {res_idx: (partner_idx, pair_type)}
            backbone_info: backbone 构象信息 {res_idx: {pucker, torsions}}
        
        Returns:
            enhanced_features: [N_residues, feature_dim] 增强的特征
        """
        N = residue_graph.num_nodes
        features = []
        
        for i in range(N):
            feat = []
            
            # 1. 二级结构特征（one-hot）
            ss_type = ss_info.get(i, 'other') if ss_info else 'other'
            ss_onehot = [1 if ss_type == t else 0 for t in self.ss_types]
            feat.extend(ss_onehot)
            
            # 2. 配对关系特征
            if pairing_info and i in pairing_info:
                partner_idx, pair_type = pairing_info[i]
                # 配对类型 one-hot
                pair_onehot = [1 if pair_type == t else 0 for t in self.pairing_types]
                feat.extend(pair_onehot)
                # 配对距离（归一化）
                if partner_idx < N:
                    pair_dist = torch.norm(
                        residue_graph.pos[i] - residue_graph.pos[partner_idx]
                    ).item()
                    feat.append(pair_dist / 20.0)  # 归一化到 [0, 1]
                else:
                    feat.append(0.0)
            else:
                # 无配对
                pair_onehot = [0] * len(self.pairing_types)
                pair_onehot[-2] = 1  # 'no_pair'
                feat.extend(pair_onehot)
                feat.append(0.0)
            
            # 3. Backbone 构象特征
            if backbone_info and i in backbone_info:
                bb = backbone_info[i]
                # 糖环 pucker
                pucker_onehot = [1 if bb.get('pucker', 'other') == p else 0 
                                for p in self.pucker_states]
                feat.extend(pucker_onehot)
                # Torsion angles (α, β, γ, δ, ε, ζ, χ)
                torsions = bb.get('torsions', {})
                for angle_name in ['alpha', 'beta', 'gamma', 'delta', 'epsilon', 'zeta', 'chi']:
                    angle = torsions.get(angle_name, 0.0)
                    # 归一化到 [-1, 1]
                    feat.append(np.sin(angle * np.pi / 180))
                    feat.append(np.cos(angle * np.pi / 180))
            else:
                # 默认值
                feat.extend([0, 0, 1])  # pucker: other
                feat.extend([0, 1] * 7)  # torsions: 默认值
            
            # 4. 局部几何特征（相对于邻居）
            if i > 0:
                prev_dist = torch.norm(
                    residue_graph.pos[i] - residue_graph.pos[i-1]
                ).item()
                feat.append(prev_dist / 10.0)
            else:
                feat.append(0.0)
            
            if i < N - 1:
                next_dist = torch.norm(
                    residue_graph.pos[i+1] - residue_graph.pos[i]
                ).item()
                feat.append(next_dist / 10.0)
            else:
                feat.append(0.0)
            
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float32)
    
    def extract_surface_features(self,
                                surface_graph: Data,
                                residue_graph: Data,
                                groove_info: Optional[Dict] = None,
                                stacking_info: Optional[Dict] = None) -> torch.Tensor:
        """
        提取表面级别的核酸特异性特征
        
        Args:
            surface_graph: 表面图（MaSIF 特征）
            residue_graph: 残基图（用于计算到残基的距离）
            groove_info: Groove 类型信息 {surface_idx: groove_type}
            stacking_info: 堆叠信息 {surface_idx: stacking_potential}
        
        Returns:
            enhanced_features: [N_surface, feature_dim] 增强的特征
        """
        N_surf = surface_graph.num_nodes
        N_res = residue_graph.num_nodes
        
        # 计算每个表面点到最近残基的距离和类型
        surf_pos = surface_graph.pos
        res_pos = residue_graph.pos
        
        # 使用 KD-tree 快速查找最近邻
        distances = torch.cdist(surf_pos, res_pos)  # [N_surf, N_res]
        min_dist, nearest_res = torch.min(distances, dim=1)
        
        features = []
        for i in range(N_surf):
            feat = []
            
            # 1. Groove 类型特征
            groove_type = groove_info.get(i, 'other') if groove_info else 'other'
            groove_onehot = [1 if groove_type == g else 0 for g in self.groove_types]
            feat.extend(groove_onehot)
            
            # 2. 堆叠潜力
            stacking = stacking_info.get(i, 0.0) if stacking_info else 0.0
            feat.append(stacking)  # 归一化到 [0, 1]
            
            # 3. 到最近残基的距离（归一化）
            feat.append(min_dist[i].item() / 10.0)
            
            # 4. 局部曲率特征（如果 MaSIF 已提供，这里可以增强）
            # 假设 surface_graph.x 的前3个特征是 curvature, electrostatic, hydrophobic
            if surface_graph.x.shape[1] >= 3:
                feat.append(surface_graph.x[i, 0].item())  # curvature
                feat.append(surface_graph.x[i, 1].item())  # electrostatic
                feat.append(surface_graph.x[i, 2].item())  # hydrophobic
            else:
                feat.extend([0.0, 0.0, 0.0])
            
            # 5. 表面法向量与螺旋轴的角度（如果可计算）
            # 这里简化处理，实际需要计算螺旋轴
            feat.append(0.0)  # 占位符
            
            features.append(feat)
        
        return torch.tensor(features, dtype=torch.float32)


class NucleicAcidEncoder(nn.Module):
    """
    核酸特异性编码器
    将增强的特征编码为模型可用的嵌入
    """
    
    def __init__(self, 
                 residue_feature_dim: int,
                 surface_feature_dim: int,
                 emb_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.residue_encoder = nn.Sequential(
            nn.Linear(residue_feature_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        
        self.surface_encoder = nn.Sequential(
            nn.Linear(surface_feature_dim, emb_dim * 2),
            nn.LayerNorm(emb_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )
    
    def forward(self, residue_features: torch.Tensor, 
                surface_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码残基和表面特征
        
        Returns:
            residue_emb: [N_res, emb_dim]
            surface_emb: [N_surf, emb_dim]
        """
        residue_emb = self.residue_encoder(residue_features)
        surface_emb = self.surface_encoder(surface_features)
        return residue_emb, surface_emb


class SecondaryStructurePredictor(nn.Module):
    """
    二级结构预测模块（可选）
    可以从序列和结构预测二级结构
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, num_classes: int = 8):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.predictor(features)


class GrooveTypePredictor(nn.Module):
    """
    Groove 类型预测模块
    根据表面位置和几何特征预测 major/minor groove
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.predictor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, surface_features: torch.Tensor, 
                surface_pos: torch.Tensor,
                residue_pos: torch.Tensor) -> torch.Tensor:
        # 结合位置和特征信息
        pos_features = torch.cdist(surface_pos, residue_pos).min(dim=1)[0].unsqueeze(1)
        combined = torch.cat([surface_features, pos_features], dim=1)
        return self.predictor(combined)


class NucleicAcidAttentionLayer(nn.Module):
    """
    核酸特异性注意力层
    学习不同相互作用类型的重要性
    """
    
    def __init__(self, 
                 emb_dim: int,
                 num_heads: int = 4,
                 interaction_types: List[str] = None):
        super().__init__()
        
        if interaction_types is None:
            interaction_types = ['hydrophobic', 'electrostatic', 'hbond', 'stacking', 'groove']
        
        self.interaction_types = interaction_types
        self.num_heads = num_heads
        self.emb_dim = emb_dim
        
        # 为每种相互作用类型创建注意力头
        self.attention_heads = nn.ModuleDict({
            itype: nn.MultiheadAttention(emb_dim, num_heads, batch_first=True)
            for itype in interaction_types
        })
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim * len(interaction_types), emb_dim),
            nn.LayerNorm(emb_dim),
            nn.ReLU()
        )
    
    def forward(self, 
                query: torch.Tensor,
                key: torch.Tensor,
                value: torch.Tensor,
                interaction_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            query: [N, emb_dim] 查询（如配体特征）
            key: [M, emb_dim] 键（如表面特征）
            value: [M, emb_dim] 值（如表面特征）
            interaction_weights: [N, M, num_types] 相互作用权重（可选）
        
        Returns:
            attended_features: [N, emb_dim]
        """
        # 扩展维度用于多头注意力
        query = query.unsqueeze(0)  # [1, N, emb_dim]
        key = key.unsqueeze(0)  # [1, M, emb_dim]
        value = value.unsqueeze(0)  # [1, M, emb_dim]
        
        attended_list = []
        for itype in self.interaction_types:
            attn_out, _ = self.attention_heads[itype](query, key, value)
            attended_list.append(attn_out.squeeze(0))  # [N, emb_dim]
        
        # 融合所有注意力头
        combined = torch.cat(attended_list, dim=1)  # [N, emb_dim * num_types]
        output = self.fusion(combined)  # [N, emb_dim]
        
        return output


def compute_groove_type(surface_pos: torch.Tensor,
                       residue_pos: torch.Tensor,
                       helix_axis: Optional[torch.Tensor] = None) -> Dict[int, str]:
    """
    计算每个表面点的 groove 类型
    
    Args:
        surface_pos: [N_surf, 3] 表面点位置
        residue_pos: [N_res, 3] 残基位置（C4'）
        helix_axis: [3] 螺旋轴方向（可选）
    
    Returns:
        groove_dict: {surface_idx: groove_type}
    """
    groove_dict = {}
    
    # 简化实现：根据到残基的距离和角度判断
    # 实际实现需要更复杂的几何计算
    
    for i, surf_pos_i in enumerate(surface_pos):
        # 计算到最近残基的距离
        dists = torch.norm(residue_pos - surf_pos_i, dim=1)
        min_dist_idx = torch.argmin(dists)
        min_dist = dists[min_dist_idx]
        
        # 简化规则：距离较近且在特定角度范围内 -> major groove
        # 这里需要根据实际几何计算，暂时使用占位符
        if min_dist < 5.0:  # 阈值可调
            groove_dict[i] = 'major'
        elif min_dist < 8.0:
            groove_dict[i] = 'minor'
        else:
            groove_dict[i] = 'backbone'
    
    return groove_dict


def compute_stacking_potential(surface_pos: torch.Tensor,
                               residue_pos: torch.Tensor,
                               base_normals: Optional[torch.Tensor] = None) -> Dict[int, float]:
    """
    计算每个表面点的碱基堆叠潜力
    
    Args:
        surface_pos: [N_surf, 3] 表面点位置
        residue_pos: [N_res, 3] 残基位置
        base_normals: [N_res, 3] 碱基法向量（可选）
    
    Returns:
        stacking_dict: {surface_idx: stacking_potential}
    """
    stacking_dict = {}
    
    for i, surf_pos_i in enumerate(surface_pos):
        # 计算到最近残基的距离
        dists = torch.norm(residue_pos - surf_pos_i, dim=1)
        min_dist = torch.min(dists).item()
        
        # 简化：距离越近，堆叠潜力越高
        # 实际需要计算与碱基平面的角度
        stacking_potential = max(0.0, 1.0 - min_dist / 5.0)
        stacking_dict[i] = stacking_potential
    
    return stacking_dict


