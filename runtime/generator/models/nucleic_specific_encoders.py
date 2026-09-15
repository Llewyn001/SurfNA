"""
核酸特异性特征编码器模块
用于增强 SurfDock 对核酸结构的建模能力
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Optional, Tuple
import math


class BaseTypeEncoder(nn.Module):
    """
    碱基类型编码器
    扩展核苷酸类型、二级结构、配对关系等特征
    """
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        # 标准核苷酸类型 (A, C, G, U, T, X等)
        self.base_type_embedding = nn.Embedding(20, emb_dim)  # 支持标准+修饰碱基
        
        # 二级结构类型 (stem, loop, bulge, internal_loop, hairpin, etc.)
        self.secondary_structure_embedding = nn.Embedding(10, emb_dim // 2)
        
        # 配对状态 (unpaired, Watson-Crick, Hoogsteen, Sugar-edge, etc.)
        self.pairing_type_embedding = nn.Embedding(8, emb_dim // 2)
        
        # 是否参与 stacking / base-triple / noncanonical pair
        self.stacking_embedding = nn.Embedding(3, emb_dim // 4)  # no, stacking, triple
        self.noncanonical_embedding = nn.Embedding(2, emb_dim // 4)  # canonical, noncanonical
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim + emb_dim // 2 + emb_dim // 2 + emb_dim // 4 + emb_dim // 4, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )
    
    def forward(self, base_types: torch.Tensor, secondary_struct: torch.Tensor, 
                pairing_types: torch.Tensor, stacking: torch.Tensor, 
                noncanonical: torch.Tensor) -> torch.Tensor:
        """
        Args:
            base_types: [N] 碱基类型索引
            secondary_struct: [N] 二级结构类型索引
            pairing_types: [N] 配对类型索引
            stacking: [N] stacking状态索引
            noncanonical: [N] 是否非典型配对
        Returns:
            [N, emb_dim] 编码后的特征
        """
        base_emb = self.base_type_embedding(base_types)
        ss_emb = self.secondary_structure_embedding(secondary_struct)
        pair_emb = self.pairing_type_embedding(pairing_types)
        stack_emb = self.stacking_embedding(stacking)
        noncan_emb = self.noncanonical_embedding(noncanonical)
        
        fused = torch.cat([base_emb, ss_emb, pair_emb, stack_emb, noncan_emb], dim=-1)
        return self.fusion(fused)


class BackboneEncoder(nn.Module):
    """
    核酸骨架编码器
    利用 backbone torsion (α, β, γ, δ, ε, ζ, χ) 和 sugar pucker (C3'-endo / C2'-endo)
    """
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        # Torsion angles: α, β, γ, δ, ε, ζ, χ (7个角度)
        self.torsion_encoder = nn.Sequential(
            nn.Linear(7, emb_dim // 2),
            nn.ReLU(),
            nn.LayerNorm(emb_dim // 2),
            nn.Linear(emb_dim // 2, emb_dim // 2)
        )
        
        # Sugar pucker: C3'-endo (A-form) vs C2'-endo (B-form)
        # 使用伪扭转角 (pseudorotation phase) 和振幅
        self.pucker_encoder = nn.Sequential(
            nn.Linear(2, emb_dim // 4),  # phase + amplitude
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 骨架几何特征 (C4'-C3'-O3'-P 等关键距离)
        self.geometry_encoder = nn.Sequential(
            nn.Linear(5, emb_dim // 4),  # 关键距离特征
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim // 2 + emb_dim // 4 + emb_dim // 4, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )
    
    def forward(self, torsion_angles: torch.Tensor, pucker_params: torch.Tensor,
                geometry_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            torsion_angles: [N, 7] α, β, γ, δ, ε, ζ, χ (单位: 弧度)
            pucker_params: [N, 2] pseudorotation phase 和 amplitude
            geometry_features: [N, 5] 关键几何距离特征
        Returns:
            [N, emb_dim] 编码后的特征
        """
        torsion_emb = self.torsion_encoder(torsion_angles)
        pucker_emb = self.pucker_encoder(pucker_params)
        geom_emb = self.geometry_encoder(geometry_features)
        
        fused = torch.cat([torsion_emb, pucker_emb, geom_emb], dim=-1)
        return self.fusion(fused)


class HBondEncoder(nn.Module):
    """
    氢键网络编码器
    编码每个碱基的氢键模式
    """
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        # 氢键数量统计
        self.hbond_count_encoder = nn.Sequential(
            nn.Linear(3, emb_dim // 2),  # Watson-Crick, Hoogsteen, Sugar-edge 数量
            nn.ReLU(),
            nn.Linear(emb_dim // 2, emb_dim // 2)
        )
        
        # 是否参与配体相关的氢键
        self.ligand_hbond_encoder = nn.Sequential(
            nn.Linear(2, emb_dim // 4),  # donor, acceptor
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 氢键强度/距离特征
        self.hbond_strength_encoder = nn.Sequential(
            nn.Linear(3, emb_dim // 4),  # 平均距离、最小距离、数量
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim // 2 + emb_dim // 4 + emb_dim // 4, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )
    
    def forward(self, hbond_counts: torch.Tensor, ligand_hbond: torch.Tensor,
                hbond_strength: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hbond_counts: [N, 3] Watson-Crick, Hoogsteen, Sugar-edge 氢键数量
            ligand_hbond: [N, 2] 是否作为配体氢键的 donor/acceptor
            hbond_strength: [N, 3] 平均距离、最小距离、总数量
        Returns:
            [N, emb_dim] 编码后的特征
        """
        count_emb = self.hbond_count_encoder(hbond_counts)
        lig_emb = self.ligand_hbond_encoder(ligand_hbond)
        strength_emb = self.hbond_strength_encoder(hbond_strength)
        
        fused = torch.cat([count_emb, lig_emb, strength_emb], dim=-1)
        return self.fusion(fused)


class FlexibilityEncoder(nn.Module):
    """
    柔性特征编码器
    编码 RNA 的柔性特征，对 pose 预测很重要
    """
    def __init__(self, emb_dim: int = 32):
        super().__init__()
        # B-factor (晶体结构中的温度因子)
        self.bfactor_encoder = nn.Sequential(
            nn.Linear(1, emb_dim // 4),
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 局部 RMSF (通过结构邻域估计)
        self.rmsf_encoder = nn.Sequential(
            nn.Linear(1, emb_dim // 4),
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 二级结构环境 (stem通常刚性，loop/bulge通常柔性)
        self.ss_rigidity_encoder = nn.Embedding(5, emb_dim // 4)  # stem, loop, bulge, internal_loop, other
        
        # 局部构象多样性 (通过多构象或MD轨迹计算)
        self.conformational_diversity_encoder = nn.Sequential(
            nn.Linear(1, emb_dim // 4),
            nn.ReLU(),
            nn.Linear(emb_dim // 4, emb_dim // 4)
        )
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(emb_dim // 4 * 4, emb_dim),
            nn.ReLU(),
            nn.LayerNorm(emb_dim)
        )
    
    def forward(self, bfactor: torch.Tensor, rmsf: torch.Tensor,
                ss_rigidity: torch.Tensor, conf_diversity: torch.Tensor) -> torch.Tensor:
        """
        Args:
            bfactor: [N, 1] B-factor值
            rmsf: [N, 1] 局部RMSF值
            ss_rigidity: [N] 二级结构刚性类型索引
            conf_diversity: [N, 1] 构象多样性指标
        Returns:
            [N, emb_dim] 编码后的特征
        """
        bf_emb = self.bfactor_encoder(bfactor)
        rmsf_emb = self.rmsf_encoder(rmsf)
        ss_emb = self.ss_rigidity_encoder(ss_rigidity)
        conf_emb = self.conformational_diversity_encoder(conf_diversity)
        
        fused = torch.cat([bf_emb, rmsf_emb, ss_emb, conf_emb], dim=-1)
        return self.fusion(fused)


class NucleicSpecificEncoder(nn.Module):
    """
    核酸特异性特征编码器主模块
    整合所有核酸特定特征
    """
    def __init__(self, base_emb_dim: int = 32, backbone_emb_dim: int = 32,
                 hbond_emb_dim: int = 32, flexibility_emb_dim: int = 32,
                 output_dim: int = 64):
        super().__init__()
        self.base_encoder = BaseTypeEncoder(base_emb_dim)
        self.backbone_encoder = BackboneEncoder(backbone_emb_dim)
        self.hbond_encoder = HBondEncoder(hbond_emb_dim)
        self.flexibility_encoder = FlexibilityEncoder(flexibility_emb_dim)
        
        # 最终融合层
        total_dim = base_emb_dim + backbone_emb_dim + hbond_emb_dim + flexibility_emb_dim
        self.final_fusion = nn.Sequential(
            nn.Linear(total_dim, output_dim * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(output_dim * 2, output_dim),
            nn.LayerNorm(output_dim)
        )
    
    def forward(self, base_features: Dict[str, torch.Tensor],
                backbone_features: Dict[str, torch.Tensor],
                hbond_features: Dict[str, torch.Tensor],
                flexibility_features: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        前向传播
        
        Args:
            base_features: 包含 base_types, secondary_struct, pairing_types, stacking, noncanonical
            backbone_features: 包含 torsion_angles, pucker_params, geometry_features
            hbond_features: 包含 hbond_counts, ligand_hbond, hbond_strength
            flexibility_features: 包含 bfactor, rmsf, ss_rigidity, conf_diversity
        
        Returns:
            [N, output_dim] 融合后的核酸特异性特征
        """
        base_emb = self.base_encoder(
            base_features['base_types'],
            base_features['secondary_struct'],
            base_features['pairing_types'],
            base_features['stacking'],
            base_features['noncanonical']
        )
        
        backbone_emb = self.backbone_encoder(
            backbone_features['torsion_angles'],
            backbone_features['pucker_params'],
            backbone_features['geometry_features']
        )
        
        hbond_emb = self.hbond_encoder(
            hbond_features['hbond_counts'],
            hbond_features['ligand_hbond'],
            hbond_features['hbond_strength']
        )
        
        flexibility_emb = self.flexibility_encoder(
            flexibility_features['bfactor'],
            flexibility_features['rmsf'],
            flexibility_features['ss_rigidity'],
            flexibility_features['conf_diversity']
        )
        
        # 融合所有特征
        all_features = torch.cat([base_emb, backbone_emb, hbond_emb, flexibility_emb], dim=-1)
        return self.final_fusion(all_features)


def compute_nucleic_torsion_angles(residue_atoms: Dict[str, np.ndarray]) -> np.ndarray:
    """
    计算核酸骨架的二面角
    返回: [α, β, γ, δ, ε, ζ, χ]
    """
    def dihedral(p1, p2, p3, p4):
        """计算四个点的二面角"""
        b1 = p2 - p1
        b2 = p3 - p2
        b3 = p4 - p3
        
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        
        n1_norm = n1 / (np.linalg.norm(n1) + 1e-8)
        n2_norm = n2 / (np.linalg.norm(n2) + 1e-8)
        
        cos_angle = np.dot(n1_norm, n2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)
        angle = np.arccos(cos_angle)
        
        # 判断符号
        sign = np.sign(np.dot(np.cross(n1_norm, n2_norm), b2 / (np.linalg.norm(b2) + 1e-8)))
        return angle * sign
    
    torsion_angles = np.zeros(7)
    
    try:
        # α: O3'(i-1) - P - O5' - C5'
        if all(k in residue_atoms for k in ['O3p_prev', 'P', 'O5p', 'C5p']):
            torsion_angles[0] = dihedral(
                residue_atoms['O3p_prev'], residue_atoms['P'],
                residue_atoms['O5p'], residue_atoms['C5p']
            )
        
        # β: P - O5' - C5' - C4'
        if all(k in residue_atoms for k in ['P', 'O5p', 'C5p', 'C4p']):
            torsion_angles[1] = dihedral(
                residue_atoms['P'], residue_atoms['O5p'],
                residue_atoms['C5p'], residue_atoms['C4p']
            )
        
        # γ: O5' - C5' - C4' - C3'
        if all(k in residue_atoms for k in ['O5p', 'C5p', 'C4p', 'C3p']):
            torsion_angles[2] = dihedral(
                residue_atoms['O5p'], residue_atoms['C5p'],
                residue_atoms['C4p'], residue_atoms['C3p']
            )
        
        # δ: C5' - C4' - C3' - O3'
        if all(k in residue_atoms for k in ['C5p', 'C4p', 'C3p', 'O3p']):
            torsion_angles[3] = dihedral(
                residue_atoms['C5p'], residue_atoms['C4p'],
                residue_atoms['C3p'], residue_atoms['O3p']
            )
        
        # ε: C4' - C3' - O3' - P(i+1)
        if all(k in residue_atoms for k in ['C4p', 'C3p', 'O3p', 'P_next']):
            torsion_angles[4] = dihedral(
                residue_atoms['C4p'], residue_atoms['C3p'],
                residue_atoms['O3p'], residue_atoms['P_next']
            )
        
        # ζ: C3' - O3' - P(i+1) - O5'(i+1)
        if all(k in residue_atoms for k in ['C3p', 'O3p', 'P_next', 'O5p_next']):
            torsion_angles[5] = dihedral(
                residue_atoms['C3p'], residue_atoms['O3p'],
                residue_atoms['P_next'], residue_atoms['O5p_next']
            )
        
        # χ: O4' - C1' - N9/C1 (嘌呤/嘧啶)
        if all(k in residue_atoms for k in ['O4p', 'C1p', 'N1_or_N9']):
            torsion_angles[6] = dihedral(
                residue_atoms['O4p'], residue_atoms['C1p'],
                residue_atoms['N1_or_N9'], residue_atoms.get('C2_or_C4', residue_atoms['C1p'])
            )
    except:
        pass
    
    return torsion_angles


def compute_sugar_pucker(C1p, C2p, C3p, C4p, O4p):
    """
    计算糖环的伪扭转角 (pseudorotation phase) 和振幅
    返回: [phase, amplitude]
    """
    try:
        # 简化的 pucker 计算
        # 使用 C1'-C2'-C3'-C4' 的扭转角作为近似
        v1 = C2p - C1p
        v2 = C3p - C2p
        v3 = C4p - C3p
        
        n1 = np.cross(v1, v2)
        n2 = np.cross(v2, v3)
        
        n1_norm = n1 / (np.linalg.norm(n1) + 1e-8)
        n2_norm = n2 / (np.linalg.norm(n2) + 1e-8)
        
        cos_phase = np.dot(n1_norm, n2_norm)
        cos_phase = np.clip(cos_phase, -1.0, 1.0)
        phase = np.arccos(cos_phase)
        
        # 振幅近似为 C2'-O4' 距离
        amplitude = np.linalg.norm(C2p - O4p) * 0.1
        
        return np.array([phase, amplitude])
    except:
        return np.array([0.0, 0.0])



