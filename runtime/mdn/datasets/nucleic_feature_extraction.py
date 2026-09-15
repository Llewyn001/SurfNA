"""
核酸特征提取工具
从核酸结构中提取各种特征用于编码器
"""
import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from scipy.spatial.distance import cdist
from Bio.PDB import PDBParser
import warnings


def extract_base_type_features(residue, base_type_dict: Dict[str, int]) -> int:
    """
    提取碱基类型索引
    """
    resname = residue.resname.strip()
    return base_type_dict.get(resname, base_type_dict.get('UNK', 0))


def extract_secondary_structure(residue, ss_dict: Dict[str, int], 
                                default_ss: str = 'loop') -> int:
    """
    提取二级结构类型
    如果结构中没有二级结构信息，可以根据几何特征推断
    """
    # 这里可以从 DSSR 或其他工具获取，暂时使用默认值
    # 实际应用中应该从外部工具（如 DSSR, RNAView）获取
    return ss_dict.get(default_ss, 0)


def extract_pairing_type(residue, pairing_dict: Dict[str, int],
                         base_pairs: Optional[List[Tuple[int, int, str]]] = None) -> int:
    """
    提取配对类型
    base_pairs: [(res_idx1, res_idx2, pair_type), ...]
    """
    # 如果没有配对信息，返回 unpaired
    if base_pairs is None:
        return pairing_dict.get('unpaired', 0)
    
    # 检查当前残基是否在配对中
    res_idx = residue.id[1] if hasattr(residue, 'id') else -1
    for idx1, idx2, ptype in base_pairs:
        if idx1 == res_idx or idx2 == res_idx:
            return pairing_dict.get(ptype, pairing_dict.get('Watson-Crick', 1))
    
    return pairing_dict.get('unpaired', 0)


def extract_stacking_info(residue, residues_list: List, 
                          stacking_dict: Dict[str, int]) -> int:
    """
    检测 stacking 关系
    通过碱基平面的距离和角度判断
    """
    try:
        # 获取碱基平面原子
        base_atoms = []
        for atom in residue:
            if atom.name in ['N1', 'N3', 'N9', 'C2', 'C4', 'C5', 'C6', 'C8']:
                base_atoms.append(atom.coord)
        
        if len(base_atoms) < 3:
            return stacking_dict.get('no', 0)
        
        # 计算碱基平面中心
        base_center = np.mean(base_atoms, axis=0)
        
        # 检查与相邻残基的 stacking
        res_idx = residue.id[1] if hasattr(residue, 'id') else -1
        for other_res in residues_list:
            if other_res.id[1] == res_idx:
                continue
            
            other_base_atoms = []
            for atom in other_res:
                if atom.name in ['N1', 'N3', 'N9', 'C2', 'C4', 'C5', 'C6', 'C8']:
                    other_base_atoms.append(atom.coord)
            
            if len(other_base_atoms) < 3:
                continue
            
            other_center = np.mean(other_base_atoms, axis=0)
            dist = np.linalg.norm(base_center - other_center)
            
            # Stacking 通常距离在 3-4 Å
            if 3.0 < dist < 4.5:
                return stacking_dict.get('stacking', 1)
        
        return stacking_dict.get('no', 0)
    except:
        return stacking_dict.get('no', 0)


def extract_hbond_features(residue, residues_list: List, 
                          ligand_coords: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    提取氢键特征
    """
    hbond_counts = np.zeros(3)  # Watson-Crick, Hoogsteen, Sugar-edge
    ligand_hbond = np.zeros(2)  # donor, acceptor
    hbond_strength = np.zeros(3)  # 平均距离、最小距离、总数量
    
    try:
        # 简化的氢键检测：基于距离和角度
        # 实际应用中应该使用更精确的方法（如 HBPLUS, DSSR）
        
        # 检测与配体的氢键
        if ligand_coords is not None:
            residue_coords = np.array([atom.coord for atom in residue])
            distances = cdist(residue_coords, ligand_coords)
            min_dist = np.min(distances)
            
            if min_dist < 3.5:  # 氢键距离阈值
                # 简单判断：如果有 N/O 原子在配体附近，可能是氢键
                for atom in residue:
                    if atom.name in ['N1', 'N3', 'N7', 'O2', 'O4', 'O6']:
                        atom_coord = atom.coord.reshape(1, -1)
                        dist_to_ligand = np.min(cdist(atom_coord, ligand_coords))
                        if dist_to_ligand < 3.5:
                            if atom.name.startswith('N'):
                                ligand_hbond[0] = 1  # donor
                            else:
                                ligand_hbond[1] = 1  # acceptor
        
        # 检测残基间的氢键（简化版）
        hbond_distances = []
        for other_res in residues_list:
            if other_res.id[1] == residue.id[1]:
                continue
            
            for atom1 in residue:
                if atom1.name in ['N1', 'N3', 'N7', 'O2', 'O4', 'O6']:
                    for atom2 in other_res:
                        if atom2.name in ['N1', 'N3', 'N7', 'O2', 'O4', 'O6']:
                            dist = np.linalg.norm(atom1.coord - atom2.coord)
                            if 2.5 < dist < 3.5:
                                hbond_distances.append(dist)
        
        if len(hbond_distances) > 0:
            hbond_strength[0] = np.mean(hbond_distances)
            hbond_strength[1] = np.min(hbond_distances)
            hbond_strength[2] = len(hbond_distances)
    
    except:
        pass
    
    return {
        'hbond_counts': hbond_counts,
        'ligand_hbond': ligand_hbond,
        'hbond_strength': hbond_strength
    }


def extract_flexibility_features(residue, bfactor_dict: Optional[Dict[int, float]] = None,
                                 rmsf_dict: Optional[Dict[int, float]] = None) -> Dict[str, np.ndarray]:
    """
    提取柔性特征
    """
    bfactor = 50.0  # 默认值
    rmsf = 1.0  # 默认值
    ss_rigidity = 1  # 默认：loop (柔性)
    conf_diversity = 0.5  # 默认值
    
    try:
        # 从 B-factor 获取
        if bfactor_dict is not None:
            res_idx = residue.id[1] if hasattr(residue, 'id') else -1
            bfactor = bfactor_dict.get(res_idx, 50.0)
        
        # 从 RMSF 获取（如果有 MD 轨迹）
        if rmsf_dict is not None:
            res_idx = residue.id[1] if hasattr(residue, 'id') else -1
            rmsf = rmsf_dict.get(res_idx, 1.0)
        
        # 简化的二级结构刚性判断
        # 实际应该从 DSSR 等工具获取
        # stem 通常更刚性 (0), loop 更柔性 (1)
        ss_rigidity = 1  # 默认柔性
    
    except:
        pass
    
    return {
        'bfactor': np.array([bfactor * 0.01]),  # 归一化
        'rmsf': np.array([rmsf * 0.1]),  # 归一化
        'ss_rigidity': ss_rigidity,
        'conf_diversity': np.array([conf_diversity])
    }


def extract_backbone_torsion_angles(residue, prev_residue: Optional = None,
                                    next_residue: Optional = None) -> np.ndarray:
    """
    提取骨架二面角
    返回: [α, β, γ, δ, ε, ζ, χ]
    """
    torsion_angles = np.zeros(7)
    
    try:
        # 获取关键原子坐标
        atoms = {}
        for atom in residue:
            atoms[atom.name] = atom.coord
        
        # 获取前一个残基的 O3'
        if prev_residue is not None:
            for atom in prev_residue:
                if atom.name == "O3'":
                    atoms['O3p_prev'] = atom.coord
                    break
        
        # 获取下一个残基的 P 和 O5'
        if next_residue is not None:
            for atom in next_residue:
                if atom.name == 'P':
                    atoms['P_next'] = atom.coord
                elif atom.name == "O5'":
                    atoms['O5p_next'] = atom.coord
        
        # 计算二面角
        def dihedral(p1, p2, p3, p4):
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
            
            sign = np.sign(np.dot(np.cross(n1_norm, n2_norm), b2 / (np.linalg.norm(b2) + 1e-8)))
            return angle * sign
        
        # α: O3'(i-1) - P - O5' - C5'
        if all(k in atoms for k in ['O3p_prev', 'P', "O5'", "C5'"]):
            torsion_angles[0] = dihedral(atoms['O3p_prev'], atoms['P'],
                                        atoms["O5'"], atoms["C5'"])
        
        # β: P - O5' - C5' - C4'
        if all(k in atoms for k in ['P', "O5'", "C5'", "C4'"]):
            torsion_angles[1] = dihedral(atoms['P'], atoms["O5'"],
                                        atoms["C5'"], atoms["C4'"])
        
        # γ: O5' - C5' - C4' - C3'
        if all(k in atoms for k in ["O5'", "C5'", "C4'", "C3'"]):
            torsion_angles[2] = dihedral(atoms["O5'"], atoms["C5'"],
                                        atoms["C4'"], atoms["C3'"])
        
        # δ: C5' - C4' - C3' - O3'
        if all(k in atoms for k in ["C5'", "C4'", "C3'", "O3'"]):
            torsion_angles[3] = dihedral(atoms["C5'"], atoms["C4'"],
                                        atoms["C3'"], atoms["O3'"])
        
        # ε: C4' - C3' - O3' - P(i+1)
        if all(k in atoms for k in ["C4'", "C3'", "O3'", 'P_next']):
            torsion_angles[4] = dihedral(atoms["C4'"], atoms["C3'"],
                                        atoms["O3'"], atoms['P_next'])
        
        # ζ: C3' - O3' - P(i+1) - O5'(i+1)
        if all(k in atoms for k in ["C3'", "O3'", 'P_next', 'O5p_next']):
            torsion_angles[5] = dihedral(atoms["C3'"], atoms["O3'"],
                                        atoms['P_next'], atoms['O5p_next'])
        
        # χ: O4' - C1' - N9/C1 - C2/C4
        if "O4'" in atoms and "C1'" in atoms:
            # 根据碱基类型选择
            if 'N9' in atoms:  # 嘌呤
                if 'C4' in atoms:
                    torsion_angles[6] = dihedral(atoms["O4'"], atoms["C1'"],
                                                 atoms['N9'], atoms['C4'])
            elif 'N1' in atoms:  # 嘧啶
                if 'C2' in atoms:
                    torsion_angles[6] = dihedral(atoms["O4'"], atoms["C1'"],
                                                 atoms['N1'], atoms['C2'])
    
    except Exception as e:
        pass
    
    return torsion_angles


def extract_sugar_pucker(residue) -> np.ndarray:
    """
    提取糖环 pucker 参数
    返回: [pseudorotation phase, amplitude]
    """
    try:
        atoms = {}
        for atom in residue:
            atoms[atom.name] = atom.coord
        
        if all(k in atoms for k in ["C1'", "C2'", "C3'", "C4'", "O4'"]):
            # 简化的 pucker 计算
            v1 = atoms["C2'"] - atoms["C1'"]
            v2 = atoms["C3'"] - atoms["C2'"]
            v3 = atoms["C4'"] - atoms["C3'"]
            
            n1 = np.cross(v1, v2)
            n2 = np.cross(v2, v3)
            
            n1_norm = n1 / (np.linalg.norm(n1) + 1e-8)
            n2_norm = n2 / (np.linalg.norm(n2) + 1e-8)
            
            cos_phase = np.dot(n1_norm, n2_norm)
            cos_phase = np.clip(cos_phase, -1.0, 1.0)
            phase = np.arccos(cos_phase)
            
            amplitude = np.linalg.norm(atoms["C2'"] - atoms["O4'"]) * 0.1
            
            return np.array([phase, amplitude])
    except:
        pass
    
    return np.array([0.0, 0.0])


def extract_geometry_features(residue) -> np.ndarray:
    """
    提取骨架几何特征（关键距离）
    返回: [5个关键距离特征]
    """
    geometry_features = np.zeros(5)
    
    try:
        atoms = {}
        for atom in residue:
            atoms[atom.name] = atom.coord
        
        # C4'-C3'-O3'-P 等关键距离
        if all(k in atoms for k in ["C4'", "C3'"]):
            geometry_features[0] = np.linalg.norm(atoms["C4'"] - atoms["C3'"]) * 0.1
        
        if all(k in atoms for k in ["C3'", "O3'"]):
            geometry_features[1] = np.linalg.norm(atoms["C3'"] - atoms["O3'"]) * 0.1
        
        if all(k in atoms for k in ["O3'", "P"]):
            geometry_features[2] = np.linalg.norm(atoms["O3'"] - atoms['P']) * 0.1
        
        if all(k in atoms for k in ["P", "O5'"]):
            geometry_features[3] = np.linalg.norm(atoms['P'] - atoms["O5'"]) * 0.1
        
        if all(k in atoms for k in ["O5'", "C5'"]):
            geometry_features[4] = np.linalg.norm(atoms["O5'"] - atoms["C5'"]) * 0.1
    
    except:
        pass
    
    return geometry_features


# 定义各种字典
BASE_TYPE_DICT = {
    'A': 0, 'C': 1, 'G': 2, 'U': 3, 'T': 4,
    'DA': 0, 'DC': 1, 'DG': 2, 'DT': 4,
    'AMP': 0, 'CMP': 1, 'GMP': 2, 'UMP': 3,
    'UNK': 19
}

SECONDARY_STRUCTURE_DICT = {
    'stem': 0, 'loop': 1, 'bulge': 2, 'internal_loop': 3,
    'hairpin': 4, 'junction': 5, 'pseudoknot': 6, 'other': 7
}

PAIRING_TYPE_DICT = {
    'unpaired': 0, 'Watson-Crick': 1, 'Hoogsteen': 2,
    'Sugar-edge': 3, 'Wobble': 4, 'other': 5
}

STACKING_DICT = {
    'no': 0, 'stacking': 1, 'triple': 2
}

SS_RIGIDITY_DICT = {
    'stem': 0, 'loop': 1, 'bulge': 2, 'internal_loop': 3, 'other': 4
}



