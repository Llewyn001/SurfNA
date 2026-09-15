"""
MDN模型改进建议代码片段
这些是建议的修改，可以逐步集成到原始模型中
"""

import torch
import torch.nn.functional as F
from torch_geometric.utils import to_dense_batch

# ==================== 改进1: 更好的特征提取 ====================

def extract_improved_features(lig_node_attr, rec_node_attr, ns, num_conv_layers):
    """
    改进的特征提取，不仅使用标量特征，也考虑向量特征
    """
    if num_conv_layers >= 3:
        # 使用标量特征
        scalar_lig = torch.cat([lig_node_attr[:, :ns], lig_node_attr[:, -ns:]], dim=1)
        scalar_rec = torch.cat([rec_node_attr[:, :ns], rec_node_attr[:, -ns:]], dim=1)
        
        # 提取向量特征的模长（如果存在）
        if lig_node_attr.shape[1] > 2 * ns:
            # 假设向量特征在中间位置
            vector_start = ns
            vector_end = lig_node_attr.shape[1] - ns
            if vector_end > vector_start:
                vector_lig = lig_node_attr[:, vector_start:vector_end]
                # 计算向量特征的统计量（均值、最大值等）
                vector_lig_norm = torch.norm(vector_lig.view(vector_lig.shape[0], -1, 3), dim=-1)
                vector_lig_mean = vector_lig_norm.mean(dim=-1)
                vector_lig_max = vector_lig_norm.max(dim=-1)[0]
                scalar_lig = torch.cat([scalar_lig, vector_lig_mean.unsqueeze(-1), 
                                       vector_lig_max.unsqueeze(-1)], dim=1)
        
        if rec_node_attr.shape[1] > 2 * ns:
            vector_start = ns
            vector_end = rec_node_attr.shape[1] - ns
            if vector_end > vector_start:
                vector_rec = rec_node_attr[:, vector_start:vector_end]
                vector_rec_norm = torch.norm(vector_rec.view(vector_rec.shape[0], -1, 3), dim=-1)
                vector_rec_mean = vector_rec_norm.mean(dim=-1)
                vector_rec_max = vector_rec_norm.max(dim=-1)[0]
                scalar_rec = torch.cat([scalar_rec, vector_rec_mean.unsqueeze(-1), 
                                       vector_rec_max.unsqueeze(-1)], dim=1)
    else:
        scalar_lig = lig_node_attr[:, :ns]
        scalar_rec = rec_node_attr[:, :ns]
    
    return scalar_lig, scalar_rec


# ==================== 改进2: 使用注意力机制融合特征 ====================

class AttentionFusion(torch.nn.Module):
    """使用注意力机制融合配体和受体特征"""
    
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 1),
            torch.nn.Sigmoid()
        )
    
    def forward(self, h_l_x, h_t_x):
        """
        Args:
            h_l_x: [B, N_l, C]
            h_t_x: [B, N_t, C]
        Returns:
            fused_features: [B, N_l, N_t, C*2]
        """
        B, N_l, C = h_l_x.shape
        N_t = h_t_x.shape[1]
        
        # 扩展维度
        h_l_expanded = h_l_x.unsqueeze(2).expand(B, N_l, N_t, C)  # [B, N_l, N_t, C]
        h_t_expanded = h_t_x.unsqueeze(1).expand(B, N_l, N_t, C)  # [B, N_l, N_t, C]
        
        # 拼接
        C_concat = torch.cat([h_l_expanded, h_t_expanded], dim=-1)  # [B, N_l, N_t, 2*C]
        
        # 计算注意力权重
        attention_weights = self.attention(C_concat)  # [B, N_l, N_t, 1]
        
        # 应用注意力（可选：这里我们保留所有特征，只是加权）
        # 或者可以只保留高注意力的特征对
        return C_concat, attention_weights


# ==================== 改进3: 改进的MDN输出层 ====================

class ImprovedMDNHead(torch.nn.Module):
    """改进的MDN输出头"""
    
    def __init__(self, hidden_dim, n_gaussians=20, mdn_dropout=0.0):
        super().__init__()
        self.MLP = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.BatchNorm1d(hidden_dim),
            torch.nn.ELU(),
            torch.nn.Dropout(p=mdn_dropout),
            torch.nn.Linear(hidden_dim, hidden_dim // 2),
            torch.nn.BatchNorm1d(hidden_dim // 2),
            torch.nn.ELU(),
            torch.nn.Dropout(p=mdn_dropout)
        )
        
        # 分别预测pi, sigma, mu
        self.z_pi = torch.nn.Linear(hidden_dim // 2, n_gaussians)
        self.z_sigma = torch.nn.Linear(hidden_dim // 2, n_gaussians)
        self.z_mu = torch.nn.Linear(hidden_dim // 2, n_gaussians)
        
        # 初始化
        self._initialize_weights()
    
    def _initialize_weights(self):
        """改进的权重初始化"""
        # pi: 初始化为均匀分布
        torch.nn.init.uniform_(self.z_pi.weight, -0.01, 0.01)
        torch.nn.init.constant_(self.z_pi.bias, 0.0)
        
        # sigma: 初始化为较小的正值
        torch.nn.init.uniform_(self.z_sigma.weight, -0.01, 0.01)
        torch.nn.init.constant_(self.z_sigma.bias, 0.5)  # 初始sigma约0.5
        
        # mu: 初始化为中等距离（约3-5Å）
        torch.nn.init.uniform_(self.z_mu.weight, -0.01, 0.01)
        torch.nn.init.constant_(self.z_mu.bias, 1.0)  # 初始mu约3.0（经过ELU+1后）
    
    def forward(self, C):
        """
        Args:
            C: [N, hidden_dim] 特征
        Returns:
            pi: [N, n_gaussians] 混合权重
            sigma: [N, n_gaussians] 标准差
            mu: [N, n_gaussians] 均值
        """
        C = self.MLP(C)
        
        # pi: 使用softmax确保归一化
        pi = F.softmax(self.z_pi(C), dim=-1)
        
        # sigma: 使用softplus确保为正，初始值较小
        sigma = F.softplus(self.z_sigma(C)) + 0.1
        
        # mu: 使用ELU+偏移，但偏移更小
        mu = F.elu(self.z_mu(C)) + 0.5
        
        return pi, sigma, mu


# ==================== 改进4: 改进的forward函数片段 ====================

def improved_forward_mdn_section(self, scalar_lig_attr, scalar_rec_attr, 
                                   h_l_pos, h_t_pos, data, C_batch, B, N_l, N_t):
    """
    改进的MDN forward函数片段
    这个函数应该替换原始模型中的MDN部分（大约335-422行）
    """
    from utils.training_mdn_improved import (
        calculate_probability_improved, 
        improved_scoring
    )
    
    # 转换为dense batch
    h_l_x, l_mask = to_dense_batch(scalar_lig_attr, data['ligand'].batch, fill_value=0)
    h_t_x, t_mask = to_dense_batch(scalar_rec_attr, data['surface'].batch, fill_value=0)
    
    assert h_l_x.size(0) == h_t_x.size(0), 'Encountered unequal batch-sizes'
    
    # 扩展维度
    h_l_x = h_l_x.unsqueeze(-2).repeat(1, 1, N_t, 1)  # [B, N_l, N_t, C_out]
    h_t_x = h_t_x.unsqueeze(-3).repeat(1, N_l, 1, 1)  # [B, N_l, N_t, C_out]
    
    # 拼接特征
    C = torch.cat((h_l_x, h_t_x), -1)
    C_mask = l_mask.view(B, N_l, 1) & t_mask.view(B, 1, N_t)
    C = C[C_mask]
    
    # 通过MLP
    C = self.MLP(C)
    
    # 获取批次索引
    C_batch = torch.arange(B, device=self.device).unsqueeze(-1).unsqueeze(-1)
    C_batch = C_batch.repeat(1, N_l, N_t)[C_mask]
    
    # MDN输出
    pi = F.softmax(self.z_pi(C), -1)
    sigma = F.softplus(self.z_sigma(C)) + 0.1  # 改进：使用softplus
    mu = F.elu(self.z_mu(C)) + 0.5  # 改进：偏移更小
    
    # 计算距离
    dist = compute_euclidean_distances_matrix(h_l_pos, h_t_pos)[C_mask]
    
    if self.training:
        # 使用改进的损失函数
        from utils.training_mdn_improved import mdn_loss_fn_improved
        mdn_loss_interaction = mdn_loss_fn_improved(
            pi, sigma, mu, dist.unsqueeze(1).detach(),
            dist_threshold=self.args.mdn_dist_threshold_train if self.args.mdn_dist_threshold_train is not None else 7.0,
            weight_near=1.0, weight_far=0.1, use_distance_weighting=True
        )
        return mdn_loss_interaction
    else:
        # 使用改进的概率计算和评分
        prob = calculate_probability_improved(
            pi, sigma, mu, dist.unsqueeze(1).detach(),
            dist_threshold=self.args.mdn_dist_threshold_test if self.args.mdn_dist_threshold_test is not None else 5.0,
            use_soft_threshold=True, temperature=1.0
        )
        
        probx = improved_scoring(
            prob, C_batch, dist,
            use_distance_weighting=True, normalize=False
        )
        
        return probx


# ==================== 使用示例 ====================

"""
在原始模型中集成这些改进的步骤：

1. 在__init__中，将MDN头替换为ImprovedMDNHead:
   ```python
   if self.mdn_mode:
       self.mdn_head = ImprovedMDNHead(mdn_hidden_dim, n_gaussians, mdn_dropout)
   ```

2. 在forward中，使用改进的特征提取:
   ```python
   scalar_lig_attr, scalar_rec_attr = extract_improved_features(
       lig_node_attr, surface_node_attr, self.ns, self.num_conv_layers
   )
   ```

3. 在forward中，使用改进的MDN部分:
   ```python
   result = improved_forward_mdn_section(
       self, scalar_lig_attr, scalar_rec_attr,
       h_l_pos, h_t_pos, data, C_batch, B, N_l, N_t
   )
   ```

4. 在训练脚本中，使用改进的损失函数:
   ```python
   from utils.training_mdn_improved import mdn_loss_fn_improved
   loss = mdn_loss_fn_improved(pi, sigma, mu, dist, ...)
   ```
"""
