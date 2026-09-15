"""
改进的MDN训练函数
主要改进：
1. 改进损失函数，对所有距离计算损失但给予不同权重
2. 改进概率计算，不硬截断
3. 添加距离加权
"""
import torch
from torch.distributions import Normal
import torch.nn.functional as F

def mdn_loss_fn_improved(pi, sigma, mu, y, dist_threshold=7.0, eps=1e-10, 
                         weight_near=1.0, weight_far=0.1, use_distance_weighting=True):
    """
    改进的MDN损失函数
    
    Args:
        pi: 混合权重 [N, n_gaussians]
        sigma: 标准差 [N, n_gaussians]
        mu: 均值 [N, n_gaussians]
        y: 真实距离 [N, 1]
        dist_threshold: 距离阈值，用于区分近远距离
        weight_near: 近距离样本的权重
        weight_far: 远距离样本的权重
        use_distance_weighting: 是否使用距离加权
    """
    # 数值稳定性处理
    mu = torch.clamp(torch.nan_to_num(mu, 0.0), min=1e-6, max=50.0)
    sigma = torch.clamp(torch.nan_to_num(sigma, 0.0), min=1e-6, max=10.0)
    pi = torch.clamp(torch.nan_to_num(pi, 0.0), min=1e-6, max=1.0)
    
    # 确保pi归一化
    pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)
    
    # 计算对数似然
    normal = Normal(mu.real, sigma.real)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    
    # 计算混合模型的负对数似然
    loss = -torch.logsumexp(torch.log(pi.real + eps) + loglik, dim=1)
    
    # 根据距离给予不同权重
    if use_distance_weighting:
        # 近距离样本权重高，远距离样本权重低
        weights = torch.where(
            y.squeeze() <= dist_threshold,
            torch.ones_like(y.squeeze()) * weight_near,
            torch.ones_like(y.squeeze()) * weight_far
        )
        # 也可以使用连续权重函数
        # weights = weight_near * torch.exp(-(y.squeeze() - dist_threshold) / 2.0)
        # weights = torch.clamp(weights, min=weight_far, max=weight_near)
        loss = loss * weights
    
    # 只对有效样本计算平均（避免NaN）
    valid_mask = ~torch.isnan(loss) & ~torch.isinf(loss)
    if valid_mask.sum() > 0:
        loss = loss[valid_mask].mean()
    else:
        loss = torch.tensor(0.0, device=loss.device, requires_grad=True)
    
    return torch.nan_to_num(loss, 0.0)


def calculate_probability_improved(pi, sigma, mu, y, dist_threshold=5.0, eps=1e-10,
                                   use_soft_threshold=True, temperature=1.0):
    """
    改进的概率计算函数
    
    Args:
        pi: 混合权重 [N, n_gaussians]
        sigma: 标准差 [N, n_gaussians]
        mu: 均值 [N, n_gaussians]
        y: 距离 [N, 1]
        dist_threshold: 距离阈值
        use_soft_threshold: 是否使用软阈值（而不是硬截断）
        temperature: 温度参数，用于调整概率的锐度
    """
    # 数值稳定性处理
    mu = torch.clamp(torch.nan_to_num(mu, 0.0), min=1e-6, max=50.0)
    sigma = torch.clamp(torch.nan_to_num(sigma, 0.0), min=1e-6, max=10.0)
    pi = torch.clamp(torch.nan_to_num(pi, 0.0), min=1e-6, max=1.0)
    pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)
    
    # 计算概率
    normal = Normal(mu.real, sigma.real)
    logprob = normal.log_prob(y.expand_as(normal.loc))
    logprob += torch.log(pi.real + eps)
    prob = torch.logsumexp(logprob, dim=1)  # 使用logsumexp更稳定
    prob = prob.exp() / temperature  # 应用温度参数
    
    # 软阈值或硬阈值
    if use_soft_threshold:
        # 使用sigmoid函数实现软阈值
        # 距离越远，权重越小，但不完全为0
        distance_weight = torch.sigmoid((dist_threshold - y.squeeze()) / 1.0)
        prob = prob * distance_weight
    else:
        # 硬截断（原始方法）
        prob = torch.where(y.squeeze() > dist_threshold, 
                          torch.zeros_like(prob), prob)
    
    return prob


def improved_scoring(prob, C_batch, dist, contact_weights=None, 
                     use_distance_weighting=True, normalize=False):
    """
    改进的评分聚合函数
    
    Args:
        prob: 每个原子对的概率 [N]
        C_batch: 批次索引 [N]
        dist: 距离 [N]
        contact_weights: 可选的接触权重 [N]
        use_distance_weighting: 是否使用距离加权
        normalize: 是否归一化分数
    """
    if use_distance_weighting:
        if contact_weights is None:
            # 距离越近，权重越大
            # 使用指数衰减：exp(-dist/2.0)
            contact_weights = torch.exp(-dist / 2.0)
            # 或者使用更平滑的函数
            # contact_weights = 1.0 / (1.0 + (dist / 3.0) ** 2)
        
        weighted_prob = prob * contact_weights
    else:
        weighted_prob = prob
    
    # 聚合到批次级别
    probx = torch.scatter_add(torch.zeros((C_batch.max().item() + 1,), 
                                          device=prob.device, dtype=prob.dtype),
                              dim=0, index=C_batch, src=weighted_prob)
    
    if normalize:
        # 归一化：除以接触数量
        contact_count = torch.scatter_add(
            torch.zeros((C_batch.max().item() + 1,), 
                       device=prob.device, dtype=torch.long),
            dim=0, index=C_batch, 
            src=torch.ones_like(C_batch, dtype=torch.long))
        probx = probx / (contact_count.float() + 1e-10)
    
    return probx


def mdn_loss_fn_min_distance_atom_improved(pi, sigma, mu, y, dist_threshold=7.0, 
                                            eps=1e-10, topN=1):
    """
    改进的最小距离原子损失函数（用于CA位置预测）
    """
    mu = torch.clamp(torch.nan_to_num(mu, 0.0), min=1e-6, max=50.0)
    sigma = torch.clamp(torch.nan_to_num(sigma, 0.0), min=1e-6, max=10.0)
    pi = torch.clamp(torch.nan_to_num(pi, 0.0), min=1e-6, max=1.0)
    pi = pi / (pi.sum(dim=-1, keepdim=True) + eps)
    
    normal = Normal(mu.real, sigma.real)
    loglik = normal.log_prob(y.expand_as(normal.loc))
    loss = -torch.logsumexp(torch.log(pi.real + eps) + loglik, dim=1)
    
    # 只对近距离计算损失
    loss = loss[torch.where(y.squeeze() <= dist_threshold)[0]]
    
    if len(loss) > 0:
        loss = loss.mean()
    else:
        loss = torch.tensor(0.0, device=pi.device, requires_grad=True)
    
    return torch.nan_to_num(loss, 0.0)
