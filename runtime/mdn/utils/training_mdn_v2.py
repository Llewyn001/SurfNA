"""Grouped losses and metrics for the SurfNA V2 pose scorer."""

from __future__ import annotations

import math
from collections import Counter
from typing import NamedTuple

import torch
from torch.nn import functional as F

from models.surfna_v2_scorer_components import mixture_log_probability


class ScorerLossOutput(NamedTuple):
    loss: torch.Tensor
    prior_nll: torch.Tensor
    listwise_loss: torch.Tensor
    pairwise_loss: torch.Tensor
    ordinal_loss: torch.Tensor


def _unique_groups(group: torch.Tensor):
    for group_id in torch.unique(group):
        mask = group == group_id
        if torch.any(mask):
            yield group_id, mask


def grouped_listwise_rmsd_loss(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    rmsd_temperature: float = 1.0,
    score_temperature: float = 1.0,
) -> torch.Tensor:
    """Cross entropy between soft RMSD ordering and predicted group ordering."""
    losses = []
    for _, mask in _unique_groups(group):
        if int(mask.sum()) < 2:
            continue
        target = F.softmax(-rmsd[mask] / max(float(rmsd_temperature), 1e-6), dim=0)
        prediction = F.log_softmax(scores[mask] / max(float(score_temperature), 1e-6), dim=0)
        losses.append(-(target * prediction).sum())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def pose_quality_bin(rmsd: torch.Tensor) -> torch.Tensor:
    """Return deployment-relevant RMSD bins: <2, 2-5, 5-10, >=10 A."""
    bins = torch.zeros_like(rmsd, dtype=torch.long)
    bins = torch.where(rmsd >= 2.0, torch.ones_like(bins), bins)
    bins = torch.where(rmsd >= 5.0, torch.full_like(bins, 2), bins)
    bins = torch.where(rmsd >= 10.0, torch.full_like(bins, 3), bins)
    return bins


def grouped_threshold_listwise_loss(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    score_temperature: float = 1.0,
    target_temperature: float = 1.0,
) -> torch.Tensor:
    """Top-heavy listwise loss aligned to the <2 A deployment endpoint."""
    losses = []
    for _, mask in _unique_groups(group):
        if int(mask.sum()) < 2:
            continue
        distance = rmsd[mask]
        utility = torch.where(
            distance < 2.0,
            3.0 - 0.10 * distance,
            torch.where(
                distance < 5.0,
                1.0 - 0.05 * (distance - 2.0),
                torch.where(
                    distance < 10.0,
                    -0.5 - 0.02 * (distance - 5.0),
                    -1.5 - 0.005 * (distance - 10.0).clamp_max(20.0),
                ),
            ),
        )
        target = F.softmax(
            utility / max(float(target_temperature), 1e-6), dim=0
        )
        prediction = F.log_softmax(
            scores[mask] / max(float(score_temperature), 1e-6), dim=0
        )
        losses.append(-(target * prediction).sum())
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def grouped_pairwise_logistic_loss(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    min_rmsd_gap: float = 0.5,
    margin: float = 0.2,
    max_pair_weight: float = 5.0,
) -> torch.Tensor:
    """RMSD-gap-aware logistic preference loss within complete groups."""
    losses = []
    for _, mask in _unique_groups(group):
        score = scores[mask]
        distance = rmsd[mask]
        gap = distance[None, :] - distance[:, None]
        better = gap >= float(min_rmsd_gap)
        if not torch.any(better):
            continue
        score_delta = score[:, None] - score[None, :]
        weight = (gap / max(float(min_rmsd_gap), 1e-6)).clamp(1.0, float(max_pair_weight))
        pair_loss = F.softplus(float(margin) - score_delta)
        losses.append((pair_loss[better] * weight[better]).sum() / weight[better].sum().clamp_min(1.0))
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def grouped_boundary_pairwise_logistic_loss(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    min_rmsd_gap: float = 0.5,
    margin: float = 0.2,
    lt2_vs_2to5_weight: float = 4.0,
    lt2_vs_5to10_weight: float = 2.0,
    mid_boundary_weight: float = 1.5,
    trivial_far_weight: float = 0.5,
    within_bin_weight: float = 0.25,
) -> torch.Tensor:
    """Preference loss emphasizing pairs that cross the 2 A boundary."""
    losses = []
    for _, mask in _unique_groups(group):
        score = scores[mask]
        distance = rmsd[mask]
        gap = distance[None, :] - distance[:, None]
        better = gap >= float(min_rmsd_gap)
        if not torch.any(better):
            continue
        left = pose_quality_bin(distance)[:, None]
        right = pose_quality_bin(distance)[None, :]
        weight = torch.full_like(gap, float(within_bin_weight))
        weight = torch.where(
            (left == 0) & (right == 1),
            torch.full_like(weight, float(lt2_vs_2to5_weight)),
            weight,
        )
        weight = torch.where(
            (left == 0) & (right == 2),
            torch.full_like(weight, float(lt2_vs_5to10_weight)),
            weight,
        )
        weight = torch.where(
            ((left == 1) & (right == 2)) | ((left == 1) & (right == 3)),
            torch.full_like(weight, float(mid_boundary_weight)),
            weight,
        )
        weight = torch.where(
            (left == 0) & (right == 3),
            torch.full_like(weight, float(trivial_far_weight)),
            weight,
        )
        score_delta = score[:, None] - score[None, :]
        pair_loss = F.softplus(float(margin) - score_delta)
        losses.append(
            (pair_loss[better] * weight[better]).sum()
            / weight[better].sum().clamp_min(1.0)
        )
    if not losses:
        return scores.sum() * 0.0
    return torch.stack(losses).mean()


def cumulative_ordinal_pose_loss(
    logit_lt5: torch.Tensor,
    logit_lt2_given_lt5: torch.Tensor,
    rmsd: torch.Tensor,
    *,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Three-bin ordinal NLL with monotone ``P(<2)<=P(<5)`` probabilities.

    The first Bernoulli predicts whether a pose is below 5 A.  Only poses
    below 5 A enter the second Bernoulli, which predicts whether they are also
    below 2 A.  This is exactly the negative log-likelihood of the disjoint
    classes ``<2``, ``[2,5)`` and ``>=5`` under the cumulative factorization.
    """
    if logit_lt5.shape != rmsd.shape or logit_lt2_given_lt5.shape != rmsd.shape:
        raise ValueError("ordinal logits and RMSD must have identical shapes")
    target_lt5 = (rmsd < 5.0).to(logit_lt5.dtype)
    target_lt2 = (rmsd < 2.0).to(logit_lt5.dtype)
    if sample_weight is None:
        sample_weight = torch.ones_like(rmsd, dtype=logit_lt5.dtype)
    else:
        if sample_weight.shape != rmsd.shape:
            raise ValueError("ordinal sample weights and RMSD must have identical shapes")
        sample_weight = sample_weight.to(logit_lt5).clamp_min(0.0)
    loss_lt5_rows = F.binary_cross_entropy_with_logits(
        logit_lt5, target_lt5, reduction="none"
    )
    loss_lt5 = (loss_lt5_rows * sample_weight).sum() / sample_weight.sum().clamp_min(1e-8)
    within_five = target_lt5.bool()
    if torch.any(within_five):
        loss_lt2_rows = F.binary_cross_entropy_with_logits(
            logit_lt2_given_lt5[within_five],
            target_lt2[within_five],
            reduction="none",
        )
        weight_lt2 = sample_weight[within_five]
        loss_lt2 = (loss_lt2_rows * weight_lt2).sum() / weight_lt2.sum().clamp_min(1e-8)
    else:
        loss_lt2 = logit_lt2_given_lt5.sum() * 0.0
    return loss_lt5 + loss_lt2


def rmsd_bin_sample_weight(
    rmsd: torch.Tensor,
    weights: tuple[float, float, float, float] | None,
) -> torch.Tensor | None:
    if weights is None:
        return None
    if len(weights) != 4 or any(float(value) < 0 for value in weights):
        raise ValueError("pose-bin weights must contain four non-negative values")
    table = rmsd.new_tensor(tuple(float(value) for value in weights))
    return table[pose_quality_bin(rmsd)]


def near_native_prior_nll(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    candidate_distances: torch.Tensor,
    pair_mask: torch.Tensor,
    graph_is_prior_valid: torch.Tensor,
    *,
    contact_radius: float = 7.0,
) -> torch.Tensor:
    """MDN NLL restricted to native/near-native graphs and contact pairs.

    High-RMSD generated decoys must not teach the pose-independent prior that
    their incorrect distances are preferred interactions.
    """
    nll_sum, pair_count = near_native_prior_nll_sum_count(
        pi,
        mu,
        sigma,
        candidate_distances,
        pair_mask,
        graph_is_prior_valid,
        contact_radius=contact_radius,
    )
    return nll_sum / pair_count.clamp_min(1).to(nll_sum.dtype)


def near_native_prior_nll_sum_count(
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    candidate_distances: torch.Tensor,
    pair_mask: torch.Tensor,
    graph_is_prior_valid: torch.Tensor,
    *,
    contact_radius: float = 7.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return an additive NLL numerator and exact valid-pair count.

    This form allows a complete pose group to be recomputed in GPU
    microbatches without changing the prior objective's normalization.
    """
    if candidate_distances.ndim != 3:
        raise ValueError("candidate_distances must have shape [B,N_ligand,N_surface]")
    valid_graph = graph_is_prior_valid.bool().view(-1, 1, 1)
    valid_pair = pair_mask.bool() & valid_graph & (candidate_distances <= float(contact_radius))
    log_prob = mixture_log_probability(pi, sigma, mu, candidate_distances)
    pair_count = valid_pair.sum()
    if not torch.any(valid_pair):
        return log_prob.sum() * 0.0, pair_count
    return -log_prob[valid_pair].sum(), pair_count


def grouped_ranking_loss(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 0.25,
    rmsd_temperature: float = 1.0,
    score_temperature: float = 1.0,
    min_rmsd_gap: float = 0.5,
    pairwise_margin: float = 0.2,
    listwise_mode: str = "rmsd_softmax",
    pairwise_mode: str = "gap",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if listwise_mode == "rmsd_softmax":
        listwise = grouped_listwise_rmsd_loss(
            scores,
            rmsd,
            group,
            rmsd_temperature=rmsd_temperature,
            score_temperature=score_temperature,
        )
    elif listwise_mode == "threshold_topheavy":
        listwise = grouped_threshold_listwise_loss(
            scores,
            rmsd,
            group,
            score_temperature=score_temperature,
            target_temperature=rmsd_temperature,
        )
    else:
        raise ValueError(f"unknown listwise_mode={listwise_mode!r}")
    if pairwise_mode == "gap":
        pairwise = grouped_pairwise_logistic_loss(
            scores,
            rmsd,
            group,
            min_rmsd_gap=min_rmsd_gap,
            margin=pairwise_margin,
        )
    elif pairwise_mode == "boundary":
        pairwise = grouped_boundary_pairwise_logistic_loss(
            scores,
            rmsd,
            group,
            min_rmsd_gap=min_rmsd_gap,
            margin=pairwise_margin,
        )
    else:
        raise ValueError(f"unknown pairwise_mode={pairwise_mode!r}")
    total = float(listwise_weight) * listwise + float(pairwise_weight) * pairwise
    return total, listwise, pairwise


def ranking_score_gradients(
    detached_scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    **kwargs,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiate a full-group rank loss with respect to pose scores.

    The returned coefficients can be applied to recomputed microbatch scores,
    giving exact listwise/pairwise score gradients without retaining all pose
    forward graphs in GPU memory at once.
    """
    score_leaf = detached_scores.detach().requires_grad_(True)
    total, listwise, pairwise = grouped_ranking_loss(score_leaf, rmsd, group, **kwargs)
    gradient = torch.autograd.grad(total, score_leaf, retain_graph=False, create_graph=False)[0]
    return gradient.detach(), {
        "rank_loss": total.detach(),
        "listwise_loss": listwise.detach(),
        "pairwise_loss": pairwise.detach(),
    }


def ranking_output_gradients(
    detached_scores: torch.Tensor,
    detached_logit_lt5: torch.Tensor,
    detached_logit_lt2_given_lt5: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    ordinal_weight: float = 0.0,
    pose_bin_weights: tuple[float, float, float, float] | None = None,
    **kwargs,
) -> tuple[tuple[torch.Tensor, torch.Tensor, torch.Tensor], dict[str, torch.Tensor]]:
    """Full-group gradients for rank score and both cumulative ordinal logits."""
    score_leaf = detached_scores.detach().requires_grad_(True)
    lt5_leaf = detached_logit_lt5.detach().requires_grad_(True)
    lt2_leaf = detached_logit_lt2_given_lt5.detach().requires_grad_(True)
    rank, listwise, pairwise = grouped_ranking_loss(score_leaf, rmsd, group, **kwargs)
    ordinal = (
        cumulative_ordinal_pose_loss(
            lt5_leaf,
            lt2_leaf,
            rmsd,
            sample_weight=rmsd_bin_sample_weight(rmsd, pose_bin_weights),
        )
        if float(ordinal_weight) > 0
        else (lt5_leaf.sum() + lt2_leaf.sum()) * 0.0
    )
    total = rank + float(ordinal_weight) * ordinal
    gradients = torch.autograd.grad(
        total,
        (score_leaf, lt5_leaf, lt2_leaf),
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )
    return tuple(gradient.detach() for gradient in gradients), {
        "rank_loss": rank.detach(),
        "listwise_loss": listwise.detach(),
        "pairwise_loss": pairwise.detach(),
        "ordinal_loss": ordinal.detach(),
        "ranker_total_loss": total.detach(),
    }


def scorer_loss(
    *,
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    pi: torch.Tensor,
    mu: torch.Tensor,
    sigma: torch.Tensor,
    candidate_distances: torch.Tensor,
    pair_mask: torch.Tensor,
    graph_is_prior_valid: torch.Tensor,
    ordinal_logit_lt5: torch.Tensor | None = None,
    ordinal_logit_lt2_given_lt5: torch.Tensor | None = None,
    prior_weight: float = 0.2,
    listwise_weight: float = 1.0,
    pairwise_weight: float = 0.25,
    ordinal_weight: float = 0.0,
    rmsd_temperature: float = 1.0,
    score_temperature: float = 1.0,
    min_rmsd_gap: float = 0.5,
    pairwise_margin: float = 0.2,
    prior_contact_radius: float = 7.0,
    listwise_mode: str = "rmsd_softmax",
    pairwise_mode: str = "gap",
    pose_bin_weights: tuple[float, float, float, float] | None = None,
) -> ScorerLossOutput:
    prior_nll = near_native_prior_nll(
        pi,
        mu,
        sigma,
        candidate_distances,
        pair_mask,
        graph_is_prior_valid,
        contact_radius=prior_contact_radius,
    )
    _, listwise, pairwise = grouped_ranking_loss(
        scores,
        rmsd,
        group,
        listwise_weight=1.0,
        pairwise_weight=1.0,
        rmsd_temperature=rmsd_temperature,
        score_temperature=score_temperature,
        min_rmsd_gap=min_rmsd_gap,
        pairwise_margin=pairwise_margin,
        listwise_mode=listwise_mode,
        pairwise_mode=pairwise_mode,
    )
    if ordinal_weight:
        if ordinal_logit_lt5 is None or ordinal_logit_lt2_given_lt5 is None:
            raise ValueError("ordinal_weight>0 requires both cumulative ordinal logits")
        ordinal = cumulative_ordinal_pose_loss(
            ordinal_logit_lt5,
            ordinal_logit_lt2_given_lt5,
            rmsd,
            sample_weight=rmsd_bin_sample_weight(rmsd, pose_bin_weights),
        )
    else:
        ordinal = scores.sum() * 0.0
    total = (
        float(prior_weight) * prior_nll
        + float(listwise_weight) * listwise
        + float(pairwise_weight) * pairwise
        + float(ordinal_weight) * ordinal
    )
    return ScorerLossOutput(total, prior_nll, listwise, pairwise, ordinal)


def _rankdata(values: torch.Tensor) -> torch.Tensor:
    # Candidate scores and RMSDs are effectively continuous.  The deterministic
    # double argsort is sufficient here and avoids a scipy dependency on nodes.
    return torch.argsort(torch.argsort(values)).to(torch.float32)


def _spearman(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return math.nan
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = rx.square().sum().sqrt() * ry.square().sum().sqrt()
    if float(denom) == 0.0:
        return math.nan
    return float((rx * ry).sum() / denom)


def grouped_rerank_metrics(
    scores: torch.Tensor,
    rmsd: torch.Tensor,
    group: torch.Tensor,
    *,
    original_rank: torch.Tensor | None = None,
) -> dict[str, float]:
    """Compute metrics after all distributed ranks have been gathered."""
    rows = []
    spearman_values = []
    ndcg_values = []
    reciprocal_ranks = []
    for _, mask in _unique_groups(group):
        score = scores[mask].detach().float().cpu()
        distance = rmsd[mask].detach().float().cpu()
        if not distance.numel():
            continue
        order = torch.argsort(score, descending=True)
        top1 = float(distance[order[0]])
        top5 = float(distance[order[: min(5, distance.numel())]].min())
        oracle = float(distance.min())
        random_lt2 = float((distance < 2.0).float().mean())
        random_lt5 = float((distance < 5.0).float().mean())
        if original_rank is not None:
            source_order = torch.argsort(original_rank[mask].detach().cpu())
            sample0 = float(distance[source_order[0]])
        else:
            sample0 = math.nan

        relevance = torch.exp(-distance / 2.0)
        discounts = 1.0 / torch.log2(torch.arange(distance.numel(), dtype=torch.float32) + 2.0)
        dcg = float((relevance[order] * discounts).sum())
        ideal = torch.argsort(relevance, descending=True)
        idcg = float((relevance[ideal] * discounts).sum())
        ndcg_values.append(dcg / max(idcg, 1e-8))
        spearman = _spearman(score, -distance)
        if math.isfinite(spearman):
            spearman_values.append(spearman)
        positive_locations = torch.where(distance[order] < 2.0)[0]
        if positive_locations.numel():
            reciprocal_ranks.append(1.0 / (int(positive_locations[0]) + 1))
        rows.append((top1, top5, oracle, random_lt2, random_lt5, sample0))

    if not rows:
        return {"rerank_n_groups": 0.0}
    values = torch.tensor([[row[i] for i in range(5)] for row in rows], dtype=torch.float32)
    top1, top5, oracle, random_lt2, random_lt5 = values.T
    sample0 = torch.tensor([row[5] for row in rows], dtype=torch.float32)
    oracle_lt2 = (oracle < 2.0).float()
    oracle_lt5 = (oracle < 5.0).float()
    top1_lt2 = (top1 < 2.0).float()
    top1_lt5 = (top1 < 5.0).float()
    top5_lt2 = (top5 < 2.0).float()
    top5_lt5 = (top5 < 5.0).float()
    capable_lt2 = oracle_lt2.bool()
    capable_lt5 = oracle_lt5.bool()
    random2 = float(random_lt2.mean())
    random5 = float(random_lt5.mean())
    top1_rate2 = float(top1_lt2.mean())
    top1_rate5 = float(top1_lt5.mean())
    oracle_rate2 = float(oracle_lt2.mean())
    oracle_rate5 = float(oracle_lt5.mean())
    recovery2 = (top1_rate2 - random2) / max(oracle_rate2 - random2, 1e-8)
    recovery5 = (top1_rate5 - random5) / max(oracle_rate5 - random5, 1e-8)
    output = {
        "rerank_n_groups": float(len(rows)),
        "rerank_top1_lt2": 100.0 * top1_rate2,
        "rerank_top1_lt5": 100.0 * top1_rate5,
        "rerank_top5_lt2": 100.0 * float(top5_lt2.mean()),
        "rerank_top5_lt5": 100.0 * float(top5_lt5.mean()),
        "rerank_oracle_lt2": 100.0 * oracle_rate2,
        "rerank_oracle_lt5": 100.0 * oracle_rate5,
        "rerank_random_lt2": 100.0 * random2,
        "rerank_random_lt5": 100.0 * random5,
        "rerank_recovery_lt2": float(recovery2),
        "rerank_recovery_lt5": float(recovery5),
        "rerank_top1_median_rmsd": float(top1.median()),
        "rerank_mean_regret": float((top1 - oracle).mean()),
        "rerank_median_regret": float((top1 - oracle).median()),
        "rerank_oracle_capable_top1_recall_lt2": (
            100.0 * float(top1_lt2[capable_lt2].mean()) if torch.any(capable_lt2) else math.nan
        ),
        "rerank_oracle_capable_top5_recall_lt2": (
            100.0 * float(top5_lt2[capable_lt2].mean()) if torch.any(capable_lt2) else math.nan
        ),
        "rerank_oracle_capable_top1_recall_lt5": (
            100.0 * float(top1_lt5[capable_lt5].mean()) if torch.any(capable_lt5) else math.nan
        ),
        "rerank_mean_spearman": (
            float(sum(spearman_values) / len(spearman_values)) if spearman_values else math.nan
        ),
        "rerank_mean_ndcg": float(sum(ndcg_values) / len(ndcg_values)),
        "rerank_mrr_lt2": (
            float(sum(reciprocal_ranks) / len(reciprocal_ranks)) if reciprocal_ranks else 0.0
        ),
    }
    finite_sample0 = torch.isfinite(sample0)
    if torch.any(finite_sample0):
        output["sample0_lt2"] = 100.0 * float((sample0[finite_sample0] < 2.0).float().mean())
        output["sample0_lt5"] = 100.0 * float((sample0[finite_sample0] < 5.0).float().mean())
    return output


def gather_pose_records(accelerator, local_records: list[dict]) -> list[dict]:
    """Gather variable-length whole-group records without tensor padding.

    Validation/test samplers must shard complete groups and must not pad or
    duplicate them.  Use PyTorch's process-group primitive directly because
    accelerate 0.15's ``gather_object`` returns only the local list here.
    """
    if getattr(accelerator, "num_processes", 1) == 1:
        return list(local_records)
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        raise RuntimeError("distributed scorer metrics require an initialized torch process group")
    gathered_by_rank: list[list[dict] | None] = [None] * int(accelerator.num_processes)
    torch.distributed.all_gather_object(gathered_by_rank, list(local_records))
    gathered = [
        record
        for rank_records in gathered_by_rank
        for record in (rank_records or [])
    ]
    pose_uids = [record["pose_uid"] for record in gathered]
    duplicate_count = len(pose_uids) - len(set(pose_uids))
    if duplicate_count:
        raise RuntimeError(f"DDP metric gather duplicated {duplicate_count} pose_uid values")
    return gathered


def assert_complete_group_records(records: list[dict], expected_counts: dict[str, int]) -> None:
    observed = Counter(record["pose_group_uid"] for record in records)
    missing_or_split = {
        key: (observed.get(key, 0), count)
        for key, count in expected_counts.items()
        if observed.get(key, 0) != count
    }
    if missing_or_split:
        first = list(missing_or_split.items())[:10]
        raise RuntimeError(f"incomplete scorer groups after gather; first mismatches: {first}")


def checkpoint_selection_key(metrics: dict[str, float]) -> tuple[float, float, float, float]:
    """Lexicographic model-selection key; fixed test metrics must never enter."""
    return (
        float(metrics.get("rerank_top1_lt2", -math.inf)),
        float(metrics.get("rerank_top1_lt5", -math.inf)),
        -float(metrics.get("rerank_mean_regret", math.inf)),
        float(metrics.get("rerank_mean_spearman", -math.inf)),
    )
