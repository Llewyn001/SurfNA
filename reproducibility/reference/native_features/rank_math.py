"""Unchanged MDN/ranking mathematics, without importing any legacy run module."""
from collections import defaultdict
import hashlib


def finite(*values):
    import torch
    for value in values:
        if not bool(torch.isfinite(value).all()):
            raise RuntimeError("Nonfinite value; no numerical masking or candidate skipping")


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def dense_nearest_score(head, ligand, surface, distances, log_probability):
    """Original all-pairs computation, only for one prespecified parity check."""
    import torch
    ns = len(surface)
    parameters = [[], [], []]
    for start in range(0, len(ligand), max(1, 4096 // ns)):
        part = ligand[start:start + max(1, 4096 // ns)]
        features = torch.cat([part[:, None].expand(-1, ns, -1),
                              surface[None].expand(len(part), -1, -1)], -1)
        values = head.predict_parameters(features)
        finite(*values)
        for target, value in zip(parameters, values):
            target.append(value)
    pi, mu, sigma = [torch.cat(parts)[None] for parts in parameters]
    lp = log_probability(pi, sigma, mu, distances[None])
    nearest = distances.topk(8, largest=False, dim=-1).indices
    result = lp[0].gather(-1, nearest).mean()
    finite(lp, result)
    return float(result)


def ranking_metrics(rows, score_key):
    """Formula-identical copy of native_prior_v3_20260828/run_arm.py:351–389."""
    import numpy as np
    from scipy.stats import spearmanr
    by_group = defaultdict(list)
    for row in rows:
        by_group[row['name']].append(row)
    records = []
    for name, group in by_group.items():
        if len(group) != 8:
            raise RuntimeError('Ranking group is not exactly K8')
        ordered = sorted(group, key=lambda r: (-r[score_key], r['ordinal']))
        rmsd = np.array([r['rmsd'] for r in group])
        scores = np.array([r[score_key] for r in group])
        if not np.isfinite(rmsd).all() or not np.isfinite(scores).all():
            raise RuntimeError('Nonfinite ranking label/score')
        relevance = np.exp(-rmsd/2)
        order = np.array(sorted(range(8), key=lambda i: (-scores[i], group[i]['ordinal'])))
        discount = np.log2(np.arange(8)+2)
        ideal = np.sum(np.sort(relevance)[::-1]/discount)
        rho = float(spearmanr(scores, -rmsd)[0]) if np.ptp(scores)>0 and np.ptp(rmsd)>0 else None
        if rho is not None and not np.isfinite(rho):
            rho = None
        first = sorted(group, key=lambda r: r['ordinal'])[0]
        records.append(dict(name=name, top1=ordered[0]['rmsd'], top3=min(r['rmsd'] for r in ordered[:3]),
            oracle=float(rmsd.min()), sample0=first['rmsd'], density_lt2=float((rmsd<2).mean()),
            regret=float(ordered[0]['rmsd']-rmsd.min()), spearman=rho,
            ndcg=float(np.sum(relevance[order]/discount)/ideal) if ideal>0 else 0.0))
    if not records:
        return dict(n_groups=0)
    mean = lambda field, threshold: float(np.mean([r[field]<threshold for r in records]))
    rhos = [r['spearman'] for r in records if r['spearman'] is not None]
    return dict(n_groups=len(records), top1_lt2=mean('top1', 2), top1_lt5=mean('top1', 5),
        top3_lt2=mean('top3', 2), median_rmsd=float(np.median([r['top1'] for r in records])),
        oracle_lt2=mean('oracle', 2), oracle_lt5=mean('oracle', 5),
        fixed_first_of8_lt2=mean('sample0', 2),
        random_expected_lt2=float(np.mean([r['density_lt2'] for r in records])),
        regret=float(np.mean([r['regret'] for r in records])),
        spearman=float(np.mean(rhos)) if rhos else None, spearman_valid_groups=len(rhos),
        ndcg=float(np.mean([r['ndcg'] for r in records])), ndcg_relevance='exp(-RMSD/2)', groups=records)
