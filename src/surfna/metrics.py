"""Original ranking metrics and validation selection rule."""
import math,statistics
from collections import defaultdict

def _ranks(values):
    order = sorted(range(len(values)), key=values.__getitem__)
    out, start = [0.0]*len(values), 0
    while start < len(order):
        end = start+1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        for index in order[start:end]:
            out[index] = (start+1+end)/2
        start = end
    return out

def _spearman(x, y):
    a, b = _ranks(x), _ranks(y)
    center = (len(x)+1)/2
    a, b = [v-center for v in a], [v-center for v in b]
    denominator = math.sqrt(math.fsum(v*v for v in a)*math.fsum(v*v for v in b))
    return max(-1.0, min(1.0, math.fsum(u*v for u, v in zip(a, b))/denominator)) if denominator else None

def ranking_metrics(rows, score_key="score"):
    """Same ranking math as the frozen B recipe, with explicit full denominators."""
    by_name = defaultdict(list)
    for row in rows:
        if any(not math.isfinite(float(row[key])) for key in ("rmsd", score_key)) or row["rmsd"] < 0:
            raise ValueError("Nonfinite score or invalid RMSD")
        by_name[row["name"]].append(row)
    if not by_name:
        raise ValueError("Empty ranking population")
    records, k_seen = [], None
    for name, group in sorted(by_name.items()):
        k = len(group)
        if (sorted(row["ordinal"] for row in group) != list(range(k))
                or (k_seen is not None and k != k_seen)):
            raise ValueError("Missing/duplicate ordinals or inconsistent K")
        k_seen = k
        group.sort(key=lambda row: row["ordinal"])
        ordered = sorted(group, key=lambda row: (-row[score_key], row["ordinal"]))
        rmsd = [float(row["rmsd"]) for row in group]
        scores = [float(row[score_key]) for row in group]
        gains = sorted([math.exp(-r/2) for r in rmsd], reverse=True)
        discounts = [math.log2(index+2) for index in range(k)]
        ideal = math.fsum(gain/discount for gain, discount in zip(gains, discounts))
        observed = math.fsum(math.exp(-row["rmsd"]/2)/discount for row, discount in zip(ordered, discounts))
        records.append(dict(name=name, top1=ordered[0]["rmsd"], top1_ordinal=ordered[0]["ordinal"],
            top3=min(row["rmsd"] for row in ordered[:3]), top5=min(row["rmsd"] for row in ordered[:5]),
            oracle=min(rmsd), sample0=rmsd[0], density_lt2=sum(r < 2 for r in rmsd)/k,
            n_good=sum(r < 2 for r in rmsd), regret=ordered[0]["rmsd"]-min(rmsd),
            spearman=_spearman(scores, [-r for r in rmsd]), ndcg=observed/ideal if ideal else 0.0))
    n = len(records)
    rhos = [row["spearman"] for row in records if row["spearman"] is not None]
    result = dict(n_groups=n, n_poses=n*k_seen, k=k_seen,
        median_rmsd=statistics.median(row["top1"] for row in records),
        regret=math.fsum(row["regret"] for row in records)/n,
        spearman=math.fsum(rhos)/len(rhos) if rhos else None, spearman_valid_groups=len(rhos),
        ndcg=math.fsum(row["ndcg"] for row in records)/n, ndcg_relevance="exp(-RMSD/2)",
        random_expected_lt2=math.fsum(row["density_lt2"] for row in records)/n,
        zero_positive_groups=sum(row["n_good"] == 0 for row in records),
        all_positive_groups=sum(row["n_good"] == k_seen for row in records),
        good_pose_histogram=[sum(row["n_good"] == i for row in records) for i in range(k_seen+1)],
        groups=records)
    for field in ("top1", "top3", "top5", "oracle", "sample0"):
        for threshold in (2, 5):
            key = field+"_lt"+str(threshold)
            count = sum(row[field] < threshold for row in records)
            result[key] = count/n
            result[key+"_count"] = count
    return result

def selection_key(metrics):
    values = (metrics["top1_lt2"], -metrics["regret"], metrics["ndcg"])
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Nonfinite validation selection key")
    return values
