#!/usr/bin/env python3
"""Audit and summarize the frozen SurfNA generation-intervention supplement."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import tempfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


METRICS = ("pose_density_rmsd_lt2", "best_of_k10_coverage_rmsd_lt2", "best_of_k10_rmsd")
OLD_EPOCHS = (25, 50, 75, 100, 125, 150, 175, 200)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_csv(path: Path, rows) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def arm_slug(arm: str) -> str:
    return arm.lower().replace("-", "_")


def read_pose_manifest(path: Path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 890:
        raise RuntimeError(f"expected 890 pose rows: {path} has {len(rows)}")
    grouped = defaultdict(dict)
    for row in rows:
        name = row["complex_name"]
        sample = int(row["sample_idx"])
        rmsd = float(row["rmsd"])
        if not math.isfinite(rmsd) or sample in grouped[name]:
            raise RuntimeError(f"invalid pose row in {path}: {name}/{sample}")
        grouped[name][sample] = rmsd
    if len(grouped) != 89:
        raise RuntimeError(f"expected 89 complexes: {path} has {len(grouped)}")
    result = {}
    for name, by_sample in grouped.items():
        if sorted(by_sample) != list(range(10)):
            raise RuntimeError(f"K10 sample set mismatch in {path}: {name}")
        values = np.asarray([by_sample[index] for index in range(10)], dtype=float)
        result[name] = {
            "rmsd": values,
            "pose_density_rmsd_lt2": float(np.mean(values < 2.0)),
            "best_of_k10_coverage_rmsd_lt2": float(np.min(values) < 2.0),
            "best_of_k10_rmsd": float(np.min(values)),
        }
    return result


def unit_metrics(per_complex):
    return {
        metric: float(np.mean([values[metric] for values in per_complex.values()]))
        for metric in METRICS
    }


def benefit(a_value: float, b_value: float, metric: str) -> float:
    """Positive means A is better than B."""
    if metric == "best_of_k10_rmsd":
        return b_value - a_value
    return a_value - b_value


def bootstrap_hierarchical(seed_effects, label: str, repetitions: int = 20000):
    arrays = [np.asarray(seed_effects[seed], dtype=float) for seed in sorted(seed_effects)]
    if len(arrays) != 3 or any(array.shape != (89,) for array in arrays):
        raise RuntimeError(f"paired bootstrap input is not 3x89 for {label}")
    matrix = np.stack(arrays, axis=0)
    point = float(matrix.mean())
    rng_seed = int.from_bytes(hashlib.sha256(label.encode()).digest()[:8], "little")
    rng = np.random.default_rng(rng_seed)
    draws = np.empty(repetitions, dtype=float)
    for repeat in range(repetitions):
        selected_seeds = rng.integers(0, 3, size=3)
        replicate = []
        for seed_index in selected_seeds:
            complex_indices = rng.integers(0, 89, size=89)
            replicate.append(matrix[seed_index, complex_indices].mean())
        draws[repeat] = np.mean(replicate)
    return {
        "point_estimate": point,
        "ci95_low": float(np.quantile(draws, 0.025)),
        "ci95_high": float(np.quantile(draws, 0.975)),
        "probability_benefit_gt0": float(np.mean(draws > 0.0)),
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": rng_seed,
    }


def load_matrix(root: Path):
    with (root / "00_protocol/formal_matrix.tsv").open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    if len(rows) != 90:
        raise RuntimeError(f"formal matrix is not 90 rows: {len(rows)}")
    return rows


def new_unit_path(root: Path, row) -> Path:
    return root / "05_generation_raw" / arm_slug(row["arm"]) / f"seed{row['seed']}" / f"epoch_{int(row['epoch']):04d}" / row["condition_id"]


def old_manifest(v5: Path, arm: str, seed: int, epoch: int) -> Path:
    return v5 / "06_validation_trajectory/raw" / arm_slug(arm) / f"seed{seed}" / f"epoch_{epoch:04d}" / "validation_pose_manifest.tsv"


def summarize_new(root: Path, matrix):
    datasets = {}
    unit_rows = []
    for row in matrix:
        target = new_unit_path(root, row)
        receipt = target / "UNIT_COMPLETE.json"
        exit_ok = target / "RUN_EXIT_OK.txt"
        if not receipt.is_file() or not exit_ok.is_file():
            raise FileNotFoundError(f"incomplete formal unit: {target}")
        receipt_data = json.loads(receipt.read_text())
        if not receipt_data.get("passed") or receipt_data.get("failure_count") != 0:
            raise RuntimeError(f"failed formal unit receipt: {receipt}")
        data = read_pose_manifest(target / "validation_pose_manifest.tsv")
        key = (row["arm"], int(row["seed"]), int(row["epoch"]), row["condition_id"])
        datasets[key] = data
        metrics = unit_metrics(data)
        unit_rows.append({
            "arm": row["arm"], "training_seed": int(row["seed"]), "epoch": int(row["epoch"]),
            "condition_id": row["condition_id"], "gate_alpha": float(row["gate_alpha"]),
            "chemistry_condition": row["chemistry_condition"],
            "shuffle_seed": row["shuffle_seed"], **metrics,
        })
    return datasets, unit_rows


def new_intervention_contrasts(datasets):
    detailed = []
    bootstrap_rows = []
    for arm in ("Transfer-All", "Transfer-NonSurface"):
        for epoch in (25, 100, 200):
            comparisons = [
                ("surface_gate_true_vs_off", "surface_off_true_gate0"),
                ("chemistry_true_vs_shuffle_20260911", "chemistry_shuffle_20260911"),
                ("chemistry_true_vs_shuffle_20260912", "chemistry_shuffle_20260912"),
                ("chemistry_true_vs_shuffle_20260913", "chemistry_shuffle_20260913"),
            ]
            for comparison, altered_condition in comparisons:
                for metric in METRICS:
                    seed_effects = {}
                    for seed in (0, 1, 2):
                        baseline = datasets[(arm, seed, epoch, "baseline_true_gate1")]
                        altered = datasets[(arm, seed, epoch, altered_condition)]
                        names = sorted(set(baseline) & set(altered))
                        if len(names) != 89:
                            raise RuntimeError("new intervention pairing lost a val89 identity")
                        values = [benefit(baseline[name][metric], altered[name][metric], metric) for name in names]
                        seed_effects[seed] = values
                        detailed.append({
                            "arm": arm, "epoch": epoch, "comparison": comparison, "metric": metric,
                            "training_seed": seed, "mean_benefit": float(np.mean(values)),
                            "positive_direction": "true/gate1 better than altered",
                        })
                    boot = bootstrap_hierarchical(seed_effects, f"new|{arm}|{epoch}|{comparison}|{metric}")
                    bootstrap_rows.append({
                        "arm": arm, "epoch": epoch, "comparison": comparison, "metric": metric,
                        "positive_direction": "true/gate1 better than altered", **boot,
                    })

            for metric in METRICS:
                seed_effects = {}
                for seed in (0, 1, 2):
                    baseline = datasets[(arm, seed, epoch, "baseline_true_gate1")]
                    shuffles = [datasets[(arm, seed, epoch, f"chemistry_shuffle_{shuffle}")] for shuffle in (20260911, 20260912, 20260913)]
                    names = sorted(baseline)
                    values = []
                    for name in names:
                        shuffled_value = float(np.mean([data[name][metric] for data in shuffles]))
                        values.append(benefit(baseline[name][metric], shuffled_value, metric))
                    seed_effects[seed] = values
                    detailed.append({
                        "arm": arm, "epoch": epoch, "comparison": "chemistry_true_vs_mean_3shuffle",
                        "metric": metric, "training_seed": seed, "mean_benefit": float(np.mean(values)),
                        "positive_direction": "true chemistry better than mean shuffle",
                    })
                boot = bootstrap_hierarchical(seed_effects, f"new|{arm}|{epoch}|chemmean|{metric}")
                bootstrap_rows.append({
                    "arm": arm, "epoch": epoch, "comparison": "chemistry_true_vs_mean_3shuffle",
                    "metric": metric, "positive_direction": "true chemistry better than mean shuffle", **boot,
                })
    return detailed, bootstrap_rows


def baseline_parity(root: Path, v5: Path, datasets):
    rows = []
    for arm in ("Transfer-All", "Transfer-NonSurface"):
        for seed in (0, 1, 2):
            for epoch in (25, 100, 200):
                new = datasets[(arm, seed, epoch, "baseline_true_gate1")]
                old = read_pose_manifest(old_manifest(v5, arm, seed, epoch))
                names = sorted(set(new) & set(old))
                differences = [abs(float(new[name]["rmsd"][sample]) - float(old[name]["rmsd"][sample])) for name in names for sample in range(10)]
                new_metrics, old_metrics = unit_metrics(new), unit_metrics(old)
                rows.append({
                    "arm": arm, "training_seed": seed, "epoch": epoch,
                    "paired_complexes": len(names), "paired_poses": len(differences),
                    "max_abs_pose_rmsd_difference": float(max(differences)),
                    "pose_density_abs_difference": abs(new_metrics["pose_density_rmsd_lt2"] - old_metrics["pose_density_rmsd_lt2"]),
                    "coverage_abs_difference": abs(new_metrics["best_of_k10_coverage_rmsd_lt2"] - old_metrics["best_of_k10_coverage_rmsd_lt2"]),
                })
    return rows


def summarize_existing_transfer_vs_scratch(v5: Path):
    data = {}
    unit_rows = []
    for arm in ("Scratch-Full8", "Transfer-All", "Transfer-NonSurface"):
        for seed in (0, 1, 2):
            for epoch in OLD_EPOCHS:
                path = old_manifest(v5, arm, seed, epoch)
                dataset = read_pose_manifest(path)
                data[(arm, seed, epoch)] = dataset
                unit_rows.append({"arm": arm, "training_seed": seed, "epoch": epoch, **unit_metrics(dataset)})

    bootstrap_rows = []
    seed_rows = []
    for transfer_arm in ("Transfer-All", "Transfer-NonSurface"):
        for epoch in OLD_EPOCHS:
            for metric in METRICS:
                seed_effects = {}
                for seed in (0, 1, 2):
                    transfer = data[(transfer_arm, seed, epoch)]
                    scratch = data[("Scratch-Full8", seed, epoch)]
                    names = sorted(set(transfer) & set(scratch))
                    if len(names) != 89:
                        raise RuntimeError("Transfer-vs-Scratch pairing lost a val89 identity")
                    values = [benefit(transfer[name][metric], scratch[name][metric], metric) for name in names]
                    seed_effects[seed] = values
                    seed_rows.append({
                        "contrast": f"{transfer_arm}-Scratch-Full8", "epoch_or_aulc": str(epoch),
                        "metric": metric, "training_seed": seed, "mean_benefit": float(np.mean(values)),
                        "positive_direction": "transfer better than paired scratch",
                    })
                boot = bootstrap_hierarchical(seed_effects, f"scratch|{transfer_arm}|{epoch}|{metric}")
                bootstrap_rows.append({
                    "contrast": f"{transfer_arm}-Scratch-Full8", "epoch_or_aulc": str(epoch),
                    "metric": metric, "positive_direction": "transfer better than paired scratch", **boot,
                })

        for metric in METRICS:
            seed_effects = {}
            for seed in (0, 1, 2):
                names = sorted(data[(transfer_arm, seed, 25)])
                values = []
                for name in names:
                    transfer_curve = np.asarray([data[(transfer_arm, seed, epoch)][name][metric] for epoch in OLD_EPOCHS])
                    scratch_curve = np.asarray([data[("Scratch-Full8", seed, epoch)][name][metric] for epoch in OLD_EPOCHS])
                    if metric == "best_of_k10_rmsd":
                        effect_curve = scratch_curve - transfer_curve
                    else:
                        effect_curve = transfer_curve - scratch_curve
                    values.append(float(np.trapz(effect_curve, x=np.asarray(OLD_EPOCHS)) / 175.0))
                seed_effects[seed] = values
                seed_rows.append({
                    "contrast": f"{transfer_arm}-Scratch-Full8", "epoch_or_aulc": "AULC25-200",
                    "metric": metric, "training_seed": seed, "mean_benefit": float(np.mean(values)),
                    "positive_direction": "transfer better than paired scratch",
                })
            boot = bootstrap_hierarchical(seed_effects, f"scratch|{transfer_arm}|aulc|{metric}")
            bootstrap_rows.append({
                "contrast": f"{transfer_arm}-Scratch-Full8", "epoch_or_aulc": "AULC25-200",
                "metric": metric, "positive_direction": "transfer better than paired scratch", **boot,
            })
    return unit_rows, seed_rows, bootstrap_rows


def report_markdown(new_bootstrap, scratch_bootstrap, parity_rows):
    selected_new = [row for row in new_bootstrap if row["epoch"] == 200 and row["comparison"] in ("surface_gate_true_vs_off", "chemistry_true_vs_mean_3shuffle")]
    selected_scratch = [row for row in scratch_bootstrap if row["epoch_or_aulc"] in ("200", "AULC25-200")]
    parity_max = max(row["max_abs_pose_rmsd_difference"] for row in parity_rows)
    parity_density = max(row["pose_density_abs_difference"] for row in parity_rows)
    lines = [
        "# SurfNA generation-level Surface-transfer supplement", "",
        "## Integrity", "",
        "- Formal generation interventions: 90/90 valid units (2 arms × 3 seeds × 3 EMA epochs × 5 conditions).",
        "- Each unit contains val89 × K10 = 890 finite pose-level RMSD values.",
        "- Sampling uses 20 denoising steps and the frozen per-complex stream derived from seed 20260826.",
        f"- Patched-baseline parity audit: maximum per-pose RMSD difference {parity_max:.6g}; maximum pose-density difference {parity_density:.6g}.",
        "- The fixed test128 set was not rerun or used for intervention selection.", "",
        "## Generation-level intervention effects at EMA 200", "",
        "Positive values indicate benefit of the intact/true Surface condition.", "",
        "| Arm | Comparison | Metric | Benefit | 95% hierarchical bootstrap CI | P(benefit>0) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in selected_new:
        lines.append(
            f"| {row['arm']} | {row['comparison']} | {row['metric']} | {row['point_estimate']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['probability_benefit_gt0']:.3f} |"
        )
    lines.extend(["", "## Transfer versus paired Scratch", "", "Positive values indicate benefit of transfer.", "", "| Contrast | Epoch/AULC | Metric | Benefit | 95% hierarchical bootstrap CI | P(benefit>0) |", "|---|---|---|---:|---:|---:|"])
    for row in selected_scratch:
        lines.append(
            f"| {row['contrast']} | {row['epoch_or_aulc']} | {row['metric']} | {row['point_estimate']:.4f} | "
            f"[{row['ci95_low']:.4f}, {row['ci95_high']:.4f}] | {row['probability_benefit_gt0']:.3f} |"
        )
    lines.extend(["", "## Interpretation rule", "", "A mechanism is treated as supported only when the direction is consistent with the preregistered benefit and the paired hierarchical interval excludes zero. Chemistry-shuffle results are averaged across all three frozen permutations for the primary chemistry interpretation; individual repeats remain available in the full tables.", ""])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    contract_path = root / "00_protocol/FROZEN_CONTRACT.json"
    contract = json.loads(contract_path.read_text())
    if contract.get("formal_unit_count") != 90:
        raise RuntimeError("frozen contract does not specify 90 formal units")
    matrix = load_matrix(root)
    v5 = Path(contract["parent_v5_read_only"])
    v6 = Path(contract["parent_v6_read_only"])
    if not (v6 / "03_integrity/MECHANISM_RECOMPUTE_MATRIX_COMPLETE.json").is_file():
        raise FileNotFoundError("parent v6 mechanism integrity receipt is missing")

    datasets, new_units = summarize_new(root, matrix)
    new_seed_contrasts, new_bootstrap = new_intervention_contrasts(datasets)
    parity_rows = baseline_parity(root, v5, datasets)
    old_units, scratch_seed_contrasts, scratch_bootstrap = summarize_existing_transfer_vs_scratch(v5)

    stats = root / "06_statistics"
    write_csv(stats / "generation_intervention_unit_metrics.csv", new_units)
    write_csv(stats / "generation_intervention_seed_contrasts.csv", new_seed_contrasts)
    write_csv(stats / "generation_intervention_hierarchical_bootstrap.csv", new_bootstrap)
    write_csv(stats / "patched_baseline_parity.csv", parity_rows)
    write_csv(stats / "existing_v5_unit_metrics.csv", old_units)
    write_csv(stats / "transfer_vs_scratch_seed_contrasts.csv", scratch_seed_contrasts)
    write_csv(stats / "transfer_vs_scratch_hierarchical_bootstrap.csv", scratch_bootstrap)
    report_path = root / "FINAL_GENERATION_INTERVENTION_REPORT.md"
    report_path.write_text(report_markdown(new_bootstrap, scratch_bootstrap, parity_rows))

    outputs = [path for path in stats.iterdir() if path.is_file()] + [report_path]
    completion = {
        "schema_version": "surfna-generation-intervention-summary-v1",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "formal_generation_units_valid": 90,
        "formal_generation_units_expected": 90,
        "validation_complexes_per_unit": 89,
        "poses_per_complex": 10,
        "new_intervention_bootstrap_rows": len(new_bootstrap),
        "transfer_vs_scratch_bootstrap_rows": len(scratch_bootstrap),
        "parent_v5_validation_manifests_reused": 72,
        "test128_rerun": False,
        "output_sha256": {str(path.relative_to(root)): sha256_file(path) for path in outputs},
        "passed": True,
    }
    atomic_json(stats / "SUPPLEMENT_SUMMARY_COMPLETE.json", completion)
    print(json.dumps(completion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

