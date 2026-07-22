#!/usr/bin/env python3
"""Build the public SurfNA dataset manifests and leakage-audit tables.

This script is intentionally dependency-free. It reads the frozen split and
metadata files from the research workspace and writes small, public artifacts
that contain PDB identifiers and audit results, not coordinates.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
from collections import Counter
from pathlib import Path


BENCHMARKS = ("yan", "philips", "chen", "ruiz")
RELEASE_GENERATOR_SHA256 = "489ba24a32135cdc2cf0d1e4ea7eadf07c33d4b135d5fdd0270e283eeb25de91"
RELEASE_MDN_SHA256 = "a1ad6f136cafb3da20c89524474419159b63aa077f4404a102fe866c32023fd5"
PROTEIN_PRETRAIN_SHA256 = "1153dda9b146e9a896dc769278ac115ef29b47c131b1db0176080881816badd3"
NA_MAP = {
    "A": "A", "DA": "A", "ADE": "A",
    "C": "C", "DC": "C", "CYT": "C",
    "G": "G", "DG": "G", "GUA": "G",
    "U": "U", "DU": "U", "URA": "U",
    "T": "T", "DT": "T", "THY": "T",
}


def read_list(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def read_csv(path: Path, delimiter: str = ",") -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter=delimiter))


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def base_pdb(name: str) -> str:
    return name.split("_", 1)[0].lower()


def normalize_type(value: str) -> str:
    value = value.strip().lower()
    if value in {"rna", "dna", "rna+dna"}:
        return value
    if value == "dna_or_dna_like":
        return "dna"
    if value == "mixed_dna_rna":
        return "rna+dna"
    return "unknown"


def locate_receptor(raw_root: Path, name: str) -> Path | None:
    for directory in (raw_root / name, raw_root / base_pdb(name)):
        candidates = (
            directory / f"{name}_protein.pdb",
            directory / f"{base_pdb(name)}_protein.pdb",
            directory / f"{name}_protein_processed.pdb",
            directory / f"{base_pdb(name)}_protein_processed.pdb",
        )
        for path in candidates:
            if path.exists():
                return path.resolve()
        if directory.exists():
            matches = sorted(directory.glob("*_protein*.pdb"))
            if matches:
                return matches[0].resolve()
    return None


def receptor_sequence_signature(path: Path | None) -> tuple[str, int, int]:
    """Return an exact, chain-order-independent canonical NA sequence digest."""
    if path is None:
        return "", 0, 0
    chains: dict[str, list[str]] = {}
    seen: set[tuple[str, str, str]] = set()
    with path.open(errors="ignore") as handle:
        for line in handle:
            if not line.startswith(("ATOM  ", "HETATM")) or len(line) < 27:
                continue
            residue = line[17:20].strip().upper()
            nucleotide = NA_MAP.get(residue)
            if nucleotide is None:
                continue
            chain = line[21:22].strip() or "_"
            residue_id = line[22:27]
            key = (chain, residue_id, residue)
            if key in seen:
                continue
            seen.add(key)
            chains.setdefault(chain, []).append(nucleotide)
    sequences = sorted("".join(items) for items in chains.values() if items)
    if not sequences:
        return "", 0, 0
    payload = "|".join(sequences).encode()
    return hashlib.sha256(payload).hexdigest(), len(sequences), sum(map(len, sequences))


def count_types(rows: list[dict[str, object]]) -> tuple[int, int, int, int]:
    counts = Counter(str(row.get("na_type", "unknown")) for row in rows)
    return counts["rna"], counts["dna"], counts["rna+dna"], counts["unknown"]


def summary_row(
    stage: str,
    role: str,
    component: str,
    rows: list[dict[str, object]],
    source: str,
    note: str,
) -> dict[str, object]:
    rna, dna, mixed, unknown = count_types(rows)
    return {
        "stage": stage,
        "role": role,
        "component": component,
        "complex_instances": len(rows),
        "unique_pdb_receptors": len({str(row["pdb_id"]) for row in rows}),
        "rna": rna,
        "dna": dna,
        "rna_dna": mixed,
        "unknown_type": unknown,
        "source": source,
        "note": note,
    }


def provenance_row(
    component: str,
    checkpoint_role: str,
    run_id: str,
    checkpoint_sha256: str,
    train_split: str,
    nominal_train: int,
    loaded_train: int,
    val_split: str,
    nominal_val: int,
    loaded_val: int,
    best_epoch: int,
    evidence: str,
    note: str,
) -> dict[str, object]:
    return locals()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    workspace = args.workspace.resolve()
    split_dir = workspace / "data" / "splits"
    metadata_dir = workspace / "data" / "metadata" / "na_ligand_expansion_v1"
    diagnostics_dir = workspace / "data" / "diagnostics"
    raw_root = workspace / "data" / "raw" / "na_expanded_v1_highconf"
    args.out_dir.mkdir(parents=True, exist_ok=True)

    original_names = read_list(
        split_dir / "na_clean_trainval_no_leak_blacklist_v6_ligsuffix_fixmesh1_ddp8"
    )
    new_names = read_list(split_dir / "na_expanded_v1_highconf_selected_tierA")
    train_names = read_list(split_dir / "na_expanded_v1_highconf_trainval_tierA_v6plus")
    development_names = read_list(split_dir / "na_expanded_v1_highconf_test132_fixed")
    mdn_train_names = read_list(
        split_dir / "na_clean_train_no_leak_blacklist_v5_fixmesh1_ddp8"
    )
    mdn_val_names = read_list(
        split_dir / "na_clean_val_no_leak_blacklist_v5_fixmesh1_ddp8"
    )

    composition = read_csv(diagnostics_dir / "na_rna_dna_composition_v6_fixmesh1.csv")
    composition_map = {
        row["complex"]: normalize_type(row["category"])
        for row in composition
        if row["split"] in {"trainval_v6", "test132_v6"}
    }
    highconf_rows = read_csv(split_dir / "na_expanded_v1_highconf_selected_tierA_audit.csv")
    highconf_map = {row["complex_name"]: row for row in highconf_rows}

    sequence_cache: dict[str, tuple[str, int, int, str]] = {}

    def sequence_info(name: str) -> tuple[str, int, int, str]:
        pdb = base_pdb(name)
        if pdb not in sequence_cache:
            path = locate_receptor(raw_root, name)
            digest, chains, residues = receptor_sequence_signature(path)
            sequence_cache[pdb] = (digest, chains, residues, str(path or ""))
        return sequence_cache[pdb]

    original_set = set(original_names)
    train_manifest: list[dict[str, object]] = []
    for name in train_names:
        meta = highconf_map.get(name, {})
        digest, chains, residues, receptor_path = sequence_info(name)
        train_manifest.append(
            {
                "split": "transfer_trainval",
                "source_subset": "original_v6" if name in original_set else "expanded_highconf",
                "complex_id": name,
                "pdb_id": base_pdb(name),
                "na_type": normalize_type(meta.get("na_type", composition_map.get(name, "unknown"))),
                "experimental_method": meta.get("method", ""),
                "resolution_angstrom": meta.get("resolution", ""),
                "deposition_date": meta.get("deposition_date", ""),
                "receptor_sequence_sha256": digest,
                "na_chain_count": chains,
                "canonical_na_residue_count": residues,
                "source_receptor_path": receptor_path,
            }
        )

    development_manifest: list[dict[str, object]] = []
    for name in development_names:
        digest, chains, residues, receptor_path = sequence_info(name)
        development_manifest.append(
            {
                "split": "fixed_development",
                "complex_id": name,
                "pdb_id": base_pdb(name),
                "na_type": composition_map.get(name, "unknown"),
                "receptor_sequence_sha256": digest,
                "na_chain_count": chains,
                "canonical_na_residue_count": residues,
                "source_receptor_path": receptor_path,
            }
        )

    development_type_by_pdb = {
        str(row["pdb_id"]): str(row["na_type"]) for row in development_manifest
    }

    mdn_manifest: list[dict[str, object]] = []
    for role, names in (("train", mdn_train_names), ("validation", mdn_val_names)):
        for name in names:
            digest, chains, residues, receptor_path = sequence_info(name)
            mdn_manifest.append(
                {
                    "model_component": "mdn_scorer",
                    "role": role,
                    "complex_id": name,
                    "pdb_id": base_pdb(name),
                    "na_type": composition_map.get(name, "unknown"),
                    "receptor_sequence_sha256": digest,
                    "na_chain_count": chains,
                    "canonical_na_residue_count": residues,
                    "source_receptor_path": receptor_path,
                }
            )

    benchmark_manifest: list[dict[str, object]] = []
    benchmark_dir = split_dir / "benchmark_v6_ligsuffix_full"
    benchmark_audit = {
        row["benchmark"]: row
        for row in read_csv(benchmark_dir / "audit.tsv", delimiter="\t")
    }
    for benchmark in BENCHMARKS:
        path = benchmark_dir / f"{benchmark}_timesplit_test_v6_ligsuffix_full"
        for name in read_list(path):
            digest, chains, residues, receptor_path = sequence_info(name)
            benchmark_manifest.append(
                {
                    "split": "jiang_benchmark",
                    "benchmark": benchmark,
                    "complex_id": name,
                    "pdb_id": base_pdb(name),
                    "na_type": development_type_by_pdb.get(base_pdb(name), "unknown"),
                    "receptor_sequence_sha256": digest,
                    "na_chain_count": chains,
                    "canonical_na_residue_count": residues,
                    "source_receptor_path": receptor_path,
                }
            )

    public_train_fields = [
        "split", "source_subset", "complex_id", "pdb_id", "na_type",
        "experimental_method", "resolution_angstrom", "deposition_date",
        "receptor_sequence_sha256", "na_chain_count", "canonical_na_residue_count",
    ]
    public_development_fields = [
        "split", "complex_id", "pdb_id", "na_type", "receptor_sequence_sha256",
        "na_chain_count", "canonical_na_residue_count",
    ]
    public_benchmark_fields = [
        "split", "benchmark", "complex_id", "pdb_id", "na_type",
        "receptor_sequence_sha256", "na_chain_count", "canonical_na_residue_count",
    ]
    public_mdn_fields = [
        "model_component", "role", "complex_id", "pdb_id", "na_type",
        "receptor_sequence_sha256", "na_chain_count", "canonical_na_residue_count",
    ]
    write_csv(args.out_dir / "transfer_trainval_manifest.csv", train_manifest, public_train_fields)
    write_csv(
        args.out_dir / "fixed_development_manifest.csv",
        development_manifest,
        public_development_fields,
    )
    write_csv(args.out_dir / "jiang_benchmark_manifest.csv", benchmark_manifest, public_benchmark_fields)
    write_csv(args.out_dir / "mdn_trainval_manifest.csv", mdn_manifest, public_mdn_fields)

    original_rows = [row for row in train_manifest if row["source_subset"] == "original_v6"]
    expanded_rows = [row for row in train_manifest if row["source_subset"] == "expanded_highconf"]
    protein_train_nominal = len(read_list(split_dir / "protein_train_surfdock_fixmesh1"))
    protein_val_nominal = len(read_list(split_dir / "protein_val_surfdock_fixmesh1"))
    mdn_train_nominal = len(mdn_train_names)
    mdn_val_nominal = len(mdn_val_names)
    summary = [
        {
            "stage": "protein_surface_pretraining",
            "role": "train",
            "component": "PDBbind_v2020_surface_filtered",
            "complex_instances": 10294,
            "unique_pdb_receptors": 10294,
            "rna": 0, "dna": 0, "rna_dna": 0, "unknown_type": 0,
            "source": "PDBbind v2020",
            "note": f"{protein_train_nominal} nominal split entries; 10294 graphs actually loaded by the released model's ancestor run.",
        },
        {
            "stage": "protein_surface_pretraining",
            "role": "validation",
            "component": "PDBbind_v2020_surface_filtered",
            "complex_instances": 569,
            "unique_pdb_receptors": 569,
            "rna": 0, "dna": 0, "rna_dna": 0, "unknown_type": 0,
            "source": "PDBbind v2020",
            "note": f"{protein_val_nominal} nominal split entries; 569 graphs actually loaded.",
        },
        summary_row(
            "nucleic_acid_transfer", "trainval", "original_v6", original_rows,
            "Jiang-curated starting collection after quality/leakage filters",
            "One ligand instance retained per PDB receptor.",
        ),
        summary_row(
            "nucleic_acid_transfer", "trainval", "expanded_highconf", expanded_rows,
            "Independent RCSB PDB retrieval",
            "Passed extraction, RDKit, direct-contact, surface, and cache-graph checks.",
        ),
        summary_row(
            "nucleic_acid_transfer", "trainval", "combined", train_manifest,
            "original_v6 + expanded_highconf",
            "Frozen transfer-learning split used by the released generator.",
        ),
        summary_row(
            "model_selection", "development", "fixed_132_receptors", development_manifest,
            "Jiang benchmark receptor set",
            "132 nominal entries; 128 graphs actually loaded by the released generator run; used for checkpoint selection, not fitting.",
        ),
        {
            "stage": "mdn_scorer_transfer",
            "role": "train",
            "component": "na_clean_v5",
            "complex_instances": 1240,
            "unique_pdb_receptors": 868,
            "rna": 0, "dna": 0, "rna_dna": 0, "unknown_type": 1240,
            "source": "SurfNA v5 frozen split/cache",
            "note": f"{mdn_train_nominal} nominal entries and 1240 cached graphs actually loaded; distinct from the 2904-complex generator split.",
        },
        {
            "stage": "mdn_scorer_transfer",
            "role": "validation",
            "component": "na_clean_v5",
            "complex_instances": 24,
            "unique_pdb_receptors": 14,
            "rna": 0, "dna": 0, "rna_dna": 0, "unknown_type": 24,
            "source": "SurfNA v5 frozen split/cache",
            "note": f"{mdn_val_nominal} nominal entries and 24 cached graphs actually loaded.",
        },
    ]
    for benchmark in BENCHMARKS:
        rows = [row for row in benchmark_manifest if row["benchmark"] == benchmark]
        audit = benchmark_audit[benchmark]
        summary.append(
            summary_row(
                "final_evaluation", "test", benchmark, rows, "Jiang benchmark",
                f"Mapped usable {audit['mapped_usable_n']} of {audit['old_n']}; "
                f"excluded {audit['excluded_n']} unavailable/unusable entries.",
            )
        )
    summary.append(
        summary_row(
            "final_evaluation", "test", "jiang_union", benchmark_manifest,
            "Jiang benchmarks",
            "220 component-set rows over 132 unique PDB receptors; component sets overlap.",
        )
    )
    summary_fields = [
        "stage", "role", "component", "complex_instances", "unique_pdb_receptors",
        "rna", "dna", "rna_dna", "unknown_type", "source", "note",
    ]
    write_csv(args.out_dir / "dataset_audit_summary.csv", summary, summary_fields)

    provenance = [
        provenance_row(
            "protein_surface_pretraining",
            "released generator ancestor",
            "protein_surfdock4_probe15_parse_fixmesh1_continue_bestinf_4gpuada_surfmax1500_bs4_lr2e-4_2026_05_19_20_29_18",
            PROTEIN_PRETRAIN_SHA256,
            "protein_train_surfdock_fixmesh1",
            protein_train_nominal,
            10294,
            "protein_val_surfdock_fixmesh1",
            protein_val_nominal,
            569,
            219,
            "LogFile.log loader messages and final best-inference record",
            "The public generator descends from this checkpoint through the 2026-05-23 continuation run.",
        ),
        provenance_row(
            "nucleic_acid_generator",
            "public release and main-table generator",
            "na_surfaceonly_latestprotein_fixmesh1_expanded_highconf_noesm_4gpu_surfmax1500_bs4_lr5e-4_2026_05_24_00_33_03",
            RELEASE_GENERATOR_SHA256,
            "na_expanded_v1_highconf_trainval_tierA_v6plus",
            len(train_manifest),
            2904,
            "na_expanded_v1_highconf_test132_fixed",
            len(development_manifest),
            128,
            899,
            "Checkpoint SHA-256 identity plus LogFile.log loader and best-inference records",
            "This count describes the released generator training run, not every earlier development experiment.",
        ),
        provenance_row(
            "mdn_scorer",
            "public release and main-table scorer",
            "na_mdn4_officiallike_transfer_noesm_robust_lr1e4_effbs64_fixmesh1_2026_05_21_21_32_352026-05-22_01-16-15",
            RELEASE_MDN_SHA256,
            "na_clean_train_no_leak_blacklist_v5_fixmesh1_ddp8",
            mdn_train_nominal,
            1240,
            "na_clean_val_no_leak_blacklist_v5_fixmesh1_ddp8",
            mdn_val_nominal,
            24,
            96,
            "Checkpoint SHA-256 identity, cached-list lengths, and final best-validation record",
            "The MDN used a smaller v5 split; it did not train on the 2904-complex generator split.",
        ),
    ]
    provenance_fields = [
        "component", "checkpoint_role", "run_id", "checkpoint_sha256",
        "train_split", "nominal_train", "loaded_train", "val_split",
        "nominal_val", "loaded_val", "best_epoch", "evidence", "note",
    ]
    write_csv(args.out_dir / "model_provenance.csv", provenance, provenance_fields)

    train_ids = {str(row["complex_id"]) for row in train_manifest}
    train_pdb = {str(row["pdb_id"]) for row in train_manifest}
    train_sequence = {str(row["receptor_sequence_sha256"]) for row in train_manifest if row["receptor_sequence_sha256"]}
    dev_ids = {str(row["complex_id"]) for row in development_manifest}
    dev_pdb = {str(row["pdb_id"]) for row in development_manifest}
    dev_sequence = {str(row["receptor_sequence_sha256"]) for row in development_manifest if row["receptor_sequence_sha256"]}
    bench_ids = {str(row["complex_id"]) for row in benchmark_manifest}
    bench_pdb = {str(row["pdb_id"]) for row in benchmark_manifest}
    bench_sequence = {str(row["receptor_sequence_sha256"]) for row in benchmark_manifest if row["receptor_sequence_sha256"]}
    mdn_train_rows = [row for row in mdn_manifest if row["role"] == "train"]
    mdn_train_ids = {str(row["complex_id"]) for row in mdn_train_rows}
    mdn_train_pdb = {str(row["pdb_id"]) for row in mdn_train_rows}
    mdn_train_sequence = {
        str(row["receptor_sequence_sha256"])
        for row in mdn_train_rows
        if row["receptor_sequence_sha256"]
    }

    def audit_row(check: str, left: set[str], right: set[str], definition: str) -> dict[str, object]:
        overlap = sorted(left & right)
        return {
            "check": check,
            "left_count": len(left),
            "right_count": len(right),
            "overlap_count": len(overlap),
            "status": "PASS" if not overlap else "REVIEW",
            "definition": definition,
            "overlap_examples": ";".join(overlap[:10]),
        }

    leakage = [
        audit_row("trainval_vs_development_exact_complex", train_ids, dev_ids, "Exact complex identifier"),
        audit_row("trainval_vs_development_pdb_receptor", train_pdb, dev_pdb, "PDB accession/base receptor"),
        audit_row(
            "trainval_vs_development_exact_receptor_sequence",
            train_sequence,
            dev_sequence,
            "SHA-256 of sorted canonical NA chain sequences; exact full-receptor match only",
        ),
        audit_row("trainval_vs_jiang_exact_complex", train_ids, bench_ids, "Exact complex identifier"),
        audit_row("trainval_vs_jiang_pdb_receptor", train_pdb, bench_pdb, "PDB accession/base receptor"),
        audit_row(
            "trainval_vs_jiang_exact_receptor_sequence",
            train_sequence,
            bench_sequence,
            "SHA-256 of sorted canonical NA chain sequences; exact full-receptor match only",
        ),
        audit_row("development_vs_jiang_exact_complex", dev_ids, bench_ids, "Exact complex identifier"),
        audit_row("development_vs_jiang_pdb_receptor", dev_pdb, bench_pdb, "PDB accession/base receptor"),
        audit_row(
            "development_vs_jiang_exact_receptor_sequence",
            dev_sequence,
            bench_sequence,
            "SHA-256 of sorted canonical NA chain sequences; exact full-receptor match only",
        ),
        audit_row("mdn_train_vs_development_exact_complex", mdn_train_ids, dev_ids, "Exact complex identifier"),
        audit_row("mdn_train_vs_development_pdb_receptor", mdn_train_pdb, dev_pdb, "PDB accession/base receptor"),
        audit_row(
            "mdn_train_vs_development_exact_receptor_sequence",
            mdn_train_sequence,
            dev_sequence,
            "SHA-256 of sorted canonical NA chain sequences; exact full-receptor match only",
        ),
    ]
    leakage_fields = [
        "check", "left_count", "right_count", "overlap_count", "status",
        "definition", "overlap_examples",
    ]
    write_csv(args.out_dir / "leakage_audit.csv", leakage, leakage_fields)

    train_by_sequence: dict[str, list[dict[str, object]]] = {}
    development_by_sequence: dict[str, list[dict[str, object]]] = {}
    benchmark_by_sequence: dict[str, list[dict[str, object]]] = {}
    mdn_train_by_sequence: dict[str, list[dict[str, object]]] = {}
    for row in train_manifest:
        digest = str(row["receptor_sequence_sha256"])
        if digest:
            train_by_sequence.setdefault(digest, []).append(row)
    for row in development_manifest:
        digest = str(row["receptor_sequence_sha256"])
        if digest:
            development_by_sequence.setdefault(digest, []).append(row)
    for row in benchmark_manifest:
        digest = str(row["receptor_sequence_sha256"])
        if digest:
            benchmark_by_sequence.setdefault(digest, []).append(row)
    for row in mdn_train_rows:
        digest = str(row["receptor_sequence_sha256"])
        if digest:
            mdn_train_by_sequence.setdefault(digest, []).append(row)
    sequence_overlap_rows: list[dict[str, object]] = []
    for digest in sorted(set(train_by_sequence) & set(development_by_sequence)):
        train_group = train_by_sequence[digest]
        development_group = development_by_sequence[digest]
        benchmark_group = benchmark_by_sequence.get(digest, [])
        sequence_overlap_rows.append(
            {
                "receptor_sequence_sha256": digest,
                "canonical_na_residue_count": development_group[0]["canonical_na_residue_count"],
                "na_chain_count": development_group[0]["na_chain_count"],
                "trainval_complex_count": len(train_group),
                "trainval_complex_ids": ";".join(str(row["complex_id"]) for row in train_group),
                "development_complex_count": len(development_group),
                "development_complex_ids": ";".join(str(row["complex_id"]) for row in development_group),
                "jiang_component_row_count": len(benchmark_group),
                "jiang_components": ";".join(
                    sorted({str(row["benchmark"]) for row in benchmark_group})
                ),
                "jiang_complex_ids": ";".join(str(row["complex_id"]) for row in benchmark_group),
            }
        )
    sequence_overlap_fields = [
        "receptor_sequence_sha256", "canonical_na_residue_count", "na_chain_count",
        "trainval_complex_count", "trainval_complex_ids", "development_complex_count",
        "development_complex_ids", "jiang_component_row_count", "jiang_components",
        "jiang_complex_ids",
    ]
    write_csv(
        args.out_dir / "exact_sequence_overlap_details.csv",
        sequence_overlap_rows,
        sequence_overlap_fields,
    )

    mdn_sequence_overlap_rows: list[dict[str, object]] = []
    for digest in sorted(set(mdn_train_by_sequence) & set(development_by_sequence)):
        mdn_group = mdn_train_by_sequence[digest]
        development_group = development_by_sequence[digest]
        benchmark_group = benchmark_by_sequence.get(digest, [])
        mdn_sequence_overlap_rows.append(
            {
                "receptor_sequence_sha256": digest,
                "canonical_na_residue_count": development_group[0]["canonical_na_residue_count"],
                "na_chain_count": development_group[0]["na_chain_count"],
                "mdn_train_complex_count": len(mdn_group),
                "mdn_train_complex_ids": ";".join(str(row["complex_id"]) for row in mdn_group),
                "development_complex_count": len(development_group),
                "development_complex_ids": ";".join(str(row["complex_id"]) for row in development_group),
                "jiang_component_row_count": len(benchmark_group),
                "jiang_components": ";".join(
                    sorted({str(row["benchmark"]) for row in benchmark_group})
                ),
                "jiang_complex_ids": ";".join(str(row["complex_id"]) for row in benchmark_group),
            }
        )
    mdn_sequence_overlap_fields = [
        "receptor_sequence_sha256", "canonical_na_residue_count", "na_chain_count",
        "mdn_train_complex_count", "mdn_train_complex_ids", "development_complex_count",
        "development_complex_ids", "jiang_component_row_count", "jiang_components",
        "jiang_complex_ids",
    ]
    write_csv(
        args.out_dir / "mdn_exact_sequence_overlap_details.csv",
        mdn_sequence_overlap_rows,
        mdn_sequence_overlap_fields,
    )

    missing_train = sum(not row["receptor_sequence_sha256"] for row in train_manifest)
    missing_dev = sum(not row["receptor_sequence_sha256"] for row in development_manifest)
    readme = f"""# SurfNA dataset and model-provenance audit

These files were generated from the frozen SurfNA research workspace by
`scripts/build_dataset_audit.py`.

## Headline counts

- Protein-surface pretraining: {protein_train_nominal} nominal training entries, of which 10,294 cached graphs were actually loaded.
- Released nucleic-acid generator: {len(train_manifest)} nominal entries and 2,904 graphs actually loaded ({len(original_rows)} original + {len(expanded_rows)} independently PDB-mined high-confidence complexes).
- Released MDN scorer: {mdn_train_nominal} nominal entries and 1,240 cached graphs actually loaded from a separate v5 split.
- Fixed generator development set: {len(development_manifest)} nominal receptors, of which 128 graphs were actually loaded.
- Final Jiang evaluation: {len(benchmark_manifest)} component-set rows over {len(bench_pdb)} unique PDB receptors.

`model_provenance.csv` ties these counts to the exact public checkpoint hashes,
run identifiers, split names, loader/cache evidence, and best epochs. The 2,904
count applies specifically to the released generator run and must not be read
as the size of every SurfNA development experiment or of the MDN training set.
`mdn_trainval_manifest.csv` separately lists the 1,240 MDN training instances
and 24 validation instances; these correspond to 868 and 14 unique PDB
receptors, respectively.

## Leakage checks

`leakage_audit.csv` tests exact complex identifiers, PDB/base-receptor identity,
and exact full-receptor canonical nucleic-acid sequence signatures. The released
generator training split has zero exact complex-ID and PDB-receptor overlap with
the Jiang receptors. Its sequence audit finds {len(sequence_overlap_rows)} shared
exact signatures and is marked `REVIEW`; corresponding receptors are listed in
`exact_sequence_overlap_details.csv`.

The fixed development set used for generator checkpoint selection contains the
same {len(dev_pdb)} PDB receptor accessions as the Jiang component-set union
({len(dev_pdb & bench_pdb)} shared). This is model-selection leakage even though
the structures were not used for gradient fitting. A strictly independent final
benchmark requires checkpoint selection on a receptor-disjoint validation set
and regeneration of the reported Jiang results.

The released MDN split is audited separately. It has zero exact complex-ID and
PDB-accession overlap with the development set, but {len(mdn_sequence_overlap_rows)}
shared exact sequence signatures; those cases are listed in
`mdn_exact_sequence_overlap_details.csv`.

Sequence signatures are chain-order-independent SHA-256 digests. They detect
exact canonical sequence reuse but are **not** a homology-threshold or
structural-similarity audit, and modified nucleotides are ignored. Canonical
sequence extraction was unavailable for {missing_train} train/validation and
{missing_dev} development receptors.

The development list represents the same 132 receptor accessions underlying
the four Jiang component sets. The component sets contain overlapping receptors
and together contribute 220 evaluation rows.

No coordinate files or private cluster paths are included in the public
manifests.
"""
    (args.out_dir / "README.md").write_text(readme)
    print(f"Wrote SurfNA dataset audit to {args.out_dir}")


if __name__ == "__main__":
    main()
