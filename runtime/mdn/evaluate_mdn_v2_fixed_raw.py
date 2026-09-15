#!/usr/bin/env python3
"""Score one frozen raw K-pose manifest with a frozen SurfNA V2 scorer."""

from __future__ import annotations

import argparse
import csv
import json
from functools import partial
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import set_seed

from datasets.group_sampler import DistributedGroupBatchSampler
from datasets.mdn_rerank_v2 import DecoyPDBBindV2
from train_mdn_v2_accelerate import (
    checkpoint_state,
    dataset_kwargs,
    evaluate,
    make_loader,
    sha256_file,
    validate_gate,
)
from utils.diffusion_utils import t_to_sigma as t_to_sigma_compl
from utils.scorer_config_v2 import load_scorer_model_args


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-config", required=True)
    parser.add_argument(
        "--evaluation-role",
        choices=("final_test", "validation_diagnostic"),
        default="final_test",
    )
    parser.add_argument(
        "--g22-source-root",
        help=(
            "Exact SurfNA_V2_G22/src used by full_v2_b. The matching "
            "SURFNA_V2_G22_SOURCE_ROOT environment variable must be exported "
            "before Python starts so the G2.2 backbone is imported."
        ),
    )
    parser.add_argument(
        "--scorer-variant",
        choices=(
            "aligned_a",
            "full_v2_b",
            "distribution_s1",
            "gated_na_s2",
        ),
        required=True,
    )
    parser.add_argument("--scorer-checkpoint", required=True)
    parser.add_argument("--expected-scorer-sha256", required=True)
    parser.add_argument("--expected-scorer-epoch", type=int, required=True)
    parser.add_argument("--mdn-reference-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--manifest-meta", required=True)
    parser.add_argument("--gate-report", required=True)
    parser.add_argument("--test-split", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--cache-path", required=True)
    parser.add_argument("--surface-path", required=True)
    parser.add_argument("--surface-scaler-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pose-microbatch-size", type=int, default=4)
    parser.add_argument("--loader-workers", type=int, default=0)
    parser.add_argument("--dataset-workers", type=int, default=2)
    parser.add_argument("--max-surface-vertices", type=int, default=512)
    parser.add_argument("--per-complex-timeout-sec", type=int, default=180)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--remove-hs", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    accelerator = Accelerator()
    set_seed(args.seed, device_specific=False)
    gate = validate_gate(args.gate_report, args.manifest, allow_nonformal=False)
    meta = json.loads(Path(args.manifest_meta).read_text())
    if args.evaluation_role == "final_test":
        if meta.get("final_eval_authorized") is not True or meta.get("split") != "test":
            raise ValueError("manifest is not an authorized frozen final-eval manifest")
        if not isinstance(meta.get("test_guided_generator"), bool):
            raise ValueError(
                "frozen final-eval metadata must explicitly declare test_guided_generator"
            )
    else:
        if meta.get("split") != "val":
            raise ValueError("validation diagnostic requires a formal split=val manifest")
        if meta.get("final_eval_authorized") is True:
            raise ValueError("validation diagnostic cannot consume a final-test authorization")

    scorer_sha = sha256_file(args.scorer_checkpoint)
    if scorer_sha != args.expected_scorer_sha256:
        raise ValueError(f"scorer checkpoint SHA mismatch: {scorer_sha}")
    checkpoint = torch.load(args.scorer_checkpoint, map_location="cpu")
    if not isinstance(checkpoint, dict) or int(checkpoint.get("epoch", -1)) != args.expected_scorer_epoch:
        raise ValueError(
            f"scorer checkpoint epoch mismatch: expected={args.expected_scorer_epoch}, "
            f"got={checkpoint.get('epoch') if isinstance(checkpoint, dict) else None}"
        )

    model_args = load_scorer_model_args(args.model_config)
    model_args.scorer_variant = args.scorer_variant
    model_args.g22_source_root = args.g22_source_root
    model_args.mdn_reference_json = args.mdn_reference_json
    model_args.allow_unfitted_mdn_reference = False
    t_to_sigma = partial(t_to_sigma_compl, args=model_args)
    from models.surface_interaction_backbone_v2 import BACKBONE_SOURCE_FILE
    from utils.scorer_factory_v2 import build_surfna_v2_scorer

    model = build_surfna_v2_scorer(model_args, accelerator.device, t_to_sigma)
    model.load_state_dict(checkpoint_state(checkpoint), strict=True)

    kwargs = dataset_kwargs(args, model_args)
    dataset = DecoyPDBBindV2(
        args.manifest,
        args.test_split,
        gate_report_path=args.gate_report,
        expected_profile="raw",
        **kwargs,
    )
    sampler = DistributedGroupBatchSampler(
        dataset.pose_group_uids,
        groups_per_batch=1,
        shuffle=False,
        seed=args.seed,
        num_replicas=accelerator.num_processes,
        rank=accelerator.process_index,
        drop_rank_tail=False,
    )
    loader = make_loader(dataset, sampler, args.loader_workers)
    model = accelerator.prepare(model)
    metrics, records = evaluate(model, loader, dataset, accelerator, args)

    if accelerator.is_main_process:
        pose_meta = {}
        with open(args.manifest, newline="") as handle:
            for row in csv.DictReader(handle, delimiter="\t"):
                pose_meta[row["pose_uid"]] = row
        output = Path(args.output_dir)
        output.mkdir(parents=True, exist_ok=True)
        fields = (
            "complex_name",
            "pose_group_uid",
            "pose_uid",
            "candidate_rank_input",
            "rmsd",
            "score",
            "prior_score",
            "cross_score",
            "na_score",
            "clash_penalty",
            "strain_penalty",
            "ordinal_logit_lt5",
            "ordinal_logit_lt2_given_lt5",
            "prob_lt2",
            "prob_lt5",
            "coverage_score",
            "na_gate",
            "pose_sdf",
        )
        with (output / "pose_scores.tsv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
            writer.writeheader()
            for record in sorted(
                records,
                key=lambda item: (
                    pose_meta[item["pose_uid"]]["complex_name"],
                    item["candidate_rank_input"],
                ),
            ):
                row = pose_meta[record["pose_uid"]]
                writer.writerow(
                    {
                        "complex_name": row["complex_name"],
                        "pose_group_uid": record["pose_group_uid"],
                        "pose_uid": record["pose_uid"],
                        "candidate_rank_input": record["candidate_rank_input"],
                        "rmsd": record["rmsd"],
                        "score": record["score"],
                        "prior_score": record["prior_score"],
                        "cross_score": record["cross_score"],
                        "na_score": record["na_score"],
                        "clash_penalty": record["clash_penalty"],
                        "strain_penalty": record["strain_penalty"],
                        "ordinal_logit_lt5": record["ordinal_logit_lt5"],
                        "ordinal_logit_lt2_given_lt5": record[
                            "ordinal_logit_lt2_given_lt5"
                        ],
                        "prob_lt2": record["prob_lt2"],
                        "prob_lt5": record["prob_lt5"],
                        "coverage_score": record["coverage_score"],
                        "na_gate": record["na_gate"],
                        "pose_sdf": row["pose_sdf"],
                    }
                )
        report = {
            "status": "complete",
            "evaluation_role": args.evaluation_role,
            "metrics": metrics,
            "scorer_checkpoint": str(Path(args.scorer_checkpoint).resolve()),
            "scorer_checkpoint_sha256": scorer_sha,
            "scorer_epoch": args.expected_scorer_epoch,
            "scorer_variant": args.scorer_variant,
            "g22_source_root": args.g22_source_root,
            "backbone_source_file": BACKBONE_SOURCE_FILE,
            "generator_checkpoint": meta.get("generator_checkpoint"),
            "generator_checkpoint_sha256": meta.get("generator_checkpoint_sha256"),
            "manifest": str(Path(args.manifest).resolve()),
            "manifest_sha256": sha256_file(args.manifest),
            "gate_report": gate,
            "benchmark_name": meta.get(
                "benchmark_name",
                "strict_validation" if args.evaluation_role == "validation_diagnostic" else "fixed132",
            ),
            "test_guided_generator": meta.get("test_guided_generator"),
            "end_to_end_identity_strict_validation": (
                meta.get("end_to_end_identity_strict_validation", False)
                if args.evaluation_role == "final_test"
                else False
            ),
            "unseen_heldout_claim_allowed": bool(
                args.evaluation_role == "final_test"
                and meta.get("end_to_end_identity_strict_validation", False)
                and not meta.get("test_guided_generator", True)
            ),
            "benchmark_canonical_count": meta.get(
                "benchmark_canonical_count", meta.get("fixed132_canonical_count")
            ),
            "benchmark_graph_ready_count": meta.get(
                "benchmark_graph_ready_count", meta.get("fixed132_graph_ready_count")
            ),
            "benchmark_missing_identities": meta.get(
                "benchmark_missing_identities", meta.get("fixed132_missing_identities", [])
            ),
            "frozen_test_overlap_audit": meta.get("frozen_test_overlap_audit", {}),
            "sampler": sampler.audit(),
        }
        for key in (
            "fixed132_canonical_count",
            "fixed132_graph_ready_count",
            "fixed132_missing_identities",
        ):
            if key in meta:
                report[key] = meta[key]
        (output / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        (output / "READY").touch()
        print(json.dumps(report, sort_keys=True), flush=True)
    accelerator.wait_for_everyone()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
