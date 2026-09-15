"""Factory for the SurfNA V2 dual-branch scorer."""

from __future__ import annotations

import copy
from pathlib import Path

from models.surface_interaction_backbone_v2 import (
    EQUIVARIANT_GRAPH_RMS_CAP,
    SurfaceInteractionBackboneFromDiffusion,
)
from models.surface_interaction_backbone_v2 import BACKBONE_SOURCE_FILE
from models.surfna_v2_scorer import SurfNAV2Scorer
from models.surfna_v2_scorer_components import DistanceReferencePrior
from utils.diffusion_utils import get_timestep_embedding


def build_surfna_v2_scorer(args, device, t_to_sigma):
    if getattr(args, "surface_feature_schema", None) != "v2_full8":
        raise ValueError("SurfNA V2 scorer requires surface_feature_schema=v2_full8")
    if int(getattr(args, "surface_feature_dim", 0)) != 8:
        raise ValueError("SurfNA V2 scorer requires surface_feature_dim=8")
    if int(args.num_conv_layers) != 6 or int(args.ns) != 48 or int(args.nv) != 10:
        raise ValueError("protein-transfer scorer contract requires layers=6, ns=48, nv=10")
    scorer_variant = getattr(args, "scorer_variant", "aligned_a")
    if scorer_variant not in {
        "aligned_a",
        "full_v2_b",
        "distribution_s1",
        "gated_na_s2",
    }:
        raise ValueError(f"unknown scorer_variant={scorer_variant!r}")
    protein_surface_variants = {"aligned_a", "distribution_s1", "gated_na_s2"}
    if scorer_variant in protein_surface_variants and (
        bool(getattr(args, "use_nusurf_fusion", False))
        or bool(getattr(args, "use_nucleic_feat_fusion", False))
    ):
        raise ValueError(
            f"{scorer_variant} must use the shared protein config without "
            "NuSurf/nucleic backbone replacement"
        )
    if scorer_variant == "full_v2_b" and not (
        bool(getattr(args, "use_nusurf_fusion", False))
        and bool(getattr(args, "use_nucleic_feat_fusion", False))
    ):
        raise ValueError(
            "full_v2_b must be instantiated from the G2.2 NA config with both "
            "use_nusurf_fusion and use_nucleic_feat_fusion enabled"
        )
    if scorer_variant == "full_v2_b":
        expected_g22_root = getattr(args, "g22_source_root", None)
        if not expected_g22_root:
            raise ValueError("full_v2_b requires explicit g22_source_root provenance")
        expected_source = Path(expected_g22_root).resolve() / "models" / "surface_score_model_v3.py"
        if Path(BACKBONE_SOURCE_FILE).resolve() != expected_source:
            raise ValueError(
                "full_v2_b imported the wrong backbone source; set "
                "SURFNA_V2_G22_SOURCE_ROOT before Python starts. "
                f"expected={expected_source} actual={BACKBONE_SOURCE_FILE}"
            )

    # Diffusion-only native-contact/pretraining heads are neither called nor
    # transferred by the scorer.  Disable their construction explicitly while
    # retaining every shared static, cross and NuSurf module from the source
    # configuration/checkpoint.
    backbone_args = copy.copy(args)
    backbone_args.task_aligned_aux = False
    backbone_args.pretrain_v2_mode = False
    timestep_embedding = get_timestep_embedding(
        embedding_type=args.embedding_type,
        embedding_dim=args.sigma_embed_dim,
        embedding_scale=args.embedding_scale,
    )
    lm_embedding_type = None
    if getattr(args, "esm_embeddings_path", None) is not None:
        lm_embedding_type = getattr(args, "lm_embedding_type", "auto")
        if lm_embedding_type == "auto":
            lm_embedding_type = "esm"
    backbone = SurfaceInteractionBackboneFromDiffusion(
        t_to_sigma=t_to_sigma,
        device=device,
        no_torsion=args.no_torsion,
        timestep_emb_func=timestep_embedding,
        num_conv_layers=args.num_conv_layers,
        lig_max_radius=args.max_radius,
        scale_by_sigma=args.scale_by_sigma,
        sigma_embed_dim=args.sigma_embed_dim,
        ns=args.ns,
        nv=args.nv,
        distance_embed_dim=args.distance_embed_dim,
        cross_distance_embed_dim=args.cross_distance_embed_dim,
        batch_norm=not args.no_batch_norm,
        dropout=args.dropout,
        use_second_order_repr=args.use_second_order_repr,
        cross_max_distance=args.cross_max_distance,
        dynamic_max_cross=False,
        lm_embedding_type=lm_embedding_type,
        args=backbone_args,
        confidence_mode=False,
        scorer_cross_distance=getattr(args, "scorer_cross_distance", 12.0),
        scorer_clash_distance=getattr(args, "scorer_clash_distance", 2.0),
        scorer_equivariant_rms_cap=EQUIVARIANT_GRAPH_RMS_CAP,
    )
    reference_json = getattr(args, "mdn_reference_json", None)
    if reference_json:
        reference_prior = DistanceReferencePrior.from_json(reference_json)
    elif getattr(args, "allow_unfitted_mdn_reference", False):
        reference_prior = DistanceReferencePrior()
    else:
        raise ValueError(
            "formal SurfNA V2 scorer construction requires --mdn_reference_json fitted on train"
        )
    scorer = SurfNAV2Scorer.from_diffusion_backbone(
        backbone,
        hidden_dim=getattr(args, "scorer_hidden_dim", 192),
        n_gaussians=getattr(args, "n_gaussians", 20),
        topk_surface_per_atom=getattr(args, "mdn_score_topk_per_atom", 8),
        dropout=getattr(args, "mdn_dropout", 0.1),
        reference_prior=reference_prior,
        scorer_variant=scorer_variant,
    )
    return scorer.to(device)
