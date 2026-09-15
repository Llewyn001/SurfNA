"""Fresh L2 candidate-only MDN evidence; check-only unless --execute is supplied."""
from __future__ import annotations

import argparse
from collections import Counter
import os
from pathlib import Path
import time
import traceback

from rank_common import (COUNTS, atomic_json, atomic_torch, check_hash, event, key,
                         load_api, reserve_stage, sha, validate_contract)
from rank_math import dense_nearest_score, finite, ranking_metrics, state_digest


def candidate_graph(template, coordinates, clean_graph):
    """Replace the native position BEFORE removing metadata or model encoding."""
    import torch
    graph = template.clone()
    coordinates = torch.as_tensor(coordinates, dtype=torch.float32).clone()
    center = torch.as_tensor(graph.original_center).detach().cpu().reshape(1, 3)
    if tuple(coordinates.shape) != tuple(graph['ligand'].pos.shape):
        raise RuntimeError("Candidate and chemical graph atom counts differ")
    finite(coordinates, center)
    graph['ligand'].pos = coordinates - center
    return clean_graph(graph)


def evidence(model, graph, surface, surface_pos, log_probability, device):
    """Already-sanitized candidate geometry only: no labels, paths, IDs, native coordinates."""
    import torch
    from torch_geometric.data import Batch
    batch = Batch.from_data_list([graph]).to(device)
    # Deliberately called once for EACH of the 7120 candidates, never native-cache reused.
    encoded = model.backbone.encode_ligand_intra(batch)
    ligand = encoded.scalar[0]
    if not bool(encoded.mask[0].all()) or len(ligand) != len(batch['ligand'].pos):
        raise RuntimeError("Unexpected ligand padding/atom count for a single candidate")
    distances_all = torch.cdist(batch['ligand'].pos.float(), surface_pos.float())
    finite(ligand, surface, distances_all)
    if len(surface) < 8:
        raise RuntimeError("Surface too small for unchanged nearest8 protocol")
    distances, indices = distances_all.topk(8, largest=False, sorted=True, dim=-1)
    pair_features = torch.cat([ligand[:, None, :].expand(-1, 8, -1), surface[indices]], -1)
    pi, mu, sigma = model.prior_head.predict_parameters(pair_features)
    finite(pi, mu, sigma, distances)
    lp = log_probability(pi, sigma, mu, distances)
    expected = (pi * mu).sum(-1)
    variance = (pi * (sigma.square() + mu.square())).sum(-1) - expected.square()
    if float(variance.min()) < -1e-3:
        raise RuntimeError("Negative MDN predictive variance beyond roundoff")
    tokens = torch.cat([distances, lp, expected, variance.clamp_min(0).sqrt()], -1)
    baseline = lp.mean()
    finite(tokens, baseline)
    return tokens.cpu(), baseline.cpu(), (ligand, distances_all)


def run(args):
    root = Path(args.root).resolve(strict=True)
    contract, groups, rows = validate_contract(root)
    if (root / 'rank/features').exists():
        raise RuntimeError("Feature attempt already exists; no overwrite or automatic retry")
    if not args.execute:
        event('rank_features_check_only', groups=890, poses=7120, device_not_initialized=True)
        return
    out = reserve_stage(root, 'features')
    current, ordinal = None, None
    start = time.monotonic()
    try:
        for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            os.environ[name] = '1'
        import torch
        from torch_geometric.data import Batch
        torch.set_num_threads(2)
        if args.device == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is not available; no silent CPU fallback')
        native_api, pose_api = load_api(contract, 'native'), load_api(contract, 'pose')
        model = native_api.load_trained_model(root, device=args.device)
        if model.training or any(p.requires_grad for p in model.parameters()):
            raise RuntimeError('Every source model parameter must be frozen in eval mode')
        from models.surfna_v2_scorer_components import mixture_log_probability
        before = state_digest(model.state_dict())
        head_before = state_digest(model.prior_head.state_dict())
        backbone_before = state_digest(model.backbone.state_dict())
        dest = out / 'cache'
        dest.mkdir(exist_ok=False)
        index, scored = [], []
        surface_calls = ligand_calls = 0
        dense_error = None
        with torch.no_grad():
            for gi, group in enumerate(groups):
                current, ordinal = group['name'], None
                row = rows[current]
                check_hash(row['path'], row['artifact_sha256'])
                native = torch.load(row['path'], map_location='cpu', weights_only=False)
                if native['name'] != current or native['split'] != group['split']:
                    raise RuntimeError('Native graph identity/split alignment failed')
                poses = sorted(group['poses'], key=lambda pose: pose['ordinal'])
                tokens, baselines = [], []
                surface, surface_pos = None, None
                for pose in poses:
                    ordinal = pose['ordinal']
                    check_hash(pose['pose_path'], pose['pose_file_sha256'])
                    molecule = pose_api.read_molecule(pose['pose_path'])
                    if (pose_api.atom_order_signature(molecule) != pose['atom_order_sha256']
                            or molecule.GetNumAtoms() != row['ligand_atoms']):
                        raise RuntimeError('SDF ordered heavy-atom signature/graph alignment failed')
                    # Labels and identity stay outside these two numerical APIs.
                    graph = candidate_graph(native['graph'], molecule.GetConformer().GetPositions(), native_api.clean_graph)
                    if surface is None:
                        encoded = model.backbone.encode_surface_static(Batch.from_data_list([graph]).to(args.device))
                        if not bool(encoded.mask[0].all()):
                            raise RuntimeError('Unexpected padding in single-target surface encoder')
                        surface, surface_pos = encoded.scalar[0], encoded.position[0]
                        finite(surface, surface_pos)
                        if len(surface) != row['surface_vertices'] or not 8 <= len(surface) <= 512:
                            raise RuntimeError('Surface coverage differs from the prepared graph')
                        surface_calls += 1
                    x, baseline, (ligand, distances) = evidence(model, graph, surface, surface_pos,
                                                              mixture_log_probability, args.device)
                    ligand_calls += 1
                    tokens.append(x)
                    baselines.append(baseline)
                    if gi == 0 and ordinal == 0:
                        dense_error = abs(dense_nearest_score(model.prior_head, ligand, surface, distances,
                                                              mixture_log_probability) - float(baseline))
                        if dense_error > contract['baseline_absolute_tolerance']:
                            raise RuntimeError('Nearest-only vs dense MDN parity failed')
                    # RMSD is attached only after numerical scoring has returned.
                    scored.append(dict(name=current, split=group['split'], ordinal=ordinal,
                                       pose_uid=pose['pose_uid'], rmsd=pose['rmsd'], baseline=float(baseline)))
                path = dest / (key(current) + '.pt')
                if path.exists():
                    raise RuntimeError('Feature filename collision; no overwrite')
                atomic_torch(path, dict(name=current, split=group['split'], tokens=torch.stack(tokens),
                    baseline=torch.stack(baselines), pose_uids=[pose['pose_uid'] for pose in poses],
                    rank_contract_sha256=sha(root / 'rank_contract.json'),
                    native_head_sha256=contract['native_head_sha256'],
                    source_bundle_sha256=contract['native_static_bundle_sha256']))
                index.append(dict(name=current, split=group['split'], path=str(path), sha256=sha(path)))
                if (gi + 1) % 25 == 0 or gi + 1 == len(groups):
                    elapsed = time.monotonic() - start
                    event('rank_candidate_features', groups=gi+1, total=len(groups), poses=ligand_calls,
                          seconds=elapsed, estimated_remaining_seconds=elapsed/(gi+1)*(len(groups)-gi-1))
        if (surface_calls != 890 or ligand_calls != 7120 or Counter(row['split'] for row in index) != COUNTS
                or len(scored) != 7120 or dense_error is None):
            raise RuntimeError('Incomplete uncurated feature population')
        if state_digest(model.state_dict()) != before:
            raise RuntimeError('Frozen source model changed during feature extraction')
        # Recheck all frozen source inputs before publishing a ready marker.
        validate_contract(root)
        atomic_json(out / 'feature_index.json', index, new=True)
        atomic_json(out / 'baseline_metrics.json', {s: ranking_metrics([r for r in scored if r['split'] == s], 'baseline')
                                                  for s in COUNTS}, new=True)
        atomic_json(out / 'FEATURES_READY.json', dict(status='PASS', groups=890, poses=7120, split_counts=COUNTS,
            rank_contract_sha256=sha(root/'rank_contract.json'), root_contract_sha256=sha(root/'contract.json'),
            source_bundle_sha256=contract['native_static_bundle_sha256'], source_head_sha256=contract['native_head_sha256'],
            frozen_model_state_sha256=before, frozen_head_state_sha256=head_before,
            frozen_backbone_state_sha256=backbone_before, source_model_unchanged=True,
            feature_index_sha256=sha(out/'feature_index.json'), baseline_metrics_sha256=sha(out/'baseline_metrics.json'),
            dense_score_parity_max_abs=dense_error, candidate_ligand_encoder_calls=ligand_calls,
            surface_encoder_calls=surface_calls, old_V3_cache_used=False, dimensions=32,
            labels_in_feature_cache=False, test_opened=False, seconds=time.monotonic()-start), new=True)
        event('rank_features_ready', groups=890, poses=7120, seconds=time.monotonic()-start)
    except Exception:
        atomic_json(out/'FAILED.json', dict(error=traceback.format_exc(), member=current, ordinal=ordinal,
                    time=time.time(), automatic_retry=False, test_opened=False), new=True)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True)
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cuda')
    parser.add_argument('--execute', action='store_true')
    run(parser.parse_args())
