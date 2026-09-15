"""Nearest-eight static MDN evidence."""
from .scorer_common import finite

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
    # Encode each candidate in its own coordinates.
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

def clean_graph(graph):
    graph = graph.clone()
    forbidden = {"orig_pos", "original_center", "original_ligand_center", "rmsd", "rmsd_matching",
                 "name", "complex_id", "parent_pdb", "pdb_id", "split", "split_group", "pose_uid",
                 "pose_path", "native_path", "native_global_coordinates", "candidate_rank_input",
                 "ordinal", "atoms_pos"}
    for store in graph.stores:
        for field in list(store.keys()):
            if field in forbidden or field.startswith(("reference_", "native_", "label_", "chemistry_", "coordinate_frame")):
                del store[field]
    return graph
