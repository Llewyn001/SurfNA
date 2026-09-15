"""Candidate interaction features from the frozen six-layer Generator."""
from collections.abc import Mapping
import hashlib,re,sys,json

EDGE_WIDTH, TOKEN_WIDTH, STATE_KEYS = 288, 576, 463

EDGE_FEATURE_NAMES = ([f"ligand_invariant_{i}" for i in range(116)]
    + [f"surface_invariant_{i}" for i in range(116)]
    + [f"cross_embedding_{i}" for i in range(48)]
    + [f"surface_physical_{i}" for i in range(8)])

FEATURE_NAMES = [f"{pool}_{field}" for pool in ("mean", "max") for field in EDGE_FEATURE_NAMES]

CAPTURE_NAMES = ("lig_node_attr", "surface_node_attr", "surface_cross_edge_index",
                 "surface_cross_edge_attr")

NODE_FIELDS = {
    "ligand": {"x", "pos", "edge_mask"},
    "receptor": {"x", "pos", "center_pos", "nucleic_feat", "nucleic_anchor_pos",
                 "nucleic_anchor_type", "nucleic_anchor_parent_index"},
    "surface": {"x", "pos"},
}

EDGE_FIELDS = {
    ("ligand", "lig_bond", "ligand"): {"edge_index", "edge_attr"},
    ("receptor", "rec_contact", "receptor"): {"edge_index", "edge_attr"},
    ("surface", "surface_edge", "surface"): {"edge_index", "edge_attr"},
}

def require(condition, message):
    if not condition:
        raise RuntimeError(message)

def valid_sha(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None

def finite(*values):
    import torch
    for value in values:
        require(isinstance(value, torch.Tensor) and bool(torch.isfinite(value).all()),
                "Nonfinite/non-tensor numerical value; no masking, clipping, or sample skipping")

def state_digest(state):
    """Hash every parameter/buffer, including dtype/shape and integer buffers."""
    import torch
    h = hashlib.sha256()
    for name, value in sorted(state.items()):
        require(isinstance(name, str) and isinstance(value, torch.Tensor), "Malformed model state")
        finite(value)
        h.update(json.dumps([name, str(value.dtype), list(value.shape)]).encode())
        h.update(value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

def checkpoint_state(payload):
    """Only explicit whole-state wrappers; never filter out incompatible keys."""
    import torch
    require(isinstance(payload, Mapping) and payload, "Empty or non-mapping checkpoint")
    if not all(isinstance(v, torch.Tensor) for v in payload.values()):
        wrappers = [k for k in ("model", "model_state_dict", "state_dict") if k in payload]
        require(len(wrappers) == 1, "Ambiguous checkpoint state wrapper")
        payload = payload[wrappers[0]]
    require(isinstance(payload, Mapping) and payload
            and all(isinstance(k, str) and isinstance(v, torch.Tensor) for k, v in payload.items()),
            "Checkpoint must contain a complete tensor state")
    prefixed = [k.startswith("module.") for k in payload]
    require(all(prefixed) or not any(prefixed), "Mixed DDP prefixes are forbidden")
    result = {k.removeprefix("module.") if all(prefixed) else k: v for k, v in payload.items()}
    require(len(result) == len(payload), "Checkpoint normalization key collision")
    finite(*result.values())
    return result

def strict_load(model, state):
    target = model.state_dict()
    missing, unexpected = sorted(set(target)-set(state)), sorted(set(state)-set(target))
    shape = sorted(k for k in set(target) & set(state) if target[k].shape != state[k].shape)
    dtype = sorted(k for k in set(target) & set(state) if target[k].dtype != state[k].dtype)
    report = dict(loaded=len(set(target) & set(state)), missing=missing, unexpected=unexpected,
                  skipped_shape=shape, dtype_mismatch=dtype, strict=True,
                  key_coverage=len(set(target) & set(state))/max(1, len(target)))
    require(not missing and not unexpected and not shape and not dtype,
            "L2 Generator strict state coverage failed: " + json.dumps(report, sort_keys=True))
    model.load_state_dict(state, strict=True)
    require(state_digest(model.state_dict()) == state_digest(state), "Loaded state differs from source tensors")
    return report

def invariant_features(attributes, irreps):
    """Scalar channels + one stable norm per higher-l multiplicity (no cap)."""
    import torch
    finite(attributes)
    require(attributes.ndim == 2 and attributes.shape[1] == irreps.dim, "Irrep feature shape mismatch")
    parts = []
    for term, section in zip(irreps, irreps.slices()):
        block = attributes[:, section].reshape(len(attributes), term.mul, term.ir.dim)
        if term.ir.l == 0:
            parts.append(block.squeeze(-1))
        else:
            scale = block.abs().amax(-1)
            safe = scale.clamp_min(1e-12)
            norm = safe*(block/safe.unsqueeze(-1)).square().sum(-1).clamp_min(1e-12).sqrt()
            parts.append(torch.where(scale > 0, norm, torch.zeros_like(norm)))
    require(bool(parts), "Empty irreps")
    result = torch.cat(parts, -1)
    finite(result)
    return result

def atom_mean_max(edge_features, ligand_indices, n_atoms):
    """Keep all N atoms, including zero-neighbor atoms, without relabeling."""
    import torch
    require(type(n_atoms) is int and n_atoms > 0, "Invalid ligand atom count")
    require(isinstance(ligand_indices, torch.Tensor) and ligand_indices.dtype == torch.long
            and ligand_indices.ndim == 1 and edge_features.ndim == 2
            and len(edge_features) == len(ligand_indices) and edge_features.shape[1] > 0
            and edge_features.device == ligand_indices.device, "Invalid edge pooling inputs")
    finite(edge_features)
    if len(ligand_indices):
        require(int(ligand_indices.min()) >= 0 and int(ligand_indices.max()) < n_atoms,
                "Cross-edge atom mapping is out of bounds")
    counts = torch.bincount(ligand_indices, minlength=n_atoms)
    sums = edge_features.new_zeros((n_atoms, edge_features.shape[1]))
    sums.index_add_(0, ligand_indices, edge_features)
    maximum = edge_features.new_full(sums.shape, -torch.inf)
    maximum.scatter_reduce_(0, ligand_indices[:, None].expand_as(edge_features),
                            edge_features, reduce="amax", include_self=True)
    maximum = torch.where(counts[:, None] > 0, maximum, torch.zeros_like(maximum))
    tokens = torch.cat([sums/counts.clamp_min(1)[:, None], maximum], -1)
    finite(tokens)
    return tokens, counts

def _positions(value, shape, label):
    import torch
    result = torch.as_tensor(value).detach().cpu().to(torch.float32).clone()
    require(tuple(result.shape) == shape, f"Malformed {label} coordinates")
    finite(result)
    return result

def make_candidate_graph(template, xyz_global, original_center, *, graph_frame,
                         expected_atom_count, expected_atom_order_sha256,
                         candidate_atom_order_sha256):
    """Copy ONLY model input fields; candidate replaces native/matched position."""
    import torch
    from torch_geometric.data import Batch, HeteroData
    require(isinstance(template, HeteroData) and not isinstance(template, Batch),
            "Expected one unbatched chemical/receptor/surface graph")
    require(type(expected_atom_count) is int and expected_atom_count > 0
            and valid_sha(expected_atom_order_sha256)
            and expected_atom_order_sha256 == candidate_atom_order_sha256,
            "Candidate/SDF ordered atom mapping evidence is missing or inconsistent")
    require(graph_frame in ("l2_raw", "l2_prepared"), "Explicit frozen graph frame is required")
    prepared = getattr(template, "native_mdn_prepared", False)
    require(isinstance(prepared, bool) and prepared == (graph_frame == "l2_prepared"),
            "Prepared/raw frame marker mismatch; no automatic centering repair")
    require(set(template.node_types) == set(NODE_FIELDS) and set(template.edge_types) == set(EDGE_FIELDS),
            "Unexpected node/edge types; verify the G2.2 graph cache binding")
    require("original_center" in template._global_store, "Missing frozen graph original_center")
    center = torch.as_tensor(original_center).detach().cpu().reshape(-1)
    require(center.shape == (3,), "Malformed original center")
    center = _positions(center.reshape(1, 3), (1, 3), "original center")
    saved = torch.as_tensor(template.original_center).detach().cpu().reshape(-1)
    require(saved.shape == (3,) and torch.equal(saved.float(), center.reshape(3)),
            "Caller center differs from pinned graph; no frame inference")
    xyz = _positions(xyz_global, (expected_atom_count, 3), "candidate global")
    require(tuple(template["ligand"].pos.shape) == tuple(xyz.shape), "Ligand graph/candidate atom count mismatch")
    graph = HeteroData()
    removed = []
    for field in template._global_store.keys():
        removed.append("global."+field)
    for node, fields in NODE_FIELDS.items():
        require(fields <= set(template[node].keys()), f"Missing G2.2 input field in {node}")
        for field, value in template[node].items():
            if field not in fields:
                removed.append(node+"."+field)
                continue
            if node == "ligand" and field == "pos":
                # Never validate, encode, or retain the old/native ligand coordinates.
                graph[node][field] = xyz-center
                continue
            require(isinstance(value, torch.Tensor), f"Non-tensor G2.2 input: {node}.{field}")
            graph[node][field] = value.detach().cpu().clone()
            finite(graph[node][field])
    for edge, fields in EDGE_FIELDS.items():
        require(fields <= set(template[edge].keys()), "Missing chemical/surface graph edges")
        for field, value in template[edge].items():
            if field not in fields:
                removed.append(str(edge)+"."+field)
                continue
            require(isinstance(value, torch.Tensor), "Non-tensor graph edge input")
            graph[edge][field] = value.detach().cpu().clone()
            finite(graph[edge][field])
    receptor = graph["receptor"]
    if graph_frame == "l2_raw":
        receptor.center_pos = receptor.center_pos-center  # already-centered anchor_pos is unchanged.
    require(receptor.center_pos.shape == receptor.pos.shape, "Receptor center/node row mismatch")
    parent = receptor.nucleic_anchor_parent_index
    require(parent.dtype == torch.long and parent.ndim == 1
            and tuple(receptor.nucleic_anchor_pos.shape) == (len(parent), 3)
            and receptor.nucleic_anchor_type.shape == parent.shape, "Malformed NuSurf anchor inputs")
    if len(parent):
        require(int(parent.min()) >= 0 and int(parent.max()) < len(receptor.pos),
                "NuSurf anchor parent index is not local to this single graph")
    require(tuple(receptor.nucleic_feat.shape) == (len(receptor.pos), 12), "Not nucleic full12 features")
    n_surface = len(graph["surface"].pos)
    require(1 <= n_surface <= 512 and tuple(graph["surface"].x.shape) == (n_surface, 8),
            "Expected full8 L2-scaled surface with at most 512 vertices")
    require(tuple(graph["ligand"].x.shape) == (expected_atom_count, 16), "Ligand atom row identity mismatch")
    for edge in EDGE_FIELDS:
        index, attr = graph[edge].edge_index, graph[edge].edge_attr
        require(index.dtype == torch.long and index.ndim == 2 and index.shape[0] == 2
                and attr.ndim == 2 and len(attr) == index.shape[1], "Invalid edge index/feature shape")
        if index.numel():
            require(int(index.min()) >= 0 and int(index[0].max()) < len(graph[edge[0]].pos)
                    and int(index[1].max()) < len(graph[edge[2]].pos), "Graph edge identity out of range")
    require(graph["ligand"].edge_mask.dtype == torch.bool
            and graph["ligand"].edge_mask.shape == (graph["ligand", "ligand"].edge_index.shape[1],),
            "Invalid ordered ligand torsion/bond mask")
    for store in graph.stores:
        for value in store.values():
            finite(value)
    audit = dict(atom_count=expected_atom_count, atom_order_sha256=expected_atom_order_sha256,
        atom_order="unchanged_graph_and_SDF_order", coordinate_frame="global_receptor_frame",
        model_coordinate_frame="receptor_centered", ligand_center_subtractions=1,
        center_pos_subtractions=int(graph_frame == "l2_raw"), anchor_center_subtractions=0,
        original_graph_frame=graph_frame, removed_metadata=sorted(removed),
        native_coordinates_entered_model=False, labels_entered_model=False,
        surface_scaler_reapplied=False, caller_must_verify_source_file_hashes=True)
    return graph, audit

def capture_forward(model, batch):
    """Passively observe the exact Python return frame, preserving forward math."""
    code = type(model).forward.__code__
    values = {}
    previous = sys.getprofile()
    require(previous is None, "Existing profiler cannot be overwritten")

    def observer(frame, event, arg):
        if frame.f_code is code and frame.f_locals.get("self") is model and event == "return":
            for name in CAPTURE_NAMES:
                if name in frame.f_locals:
                    values[name] = frame.f_locals[name]

    sys.setprofile(observer)
    try:
        output = model(batch)
    finally:
        sys.setprofile(previous)
    require(set(values) == set(CAPTURE_NAMES), "Exact Generator forward capture is incomplete")
    require(isinstance(output, (tuple, list)) and len(output) == 3,
            "Not the diffusion Generator tr/rot/tor forward")
    finite(*values.values(), *output)
    return values, output

def captured_tokens(model, batch, values):
    import torch
    from e3nn import o3
    edges = values["surface_cross_edge_index"]
    require(edges.dtype == torch.long and edges.ndim == 2 and edges.shape[0] == 2,
            "Malformed original interaction graph")
    ligand = invariant_features(values["lig_node_attr"], o3.Irreps(model.lig_conv_layers[-1].out_irreps))
    surface = invariant_features(values["surface_node_attr"], o3.Irreps(model.surface_conv_layers[-1].out_irreps))
    physical, cross = batch["surface"].x.float(), values["surface_cross_edge_attr"]
    require(ligand.shape == (len(batch["ligand"].pos), 116)
            and surface.shape == (len(batch["surface"].pos), 116)
            and cross.shape == (edges.shape[1], 48), "Frozen G2.2 hidden-state layout changed")
    li, su = edges
    if edges.numel():
        require(int(edges.min()) >= 0 and int(li.max()) < len(ligand) and int(su.max()) < len(surface),
                "Original interaction edge indices are out of range")
    edge_features = torch.cat([ligand[li], surface[su], cross, physical[su]], -1)
    require(edge_features.shape[1] == EDGE_WIDTH, "Unexpected invariant edge width")
    tokens, counts = atom_mean_max(edge_features, li, len(ligand))
    require(tokens.shape[1] == TOKEN_WIDTH and bool((counts <= 30).all()),
            "Unexpected pooled token width/changed original cross-neighbor cap")
    # Full edge_features are ephemeral; the returned numerical artifact is only [N,576].
    return tokens.detach().to(device="cpu", dtype=torch.float32), counts.detach().cpu()
