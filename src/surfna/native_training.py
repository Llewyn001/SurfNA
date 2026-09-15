"""Native MDN contact objective and frozen graph preparation primitives."""
from .scorer_common import finite
from .mdn_features import clean_graph
from .runtime import ROOT
import json
RECIPE=json.loads((ROOT/"configs/mdn_training.json").read_text())

def encode(model, item, device):
    import torch
    from torch_geometric.data import Batch
    with torch.no_grad():
        batch = Batch.from_data_list([clean_graph(item["graph"])]).to(device)
        surface = model.backbone.encode_surface_static(batch)
        ligand = model.backbone.encode_ligand_intra(batch)
        finite(surface.scalar, surface.position, ligand.scalar, ligand.position)
        if not bool(surface.mask.all()) or not bool(ligand.mask.all()):
            raise RuntimeError("Unexpected padding in single-graph native encoding")
        value = {"name": item["name"], "split": item["split"],
            "ligand": ligand.scalar[0].cpu(), "surface": surface.scalar[0].cpu(),
            "ligand_pos": ligand.position[0].cpu(), "surface_pos": surface.position[0].cpu(),
            "contact_index": item["contact_index"], "contact_distance": item["contact_distance"]}
    return value, (surface, ligand, batch)

def native_loss(head, items, device, optimizer=None):
    import torch
    from models.surfna_v2_scorer_components import mixture_log_probability
    if not items:
        raise RuntimeError("Empty native training batch")
    total, pair_loss, pairs_total = 0.0, 0.0, 0
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    chunk = RECIPE["pair_chunk_size"]
    for item in items:
        ij = item["contact_index"]
        n = len(ij)
        if n == 0:
            raise RuntimeError("No contact pairs; no skipping is permitted")
        subtotal = 0.0
        # Copy pair features chunkwise rather than allocating all native
        # contact features on GPU. The complex-balanced objective is identical.
        for offset in range(0, n, chunk):
            pair = ij[offset:offset + chunk]
            features = torch.cat((item["ligand"][pair[:, 0]], item["surface"][pair[:, 1]]), dim=-1).to(device)
            distance = item["contact_distance"][offset:offset + chunk].to(device)
            pi, mu, sigma = head.predict_parameters(features)
            finite(features, distance, pi, mu, sigma)
            nll = -mixture_log_probability(pi, sigma, mu, distance)
            finite(nll)
            if optimizer is not None:
                (nll.sum() / (n * len(items))).backward()
            subtotal += float(nll.detach().sum())
        total += subtotal / n
        pair_loss += subtotal
        pairs_total += n
    if optimizer is not None:
        gradients = [p.grad for p in head.parameters() if p.grad is not None]
        if not gradients:
            raise RuntimeError("No native MDN-head gradients")
        finite(*gradients)
        norm = torch.nn.utils.clip_grad_norm_(head.parameters(), RECIPE["gradient_clip"], error_if_nonfinite=True)
        if not float(norm) > 0:
            raise RuntimeError("Native MDN-head gradient is zero")
        optimizer.step()
        finite(*[p.detach() for p in head.parameters()])
    return total / len(items), pair_loss, pairs_total

def validate(head, items, device):
    import torch
    if not items:
        raise RuntimeError("Empty native validation split")
    head.eval()
    losses, total, pairs = [], 0.0, 0
    with torch.no_grad():
        for item in items:
            loss, pair_sum, count = native_loss(head, [item], device)
            losses.append(loss)
            total += pair_sum
            pairs += count
    return {"n_complexes": len(items), "nll": sum(losses) / len(losses),
            "pair_weighted_nll": total / pairs, "pairs": pairs}

def native_coordinates(value):
    import numpy as np
    import torch
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    while array.ndim == 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim != 2 or array.shape[1] != 3 or not np.isfinite(array).all():
        raise RuntimeError("Native orig_pos must be one finite [atoms,3] conformation")
    return array.copy()

def center_auxiliary_once(graph, center):
    """Pinned L2 raw-cache frames, not an adaptive/retry repair heuristic.

The G2.2 get_complex function has already centered nucleic anchors. The
loader separately centers center_pos. Record both plausible errors to expose
a contradicted source frame, including centers near zero where both coincide.
"""
    import torch
    if getattr(graph, "native_mdn_prepared", False):
        raise RuntimeError("Graph was already native-prepared; refusing double centering")
    receptor = graph["receptor"]
    result = {}
    if "center_pos" in receptor:
        raw = receptor.center_pos
        shifted = raw - center.to(raw)
        if raw.shape != receptor.pos.shape:
            raise RuntimeError("Receptor center_pos / pos row identity mismatch")
        finite = torch.isfinite(raw).all() and torch.isfinite(shifted).all()
        if not finite:
            raise RuntimeError("Nonfinite receptor center coordinates")
        before = float(torch.linalg.vector_norm(raw - receptor.pos.to(raw), dim=-1).median())
        after = float(torch.linalg.vector_norm(shifted - receptor.pos.to(raw), dim=-1).median())
        if after > before + 1.0:
            raise RuntimeError("Raw center_pos contradicts the pinned global-coordinate policy")
        receptor.center_pos = shifted
        result["center_pos"] = {"input_frame": "global", "output_frame": "centered", "subtractions": 1,
                                "parent_median_before_A": before, "parent_median_after_A": after}
    if "nucleic_anchor_pos" in receptor:
        raw = receptor.nucleic_anchor_pos
        parents = receptor.nucleic_anchor_parent_index.long().reshape(-1)
        if raw.shape != (len(parents), 3) or not torch.isfinite(raw).all():
            raise RuntimeError("Invalid nucleic anchor shape/finite gate")
        if len(parents):
            if int(parents.min()) < 0 or int(parents.max()) >= len(receptor.pos):
                raise RuntimeError("Invalid nucleic anchor parent index")
            target = receptor.pos[parents].to(raw)
            before = float(torch.linalg.vector_norm(raw - target, dim=-1).median())
            if_shifted = float(torch.linalg.vector_norm(raw - center.to(raw) - target, dim=-1).median())
            if if_shifted + 1.0 < before:
                raise RuntimeError("Raw anchors contradict pinned already-centered L2 cache; no implicit repair")
        else:
            before = if_shifted = None
        # Intentionally leave anchors unchanged; do not copy the old V3 rule.
        result["nucleic_anchor_pos"] = {"input_frame": "centered", "output_frame": "centered", "subtractions": 0,
                                       "parent_median_before_A": before, "if_subtracted_again_A": if_shifted}
    graph.native_mdn_prepared = True
    return result
