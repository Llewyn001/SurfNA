"""Normalize nucleotide anchor parent indices at the model input boundary.

PyG does not increment custom index fields stored on a HeteroData node store.
Use its collation slices and recorded increments to recover local indices,
then apply receptor-node offsets. Updating the increment metadata together
with the values preserves to_data_list(), re-batching, and repeat forwards.
"""
import torch


def normalize_nucleic_anchor_parent_indices(data):
    """Return whether parent indices changed; modify only indices and metadata.

    Both legacy node-store collation (zero increments) and correctly incremented
    collation are supported. Multi-graph inputs need genuine PyG batch metadata;
    manually edited global indices with stale metadata are unsupported. Their
    intended local/global frame is not generally identifiable from values alone.
    """
    if 'receptor' not in data.node_types:
        return False
    receptor = data['receptor']
    if not hasattr(receptor, 'nucleic_anchor_pos'):
        return False
    if not hasattr(receptor, 'nucleic_anchor_parent_index'):
        raise ValueError('NuSurf anchors are missing nucleic_anchor_parent_index')
    parents = receptor.nucleic_anchor_parent_index
    if not torch.is_tensor(parents) or parents.dtype != torch.long or parents.ndim != 1:
        raise ValueError('NuSurf anchor parent indices must be a one-dimensional int64 tensor')
    if parents.numel() != receptor.nucleic_anchor_pos.shape[0]:
        raise ValueError('NuSurf anchor positions and parent indices have different lengths')

    # Bare HeteroData treats unknown *_dict attributes as collect() requests,
    # which raise KeyError rather than AttributeError in the pinned PyG version.
    try:
        slices = data._slice_dict.get('receptor', {})
    except (AttributeError, KeyError):
        slices = {}
    try:
        increments = data._inc_dict.get('receptor', {})
    except (AttributeError, KeyError):
        increments = {}
    key = 'nucleic_anchor_parent_index'
    ptr = getattr(receptor, 'ptr', None)
    # The inference sampler re-batches single-graph Batch objects. PyG omits
    # ptr in that case, but retains exactly the same receptor position slices.
    if ptr is None:
        ptr = slices.get('pos')
    if ptr is None or key not in slices or key not in increments:
        batch = getattr(receptor, 'batch', None)
        graphs = int(getattr(data, 'num_graphs', 1))
        if batch is not None and batch.numel():
            graphs = max(graphs, int(batch.max().item()) + 1)
        if graphs > 1:
            raise ValueError('NuSurf multi-graph input lacks PyG anchor slices/increment metadata')
        if bool(((parents < 0) | (parents >= receptor.num_nodes)).any().item()):
            raise ValueError('NuSurf single-graph anchor parent index is out of bounds')
        return False

    ptr = ptr.to(device=parents.device, dtype=torch.long)
    anchor_slices = slices[key].to(device=parents.device, dtype=torch.long)
    old_increments = increments[key]
    current = torch.as_tensor(old_increments, device=parents.device, dtype=torch.long).reshape(-1)
    graph_count = ptr.numel() - 1
    if (anchor_slices.numel() != graph_count + 1 or current.numel() != graph_count or
            int(anchor_slices[0].item()) != 0 or int(anchor_slices[-1].item()) != parents.numel() or
            int(ptr[0].item()) != 0 or int(ptr[-1].item()) != receptor.num_nodes):
        raise ValueError('NuSurf batch metadata has inconsistent graph counts or lengths')
    anchor_counts = anchor_slices[1:] - anchor_slices[:-1]
    receptor_counts = ptr[1:] - ptr[:-1]
    if bool(((anchor_counts < 0) | (receptor_counts < 0)).any().item()):
        raise ValueError('NuSurf batch slices must be monotonic')
    receptor_graph = torch.repeat_interleave(torch.arange(graph_count, device=parents.device), receptor_counts)
    batch = getattr(receptor, 'batch', None)
    if batch is not None and not torch.equal(batch.to(parents.device), receptor_graph):
        raise ValueError('NuSurf receptor slices disagree with receptor graph membership')
    anchor_graph = torch.repeat_interleave(torch.arange(graph_count, device=parents.device), anchor_counts)
    local = parents - current[anchor_graph]
    if bool(((local < 0) | (local >= receptor_counts[anchor_graph])).any().item()):
        raise ValueError('NuSurf local parent index is out of bounds after undoing recorded PyG increments')
    target = ptr[:-1]
    corrected = local + target[anchor_graph]
    changed = not torch.equal(corrected, parents)
    if changed:
        receptor.nucleic_anchor_parent_index = corrected
    if not torch.equal(current, target):
        # Copy dictionaries so clones cannot accidentally share a changed mapping.
        # Preserve the metadata device/shape expected by PyG separate().
        if torch.is_tensor(old_increments):
            new_increments = target.to(device=old_increments.device, dtype=old_increments.dtype).reshape(old_increments.shape)
        else:
            new_increments = target.detach().cpu()
        data._inc_dict = {
            **data._inc_dict,
            'receptor': {**increments, key: new_increments},
        }
    return changed
