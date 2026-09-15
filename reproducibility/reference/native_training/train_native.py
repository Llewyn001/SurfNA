"""One-shot L2 native-MDN B: smoke, factorized cache, head-only NLL, export.

Adapted from iterations/native_prior_v3_20260828/{run_arm,export_prior_bundle}.py.
No generated poses, legacy NA heads, RMSD selection, or test evaluation here.
The original B 20-epoch cap / patience-4 / fresh-optimizer recipe is unchanged.
"""
from __future__ import annotations

import argparse
import os
import random
import time
import traceback
from pathlib import Path

from native_contract import (RECIPE, atomic_json, atomic_torch, checked_file,
    event, exclusive_json, key, load_graph_index, read_config, read_json, sha)
from model_api import build_model, clean_graph, finite, seed_all, state_digest, static_prior_state


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


def load_item(row):
    import torch
    checked_file(row["path"], row["artifact_sha256"])
    item = torch.load(row["path"], map_location="cpu", weights_only=False)
    if item["name"] != row["name"] or item["split"] != row["split"]:
        raise RuntimeError("Native artifact identity drift")
    return item


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


def smoke(model, metadata, rows, output, device):
    import torch
    from torch_geometric.data import Batch
    started = time.monotonic()
    selected = sorted((r for r in rows if r["split"] == "train"),
                      key=lambda r: key("smoke|" + r["name"]))[:RECIPE["smoke_train_complexes"]]
    if len(selected) != RECIPE["smoke_train_complexes"]:
        raise RuntimeError("Insufficient training members for the fixed native8 smoke gate")
    items, parity_max, rotation_max = [], 0.0, 0.0
    for index, row in enumerate(selected):
        item = load_item(row)
        value, (surface, ligand, batch) = encode(model, item, device)
        items.append(value)
        if index == 0:
            with torch.no_grad():
                direct = model.build_static_pair_features(ligand, surface)[0]
                ls, ss = value["ligand"].to(device), value["surface"].to(device)
                cached = torch.cat((ls[:, None].expand(-1, len(ss), -1),
                                    ss[None].expand(len(ls), -1, -1)), dim=-1)[None]
                parity_max = float((direct - cached).abs().max())
                if not torch.equal(direct, cached):
                    raise RuntimeError("Factorized/direct native pair features differ")
                direct_parameters = model.prior_head.predict_parameters(direct)
                cached_parameters = model.prior_head.predict_parameters(cached)
                finite(*direct_parameters, *cached_parameters)
                if any(not torch.equal(a, b) for a, b in zip(direct_parameters, cached_parameters)):
                    raise RuntimeError("Factorized/direct MDN parameter parity failed")
                translated = clean_graph(item["graph"])
                translated["ligand"].pos += torch.tensor([11., -7., 3.], dtype=translated["ligand"].pos.dtype)
                shifted = model.backbone.encode_ligand_intra(Batch.from_data_list([translated]).to(device))
                finite(shifted.scalar)
                if not torch.allclose(ligand.scalar, shifted.scalar, atol=3e-3, rtol=3e-3):
                    raise RuntimeError("Ligand-only translation invariance gate failed")
                rotated = clean_graph(item["graph"])
                rot = torch.tensor([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]], dtype=rotated["ligand"].pos.dtype)
                rotated["ligand"].pos = rotated["ligand"].pos @ rot.T
                rotated_state = model.backbone.encode_ligand_intra(Batch.from_data_list([rotated]).to(device))
                finite(rotated_state.scalar)
                rotation_max = float((ligand.scalar - rotated_state.scalar).abs().max())
                if not torch.allclose(ligand.scalar, rotated_state.scalar, atol=3e-3, rtol=3e-3):
                    raise RuntimeError("Ligand-only rotation invariance gate failed")
        event("native_smoke_encode", complete=index + 1, total=len(selected))
    saved = {k: v.detach().cpu().clone() for k, v in model.prior_head.state_dict().items()}
    initial = validate(model.prior_head, items, device)["nll"]
    optimizer = torch.optim.AdamW(model.prior_head.parameters(), lr=RECIPE["smoke_lr"],
                                 weight_decay=RECIPE["weight_decay"])
    for step in range(RECIPE["smoke_steps"]):
        model.prior_head.train()
        native_loss(model.prior_head, items, device, optimizer)
        if step % 20 == 0:
            event("native_smoke_memorization", step=step,
                  nll=validate(model.prior_head, items, device)["nll"])
    final = validate(model.prior_head, items, device)["nll"]
    changed = state_digest(model.prior_head.state_dict()) != state_digest(saved)
    if not changed or not final < initial - 1e-5:
        raise RuntimeError(f"Native8 train=val learning gate failed: {initial} -> {final}")
    model.prior_head.load_state_dict(saved, strict=True)
    model.prior_head.eval()
    model.prior_head.zero_grad(set_to_none=True)
    if state_digest(model.prior_head.state_dict()) != metadata["initial_head_sha256"]:
        raise RuntimeError("Smoke head did not reset exactly")
    if state_digest(model.backbone.state_dict()) != metadata["initial_backbone_sha256"]:
        raise RuntimeError("Backbone changed during smoke")
    # No smoke optimizer state survives. Formal RNG is reset again after encode.
    del optimizer, saved
    exclusive_json(output / "SMOKE_READY.json", {
        "status": "PASS", "graph_count": len(selected), "members": [r["name"] for r in selected],
        "initial_nll": initial, "final_memorization_nll": final, "parameters_changed": changed,
        "formal_weights_reset": True, "fresh_formal_optimizer": True, "backbone_unchanged": True,
        "factorized_parity_max_abs": parity_max, "ligand_rotation_max_abs": rotation_max,
        "test_opened": False, "seconds": time.monotonic() - started})
    event("native_smoke_pass", initial_nll=initial, final_nll=final, formal_weights_reset=True)


def cache_features(model, metadata, rows, output, device):
    features = output / "features"
    features.mkdir(exist_ok=False)
    items, index = [], []
    started = time.monotonic()
    for ordinal, row in enumerate(rows):
        value, _ = encode(model, load_item(row), device)
        path = features / (key(row["name"]) + ".pt")
        atomic_torch(path, value)
        index.append({"name": row["name"], "split": row["split"], "path": str(path), "sha256": sha(path)})
        items.append(value)
        if (ordinal + 1) % 25 == 0 or ordinal == len(rows) - 1:
            event("native_features", complete=ordinal + 1, total=len(rows), seconds=time.monotonic() - started)
    atomic_json(output / "feature_index.json", index)
    exclusive_json(output / "FEATURES_READY.json", {
        "status": "PASS", "count": len(rows), "feature_index_sha256": sha(output / "feature_index.json"),
        "frozen_backbone_sha256": metadata["initial_backbone_sha256"], "test_opened": False,
        "seconds": time.monotonic() - started})
    return items


def save_head(path, head, epoch, validation, metadata):
    atomic_torch(path, {"prior_head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
        "epoch": epoch, "native_val": validation, "metadata": metadata,
        "selection": RECIPE["selection"], "backbone_trainable": False})


def train_head(model, metadata, items, output, config, device):
    import torch
    head = model.prior_head
    train = [x for x in items if x["split"] == "train"]
    val = [x for x in items if x["split"] == "val"]
    counts = {s: config["dataset"][s]["count"] for s in ("train", "val")}
    if len(train) != counts["train"] or len(val) != counts["val"]:
        raise RuntimeError("Native feature counts differ from frozen L2 split")
    initial = validate(head, val, device)
    save_head(output / "initial_head.pt", head, -1, initial, metadata)
    save_head(output / "best_head.pt", head, -1, initial, metadata)
    best, best_epoch, stale = initial["nll"], -1, 0
    optimizer = torch.optim.AdamW(head.parameters(), lr=RECIPE["lr"], weight_decay=RECIPE["weight_decay"])
    rng = random.Random(RECIPE["seed"])
    history = [{"epoch": -1, "val": initial, "selection": "initial", "seconds": 0}]
    atomic_json(output / "history.json", history)
    for epoch in range(RECIPE["max_epochs"]):
        start = time.monotonic()
        order = list(range(len(train)))
        rng.shuffle(order)
        head.train()
        loss_sum, complex_count = 0.0, 0
        for offset in range(0, len(order), RECIPE["batch_size"]):
            group = [train[i] for i in order[offset:offset + RECIPE["batch_size"]]]
            loss, _, _ = native_loss(head, group, device, optimizer)
            loss_sum += loss * len(group)
            complex_count += len(group)
            if (offset // RECIPE["batch_size"] + 1) % 100 == 0:
                event("native_train", epoch=epoch, complete=complex_count, total=len(train))
        result = validate(head, val, device)
        improved = result["nll"] < best - RECIPE["improvement_epsilon"]
        if improved:
            best, best_epoch, stale = result["nll"], epoch, 0
            save_head(output / "best_head.pt", head, epoch, result, metadata)
        else:
            stale += 1
        save_head(output / "last_head.pt", head, epoch, result, metadata)
        record = {"epoch": epoch, "train_nll": loss_sum / complex_count, "val": result,
                  "improved": improved, "best_epoch": best_epoch, "stale_epochs": stale,
                  "seconds": time.monotonic() - start}
        history.append(record)
        atomic_json(output / "history.json", history)
        event("native_epoch", **record)
        if epoch + 1 >= RECIPE["min_epochs"] and stale >= RECIPE["patience"]:
            break
    if any(p.grad is not None for p in model.backbone.parameters()):
        raise RuntimeError("Unexpected backbone gradients")
    if state_digest(model.backbone.state_dict()) != metadata["initial_backbone_sha256"]:
        raise RuntimeError("Frozen backbone changed during native head training")
    result = {"status": "PASS", "best_epoch": best_epoch, "best_native_val_nll": best,
        "completed_epochs": epoch + 1, "stop_reason": "patience4" if stale >= 4 else "max20",
        "best_head_sha256": sha(output / "best_head.pt"), "initial_head_sha256": sha(output / "initial_head.pt"),
        "history_sha256": sha(output / "history.json"), "selection": RECIPE["selection"],
        "contract_sha256": metadata["contract_sha256"], "root_contract_sha256": config["root_contract_sha256"],
        "graph_index_sha256": sha(output / "graph_index.json"), "split_counts": counts,
        "fresh_optimizer": True, "scheduler": None, "backbone_unchanged": True,
        "test_opened": False, "skipped_batches": 0}
    exclusive_json(output / "TRAIN_READY.json", result)
    return result


def export_bundle(config, output, training):
    import torch
    best = torch.load(output / "best_head.pt", map_location="cpu", weights_only=False)
    checked_file(output / "best_head.pt", training["best_head_sha256"])
    model, metadata = build_model(config, output / "native_train_reference.json", "cpu")
    if metadata["initial_backbone_sha256"] != best["metadata"]["initial_backbone_sha256"]:
        raise RuntimeError("CPU reconstruction differs from trained frozen backbone")
    model.prior_head.load_state_dict(best["prior_head"], strict=True)
    selected = static_prior_state(model)
    bundle_meta = {"arm": "protein_mdn_transfer", "epoch": best["epoch"],
        "source_head_sha256": training["best_head_sha256"], "contract_sha256": training["contract_sha256"],
        "root_contract_sha256": config["root_contract_sha256"],
        "scope": "shared_static_backbone_plus_native_MDN_prior_only",
        "dynamic_ranker_trained": False, "virtual_screening_validated": False,
        "surface_schema": "v2_full8", "surface_vertices": 512,
        "surface_scaler_sha256": config["dataset"]["surface_scaler"]["sha256"],
        "reference_sha256": sha(output / "native_train_reference.json"),
        "equivariant_graph_rms_cap": 100, "native_val": best["native_val"],
        "benchmark_test_opened": False, "test_opened": False,
        "initial_backbone_sha256": best["metadata"]["initial_backbone_sha256"],
        "status": "L2_native_prior_ready_for_separate_rank_training"}
    destination = output / "native_prior_static_bundle.pt"
    if destination.exists():
        raise RuntimeError("Native bundle exists; refusing overwrite")
    atomic_torch(destination, {"model": selected, "epoch": best["epoch"], "metadata": bundle_meta,
        "architecture_args": metadata["architecture_args"], "architecture_args_not_a_training_config": True,
        "surface_scaler": read_json(config["dataset"]["surface_scaler"]["path"]),
        "reference_density": read_json(output / "native_train_reference.json")})
    reloaded = torch.load(destination, map_location="cpu", weights_only=False)
    if set(reloaded["model"]) != set(selected) or state_digest(reloaded["model"]) != state_digest(selected):
        raise RuntimeError("Exported native bundle tensor readback mismatch")
    bundle_meta.update(bundle_sha256=sha(destination), tensor_count=len(selected),
                       bytes=destination.stat().st_size, path=str(destination))
    exclusive_json(output / "native_prior_static_bundle.json", bundle_meta)
    return bundle_meta


def run(root):
    root = Path(root).resolve()
    config = read_config(root)
    rows = load_graph_index(root, config)
    output = root / "native"
    exclusive_json(output / "TRAINING_ATTEMPT.json", {
        "pid": os.getpid(), "started_at": time.time(), "contract_sha256": sha(root / "native_contract.json"),
        "arm": "protein_mdn_transfer", "resume": False, "test_opened": False})
    stage = "initialization"
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one CUDA GPU; this is not DDP and does not choose a GPU automatically")
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Native scorer requires one process; no torchrun/accelerate launch")
        device = torch.device("cuda:0")
        model, metadata = build_model(config, output / "native_train_reference.json", device)
        metadata.update(contract_sha256=sha(root / "native_contract.json"),
                        graph_index_sha256=sha(output / "graph_index.json"),
                        hostname=os.uname().nodename, pid=os.getpid(), gpu=torch.cuda.get_device_name(0))
        exclusive_json(output / "initialization.json", metadata)
        if not metadata["trainable_names"] or any(not name.startswith("prior_head.") for name in metadata["trainable_names"]):
            raise RuntimeError("Only the native MDN prior_head may be trainable")
        stage = "native8_smoke"
        smoke(model, metadata, rows, output, device)
        stage = "frozen_factorized_feature_cache"
        items = cache_features(model, metadata, rows, output, device)
        seed_all()
        stage = "native_head_training"
        training = train_head(model, metadata, items, output, config, device)
        # Export reconstructs on CPU, checking exact frozen-backbone parity.
        del items, model
        torch.cuda.empty_cache()
        stage = "static_prior_export"
        bundle = export_bundle(config, output, training)
        exclusive_json(output / "READY.json", {
            "status": "PASS", "contract_sha256": sha(root / "native_contract.json"),
            "root_contract_sha256": config["root_contract_sha256"],
            "best_head_sha256": training["best_head_sha256"], "bundle_sha256": bundle["bundle_sha256"],
            "bundle_json_sha256": sha(output / "native_prior_static_bundle.json"),
            "graph_index_sha256": sha(output / "graph_index.json"),
            "reference_sha256": sha(output / "native_train_reference.json"),
            "train_ready_sha256": sha(output / "TRAIN_READY.json"), "split_counts": training["split_counts"],
            "best_epoch": training["best_epoch"], "best_native_val_nll": training["best_native_val_nll"],
            "test_opened": False, "virtual_screening_validated": False,
            "scope": "native_prior_stage_only; rank training is a separate stage"})
        event("native_complete", status="PASS", best_epoch=training["best_epoch"],
              best_native_val_nll=training["best_native_val_nll"], test_opened=False)
    except Exception:
        exclusive_json(output / "TRAINING_FAILED.json", {
            "stage": stage, "error": traceback.format_exc(), "time": time.time(),
            "no_retry": True, "no_resume": True, "test_opened": False})
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    run(parser.parse_args().root)
