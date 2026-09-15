#!/usr/bin/env python3
"""DDP rank entry point: immutable paired initialization plus frozen-cache guard."""

from __future__ import annotations

import atexit
import hashlib
import importlib
import json
import os
import runpy
import sys
import tempfile
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def state_digest(state: dict[str, Any], keys: list[str] | None = None) -> str:
    digest = hashlib.sha256()
    for key in sorted(state if keys is None else keys):
        value = state[key]
        digest.update(key.encode())
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def deny_rebuild(*_args, **_kwargs):
    raise RuntimeError("Frozen graph cache missing or mismatched: automatic preprocessing/rebuilding is forbidden in every training rank")


def attach_forward_trace(model: Any, trace_path: Path, roots: list[str]) -> None:
    import torch
    if os.environ.get("LOCAL_RANK", "0") != "0":
        return
    counters = {root: 0 for root in roots}

    class TraceProxy(torch.nn.Module):
        """Call-through wrapper usable around e3nn TorchScript submodules."""
        def __init__(self, inner: torch.nn.Module, label: str) -> None:
            super().__init__()
            self.inner = inner
            self.label = label

        def forward(self, *args, **kwargs):
            counters[self.label] += 1
            return self.inner(*args, **kwargs)

        def __getattr__(self, name: str):
            # SurfaceScoreModelV3 reads patch-tokenizer diagnostics after the
            # call.  Preserve the wrapped module's public runtime attributes.
            try:
                return super().__getattr__(name)
            except AttributeError:
                inner = super().__getattr__("inner")
                return getattr(inner, name)

    wrapped = 0
    for root in roots:
        if not hasattr(model, root):
            raise RuntimeError(f"SurfacePath candidate is absent from model: {root}")
        module = getattr(model, root)
        if isinstance(module, torch.nn.ModuleList):
            for index in range(len(module)):
                module[index] = TraceProxy(module[index], root)
                wrapped += 1
        elif isinstance(module, torch.nn.Module):
            setattr(model, root, TraceProxy(module, root))
            wrapped += 1
        else:
            raise RuntimeError(f"SurfacePath candidate is not a module: {root}")
    if wrapped == 0:
        raise RuntimeError("No candidate SurfacePath modules could be wrapped for real-forward tracing")
    def write_trace() -> None:
        atomic_json(trace_path, {"schema_version": "surfna-surface-forward-trace-v1", "status": "process_exit", "observer": "transparent_call_proxy", "candidate_module_call_counts": counters, "all_surface_roots_observed": all(value > 0 for value in counters.values())})
    atexit.register(write_trace)


def apply_paired_initialization(model: Any, config_path: Path) -> dict[str, Any]:
    import torch
    config = json.loads(config_path.read_text())
    manifest_path = Path(config["manifest"]["path"]).resolve(strict=True)
    if sha256_file(manifest_path) != config["manifest"]["sha256"]:
        raise RuntimeError("Parameter manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    status = manifest.get("status")
    allowed = {"provisional_pending_real_graph_smoke_trace"} if os.environ.get("SURFNA_SMOKE", "0") == "1" else {"final_after_real_graph_smoke"}
    if status not in allowed:
        raise RuntimeError(f"Manifest state {status!r} is not allowed for this run")
    base_path = Path(config["base_init"]["path"]).resolve(strict=True)
    source_path = Path(config["source_checkpoint"]["path"]).resolve(strict=True)
    if sha256_file(base_path) != config["base_init"]["sha256"]:
        raise RuntimeError("Paired base initialization hash mismatch")
    if sha256_file(source_path) != config["source_checkpoint"]["sha256"]:
        raise RuntimeError("Protein source checkpoint hash mismatch")
    target_state = model.state_dict()
    base_raw = torch.load(base_path, map_location="cpu")
    base_state = base_raw["model"] if isinstance(base_raw, dict) and "model" in base_raw else base_raw
    if set(base_state) != set(target_state):
        raise RuntimeError("Paired base state keys differ from the target model")
    for key in target_state:
        if tuple(base_state[key].shape) != tuple(target_state[key].shape):
            raise RuntimeError(f"Paired base tensor shape mismatch: {key}")
    model.load_state_dict(base_state, strict=True)
    source_raw = torch.load(source_path, map_location="cpu")
    source_state = source_raw.get("model", source_raw) if isinstance(source_raw, dict) else source_raw
    source_state = {(key[7:] if key.startswith("module.") else key): value for key, value in source_state.items()}
    compatible = {key: value for key, value in source_state.items() if key in target_state and tuple(value.shape) == tuple(target_state[key].shape)}
    mismatch = sorted(key for key, value in source_state.items() if key in target_state and tuple(value.shape) != tuple(target_state[key].shape))
    missing = sorted(key for key in target_state if key not in compatible)
    raw_compatibility = {"loaded": len(compatible), "missing": len(missing), "unexpected": 0, "skipped_shape": len(mismatch)}
    if raw_compatibility != manifest["source_compatibility"]:
        raise RuntimeError(f"Source compatibility drift: {raw_compatibility} != {manifest['source_compatibility']}")
    selected = list(config["transfer_keys"])
    if not set(selected).issubset(compatible):
        raise RuntimeError("Initialization recipe selected source-incompatible tensors")
    if config["arm"] == "Scratch-Full8" and selected:
        raise RuntimeError("Scratch arm may not overlay source tensors")
    model.load_state_dict({key: compatible[key] for key in selected}, strict=False)
    current = model.state_dict()
    expected = {key: (compatible[key] if key in selected else base_state[key]) for key in current}
    for key in current:
        if not torch.equal(current[key].detach().cpu(), expected[key].detach().cpu()):
            raise RuntimeError(f"Initialization tensor mismatch after overlay: {key}")
    param_names = set(name for name, _ in model.named_parameters())
    untrainable = sorted(name for name, parameter in model.named_parameters() if not parameter.requires_grad)
    if untrainable:
        raise RuntimeError(f"Full-model fine-tuning violated; frozen parameters: {untrainable[:10]}")
    report = {
        "schema_version": "surfna-paired-initialization-runtime-v1", "arm": config["arm"], "seed": config["seed"],
        "base_init_sha256": sha256_file(base_path), "source_checkpoint_sha256": sha256_file(source_path),
        "manifest_sha256": sha256_file(manifest_path), "manifest_status": status,
        "raw_source_compatibility": raw_compatibility, "selected_source_tensors": len(selected),
        "selected_parameter_tensors": len([key for key in selected if key in param_names]),
        "initial_full_state_sha256": state_digest(current), "initial_selected_state_sha256": state_digest(current, selected) if selected else None,
        "optimizer_scheduler": "fresh_not_loaded", "full_model_finetuning": True,
    }
    report_path = Path(os.environ["SURFNA_INIT_REPORT"]).resolve()
    if os.environ.get("LOCAL_RANK", "0") == "0":
        atomic_json(report_path, report)
    trace_path = os.environ.get("SURFNA_FORWARD_TRACE")
    if trace_path:
        attach_forward_trace(model, Path(trace_path).resolve(), manifest["surface_path_roots"])
    print("SurfNA paired initialization active: " + json.dumps(report, sort_keys=True), file=sys.stderr, flush=True)
    return report


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        raise ValueError("Usage: paired_train_rank.py /path/to/train_accelerate.py [training args]")
    script = Path(args[0]).resolve(strict=True)
    if script.name != "train_accelerate.py":
        raise ValueError("The guarded entry must be train_accelerate.py")
    config = Path(os.environ["SURFNA_PAIRED_INIT_CONFIG"]).resolve(strict=True)
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(script.parent))
    dataset = importlib.import_module("datasets.pdbbind").PDBBind
    for name in ("preprocessing", "inference_preprocessing"):
        if not callable(getattr(dataset, name, None)):
            raise RuntimeError(f"Cannot install frozen-cache guard: PDBBind.{name} is missing")
        setattr(dataset, name, deny_rebuild)
    # The frozen entry creates its global Accelerator only when executed as
    # ``__main__``.  Patch the factory in its defining module, then execute
    # the unmodified entry with runpy so that lifecycle remains identical to
    # the validated DDP launcher.
    model_utils = importlib.import_module("utils.utils")
    original_get_model = model_utils.get_model
    def initialized_model(*model_args, **model_kwargs):
        model = original_get_model(*model_args, **model_kwargs)
        apply_paired_initialization(model, config)
        return model
    model_utils.get_model = initialized_model
    previous = sys.argv[:]
    try:
        sys.argv = [str(script), *args[1:]]
        runpy.run_path(str(script), run_name="__main__")
    finally:
        model_utils.get_model = original_get_model
        sys.argv = previous


if __name__ == "__main__":
    main()
