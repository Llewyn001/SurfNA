#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

: "${DATA_DIR:?Set DATA_DIR to folders containing <complex>/<complex>_protein.pdb and ligand files}"
: "${OUT_DIR:?Set OUT_DIR to the surface output directory}"
: "${TARGET_KIND:=auto}"
: "${NUM_WORKERS:=8}"
: "${SURFACE_VERTEX_CUTOFF:=8}"
: "${POCKET_RESIDUE_CUTOFF:=13}"
: "${APBS_CONTEXT_CUTOFF:=30}"
: "${TOOLS_DIR:=$ROOT_DIR/tools/transfer}"

python "$ROOT_DIR/surface/generate_surfaces.py" \
  --data_dir "$DATA_DIR" \
  --out_dir "$OUT_DIR" \
  --target_kind "$TARGET_KIND" \
  --surface_vertex_cutoff "$SURFACE_VERTEX_CUTOFF" \
  --pocket_residue_cutoff "$POCKET_RESIDUE_CUTOFF" \
  --apbs_context_cutoff "$APBS_CONTEXT_CUTOFF" \
  --probe_radius 1.4 \
  --density 4.0 \
  --hdensity 4.0 \
  --mesh_res 1.0 \
  --num_workers "$NUM_WORKERS" \
  --tools_dir "$TOOLS_DIR"

python "$ROOT_DIR/surface/check_surface_schema.py" "$OUT_DIR"
