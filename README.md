# SurfNA Inference Release

SurfNA is a surface-aware docking method for nucleic-acid ligand recognition.
This repository is the lightweight release layout for:

1. receptor surface generation;
2. pose generation and MDN reranking;
3. single-target virtual screening.

Training scripts, raw benchmark outputs, and cluster-specific experiment logs are
not part of this release directory.

## Repository Layout

```text
surface/                 Surface generation and schema checks
src/                     Model, dataset, sampling, inference, and scoring code
src/score_in_place_dataset/
                         Inference-only virtual screening dataset
scripts/                 Release entry points
examples/                Minimal CSV examples
checkpoints/             Place generator and MDN scorer checkpoints here
tools/                   Optional APBS/MSMS/PDB2PQR binaries
```

## Environment

```bash
conda env create -f environment.yml
conda activate surfna-infer
export PYTHONPATH=$PWD/src
```

The exact CUDA/PyTorch versions may need adjustment for your cluster. The
research runs used CUDA-enabled PyTorch, PyG, RDKit, MDAnalysis, APBS, MSMS, and
PDB2PQR.

The SO(3)/torus diffusion lookup tables are not stored in Git. They are
generated automatically the first time `utils.so3` and `utils.torus` are
imported, matching the original DiffDock/SurfDock behavior. To generate them
explicitly before a run:

```bash
PYTHONPATH=$PWD/src python scripts/precompute_diffusion_tables.py
```

## 1. Generate Surfaces

Input complexes are expected as:

```text
data/raw/na/<complex>/<complex>_protein.pdb
data/raw/na/<complex>/<complex>_ligand.sdf
```

Run:

```bash
DATA_DIR=/path/to/data/raw/na \
OUT_DIR=/path/to/data/surfaces/na_4feat_8A \
TARGET_KIND=na \
NUM_WORKERS=8 \
bash scripts/run_surface_generation.sh
```

Each output PLY uses the fixed SurfNA schema:

```text
x, y, z, nx, ny, nz, hbond, hphob, charge, si
```

The surface route follows the current paper code: ligand-proximal pocket
selection, MSMS with `-one_cavity`, mesh regularization at `mesh_res=1.0`, APBS
electrostatics with Amber/PDB2PQR, and an 8 A surface-vertex crop around the
ligand.

## 2. Build an Inference CSV

For a folder of complexes:

```bash
python scripts/prepare_inference_csv.py \
  --data_dir /path/to/data/raw/na \
  --surface_dir /path/to/data/surfaces/na_4feat_8A \
  --out_csv runs/example/input.csv
```

For virtual screening against one target:

```bash
python scripts/prepare_inference_csv.py \
  --protein_path /path/to/receptor.pdb \
  --pocket_path /path/to/surface_dir/target_pocket.pdb \
  --surface_ply /path/to/surface_dir/target.ply \
  --ligand_library /path/to/library.sdf \
  --ref_ligand /path/to/reference_ligand.sdf \
  --out_csv runs/example/virtual_screen.csv
```

Required CSV columns are:

```text
protein_path,pocket_path,ref_ligand,ligand_path,protein_surface
```

## 3. Run Docking Inference

```bash
DATA_CSV=runs/example/input.csv \
MODEL_DIR=checkpoints/generator \
CONFIDENCE_MODEL_DIR=checkpoints/mdn_scorer \
OUT_DIR=runs/example/inference \
SAMPLES_PER_COMPLEX=40 \
SAVE_TOP_N=40 \
bash scripts/run_inference.sh
```

The default release behavior samples 40 poses per complex and ranks them with
the MDN scorer. Output SDFs and confidence CSVs are written under `OUT_DIR`.

## 4. Run Virtual Screening

```bash
PROTEIN_PATH=/path/to/receptor.pdb \
POCKET_PATH=/path/to/target_pocket.pdb \
SURFACE_PLY=/path/to/target.ply \
LIGAND_LIBRARY=/path/to/library.sdf \
MODEL_DIR=checkpoints/generator \
CONFIDENCE_MODEL_DIR=checkpoints/mdn_scorer \
OUT_DIR=runs/virtual_screen_target \
bash scripts/run_virtual_screen.sh
```

Large libraries can be split into multiple CSV ranges with
`--head_index/--tail_index` passed directly to `src/inference_accelerate.py`.

## Notes for the Paper Version

The fixed SurfNA method used in the current manuscript is:

```text
expanded high-confidence generator + official-like MDN scorer + 40 poses
```

This release keeps only the components needed to reproduce inference and
screening once trained checkpoints are supplied.
