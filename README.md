# SurfNA

**Protein-to-nucleic-acid transfer through a shared molecular surface for ligand pose generation and ranking.**

This is the minimal SurfNA V2 inference release corresponding to the current
manuscript. It generates 40 ligand poses with 20 diffusion steps and ranks the
complete candidate pool with the frozen Scorer. It supports a **user-specified,
known nucleic-acid binding site**. Training datasets, benchmark structures,
Pepper screening data and training pipelines are outside this release.

Repository: [Llewyn001/SurfNA](https://github.com/Llewyn001/SurfNA).

This revision replaces the earlier EMA350 default with the manuscript's fixed-run
validation-selected Generator and repaired Scorer. The earlier distribution
remains in Git history and the `v2.0.0-rc1` release; its weights must not be mixed
with this model set.

## Installation

Linux, Python 3.10 and an NVIDIA CUDA-capable GPU are the tested platform.
Run inference on an allocated GPU if using a shared cluster. CPU execution is
available with `--device cpu` but is much slower. Start from a fresh environment;
the compiled PyG extensions must match the PyTorch/CUDA build.

```bash
git clone https://github.com/Llewyn001/SurfNA.git
cd SurfNA
conda env create -f environment.yml
conda activate surfna-v2
```

The environment pins PyTorch 2.1.2/CUDA 12.1 and PyG 2.5.0. Installation follows
the upstream [PyTorch previous-version instructions](https://pytorch.org/get-started/previous-versions/)
and [PyG binary wheel instructions](https://pytorch-geometric.readthedocs.io/en/2.5.0/install/installation.html).
The CUDA driver must support the CUDA 12.1 wheel. Native Windows/macOS execution
and other dependency versions have not been validated.

## Model weights

The matching model asset is `surfna-v2-minimal-weights.tar.gz`. Install it from
a locally downloaded copy; the installer verifies every file against the pinned
SHA-256 values in `MODEL_CARD.json` before installing anything.

```bash
python scripts/install_weights.py --archive /path/to/surfna-v2-minimal-weights.tar.gz
python scripts/verify_weights.py
```

An HTTPS release-asset URL can also be supplied with `--url`. Model download
locations are listed on the repository's Releases page when the matching release
is published. The prepared local and server workspaces already contain these weights.

| Component | Role | Size | SHA-256 prefix |
|---|---|---:|---|
| Generator | Validation-selected EMA, fixed 500-epoch run | 76.4 MB | `f15d3e88ade7` |
| Scorer | Validation-frozen repaired W0 head | 0.26 MB | `101133f99c29` |
| MDN static feature model | Frozen nearest-surface distance evidence | 38.1 MB | `8971ea50d6e7` |

The Scorer includes its fitted feature-normalization buffers. Checkpoint containers
contain model tensors only; training-file manifests and run metadata have been
removed without changing any model tensor. `MODEL_CARD.json` records both the
original selected-file hashes and the public-file hashes. The small surface
normalizer and MDN reference-distribution JSON files are model parameters, not
structural datasets. All three weight files are needed for generation **and**
ranking. Historical V1/Pepper and earlier V2 weights are not interchangeable with
this release. “Best” identifies the paper's validation-selected model; no new
test-set model selection is performed for this release.

Build the data-independent diffusion lookup tables once before the first run:

```bash
python scripts/precompute.py
python scripts/check_models.py
```

This CPU initialization can take several minutes and writes approximately
432 MB of probability tables under `.cache/diffusion/`. Set `SURFNA_CACHE_DIR`
to a writable location to reuse the cache across runs. These mathematical tables
contain no molecular data. They are not bundled in the source or model archives.

## Inputs

Supply your own three files in the **same Angstrom coordinate frame**:

| File | Requirement |
|---|---|
| Receptor PDB | Prepared RNA/DNA pocket context around the known binding site; preserve residue and atom names |
| Ligand SDF | Exactly one chemically valid 3D starting conformer placed near that site; preserve stereochemistry and protonation |
| Surface PLY | Triangular Surface-v2 mesh of that site, with raw full8 properties |

The PLY vertex properties must include `x, y, z, hbond, hphob, charge, si, donor,
acceptor, apolar, boundary`. Use raw APBS electrostatic values, not an already
normalized charge field. The supplied frozen normalizer is applied exactly once.
The old four-property surface format is incompatible with these V2 weights.

No crystal pose, RMSD labels or affinity values are needed. The input ligand
provides chemistry and a starting conformer; translation, rotation and rotatable
torsions are randomized before diffusion. The known site and supplied surface
define the search region. The model treats the receptor coordinates as fixed.

### Optional: generate a Surface-v2 mesh

If a compatible PLY is already available, skip this section. Otherwise the
original strict Surface-v2 preparation scripts are included in `surface/`.
Install APBS, PDB2PQR, MSMS and a PyMesh-compatible Python environment separately.
They are external tools and their binaries are not redistributed here.

Arrange your own inputs as:

```text
my_inputs/query/query_protein_processed.pdb
my_inputs/query/query_ligand.sdf
```

The ligand coordinates should identify the known site. For a new ligand,
position an independently generated 3D conformer in the intended binding region.
Do not use a freely placed ligand far from the receptor to define the crop.

```bash
python surface/generate_surfaces_v2.py \
  --data_dir my_inputs --out_dir my_surfaces --target_kind na \
  --complexes query --num_workers 1 \
  --pdb2pqr_bin /path/to/pdb2pqr --apbs_bin /path/to/apbs \
  --msms_bin /path/to/msms --pymesh_python /path/to/pymesh-env/bin/python
```

Default geometry uses a 15 Å residue context, 8 Å surface crop and 30 Å
electrostatic context. The strict pipeline audits heavy-atom coordinate integrity
and refuses missing or invalid surface features. Inspect `surface_v2_audit.csv`;
use the generated pocket PDB and PLY together for docking. The surface tools are
provided for preparation; a fresh installation of all external tools is separate
from the validated inference environment.

## Run generation and ranking

```bash
python scripts/dock.py \
  --receptor /path/to/query_pocket.pdb \
  --ligand /path/to/query_ligand.sdf \
  --surface /path/to/query.ply \
  --output outputs/query --device cuda --batch-size 5 --seed 20260903
```

Use a fresh output directory for each run. Reduce `--batch-size` if GPU memory
is limited. The Scorer always requires the full 40-candidate pool; reducing
the inference batch size changes memory use, not the pool size. Stochastic
sampling, batch size and CUDA kernels may change individual poses across runs;
this single-input interface is not a reproduction of every benchmark random
stream used in the paper.

Outputs:

```text
outputs/query/ranked_poses.sdf        # all 40 poses, best first; score/rank properties
outputs/query/scores/ranking.csv     # scores and original pose indices
outputs/query/poses/query/pose_*.sdf  # original candidate order
outputs/query/prepared/              # prepared graph and copied user inputs
```

**Higher Scorer scores rank better.** Scores are relative within a candidate
pool, not calibrated binding probabilities or physical energies. The ranking
score is `success_logit - 0.25 * predicted_log_rmsd`. The 711 per-atom features
comprise 32 frozen MDN features, 576 Generator interaction features at the clean
timestep and 103 geometry/chemistry features. Feature families run in isolated
processes so the separate frozen model namespaces cannot contaminate one another.

Lower-level commands (`prepare.py`, `generate.py`, `extract_features.py`,
`rank_poses.py`, `rank.py`) are available for debugging and reranking. `graph.pt`
and graph caches are local intermediates generated from your own inputs. Only
load trusted PyTorch/pickle graph files.

## Scope and validation

This release is intended for minimal known-site inference. It does not provide
blind site discovery, receptor flexibility, full training, dataset download or
automatic reproduction of all paper figures. Molecular preparation, ion/tautomer
choices and the supplied binding-site context affect the prediction.

`scripts/check_models.py` tests isolated model loading and Scorer permutation
invariance using synthetic features. The release validation record is in
`VALIDATION.md`; private validation structures and outputs are excluded from the
public package. Dataset publication is deferred independently of the code release.

## License and acknowledgements

The project-specific code and checkpoint license follows the repository's existing
pending status; see `LICENSE_PENDING.md`. Upstream notices are retained in `licenses/`.
See `THIRD_PARTY_NOTICES.md` for inherited SurfDock/DiffDock code and external
dependencies. Please cite the SurfNA manuscript associated with this repository;
final publication metadata will be added when available.
