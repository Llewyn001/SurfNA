# Paper reproduction map

The lightweight release provides final inference checkpoints, the PDB/sample selection manifests, numerical configurations, method implementations and Figure 2–4 numeric source tables. Download and prepare the structures locally. Generated pose banks, Scorer training arrays, prepared graph/surface datasets and intermediate training weights are not distributed in this release.

| Result | Protocol | Public material |
|---|---|---|
| Figure 2 main benchmark | seed0 EMA350, seed1 EMA500, seed2 EMA400; frozen native MDN; the same final W0; K40; 20 iterations | All three Generator checkpoints, MDN/W0, test128 PDB selections, ordered membership, per-complex and summary tables |
| Figure 3 fixed-pool ranking | Rank the same candidate ensembles with alternative scorers | W0 inference code; archived numeric comparison tables. Historical pose pools and alternative scorer dependencies are not bundled. |
| Figure 3 data efficiency | Two transfer conditions × 80/200/400/801 examples × three seeds; fixed EMA200; K10 | Nested subsets, sampling weights, transfer code, training parameters, sampling implementation and numeric results; the 24 additional model mappings are recorded for provenance, with weights outside this lightweight release. |
| Figure 4 transfer and intervention | Three arms, three seeds, epochs 25–200; paired noisy states and surface chemistry interventions | Method/analysis source, numerical configurations and summary source data; all epoch checkpoints and fixed-state archives are outside this lightweight release. |

`reproducibility/checkpoint_registry.json` distinguishes public inference assets from local historical checkpoints. A recorded hash for a historical checkpoint does not imply that its bytes are included in this repository or Release.

## Default benchmark

Follow `README.md` for preparation, generation and ranking. The model configuration, 20-step schedule and W0 objective match the recorded implementation. The newly assembled PDB downloader and full preparation-to-inference workflow have not been executed during this publication pass. Regenerating the historical complete sampling stream bit for bit has not been established; original input preparation, software versions and random-number consumption can affect results.

The four benchmark memberships overlap. Keep their reported memberships and denominators; do not treat their sizes as independent samples. Failures count in the fixed denominators. The test is known-site holo redocking.

`scripts/summarize_benchmark.py` reads the released per-complex table to recompute Figure 2 run-wise Top-1/3/5 and oracle endpoints, including mean and sample SD. These archived tables also permit inspection without regenerating every pose.

## Training and transfer

See `docs/TRAINING.md`, `configs/` and the historical source in `reproducibility/reference/`. All target Generator parameters are trainable. Transfer-All initializes all compatible source tensors; Transfer-NonSurface excludes the defined surface path; Scratch-Full8 uses target random initialization. The target training-only surface scaler is reused for all data-efficiency subsets.

The native MDN uses native structures and a frozen source backbone with an adapted prior head. W0 uses K40 ensembles, train801/val89, a fixed train-only scaler, 50 epochs and validation selection; the selected checkpoint's epoch index is 2. Feature extraction code and the Scorer training wrapper are included, but original K40 training arrays are not.

The repository preserves scientific implementation records. Some historical runners require adapting paths and rebuilding their original manifests/inputs; it does not claim a single-command retraining pipeline for all paper experiments.
