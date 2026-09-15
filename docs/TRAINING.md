# Training material

The release separates the paper's frozen training implementations from the portable inference entry points. Training was not rerun during this assembly. No full retraining or new benchmark is required to use the released checkpoints.

## Generator

`runtime/generator/train_accelerate.py` is the frozen target-domain training implementation. `configs/generator.yml` records the target architecture and numerical hyperparameters. The source initialization checkpoint is `reproducibility/checkpoints/protein_source/protein_diffusion_checkpoint.pt`; `configs/paper/` records each experiment's parameters.

The original transfer-initialization source is retained under `reproducibility/reference/transfer/`. It defines the Surface-path module membership, source-compatible tensors, paired random initializations and the three arms. Transfer-All overlays all compatible source tensors; Transfer-NonSurface excludes the defined surface path; Scratch-Full8 retains the random target initialization. All target parameters are trainable. Do not interpret the legacy `restart_key_filter=surface` option as the paper's Transfer-NonSurface condition.

The original scripts may refer to historical absolute paths and immutable experiment manifests. They are archived implementation records. The supported portable inference commands are in the main README; a fully portable training launcher for every historical experiment is not claimed in this candidate.

## MDN

The exact native preparation, frozen-backbone transfer and head training sources are under `reproducibility/reference/native_training/`. `configs/mdn_training.json` gives the numerical recipe and `configs/mdn_architecture.json` the architecture. The protein source checkpoint identities are recorded in the registry; their bytes are outside the lightweight inference release. The target distance reference is `checkpoints/mdn/native_train_reference.json` and the selected target head is `checkpoints/mdn/best_head.pt`.

`src/surfna/native_training.py` extracts the original graph-frame, encoding and contact-NLL functions without scheduler/controller dependencies. The original source files retain the complete stage sequence. Ordered train/validation membership and PDB instance selections are supplied; structures, surfaces and graph caches must be rebuilt locally. Use the same target training-only charge and distance references.

## Final W0 Scorer

For Scorer retraining, generate the K40 features and labels in the following expected format. These original arrays remain in the local archive and are not downloaded with this release:

```text
reproducibility/scorer_training/
  features.json                 # name, split, three relative feature paths
  features/mdn32/*.npz
  features/interaction/*.npz
  features/geometry/*.npz
  labels.json                   # name -> RMSD in the same 40-pose order
  candidate_identities.json
```

The 890 groups are the exact 801/89 partition. The original first eight candidates were retained when each group was extended to 40. Their original train-only scaler is provided, rather than fitting a new scaler on all 40 slots.

The assembly includes a portable W0 training wrapper:

```bash
python scripts/train_scorer.py \
  --features reproducibility/scorer_training/features.json \
  --labels reproducibility/scorer_training/labels.json \
  --output runs/w0_retraining --device cuda
```

It uses the fixed W0 objective, original scaler, 50 epochs, group ordering, optimizer/microbatch sizes and validation selection rule. **This wrapper was assembled but has not been run**, following the instruction to stop usability checks. Consult the original frozen trainer under `reproducibility/reference/w0_original/` when auditing exact training behavior. The original selected, initial and last W0 training containers remain in the local research archive. The lightweight release supplies the selected inference export and its provenance.

The default inference export contains only the W0 state and portable provenance. The original checkpoint containers contain additional historical training metadata and have separately recorded hashes.
