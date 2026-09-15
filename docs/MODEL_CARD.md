# Models and intended use

The system predicts flexible small-molecule poses in a fixed RNA or DNA receptor with a specified binding site. It uses ligand, receptor-residue and molecular-surface graphs. Eight surface channels describe hydrogen bonding, hydrophobicity, electrostatics, shape, donor, acceptor, apolar character and mesh boundary. Nucleic-acid-aware fusion incorporates base, sugar and phosphate environments.

## Default pipeline

| Component | Released model | Role |
|---|---|---|
| Generator | seed 0, EMA350 | 20-step translation, rotation and torsion diffusion; generate K=40 candidates |
| MDN | target-domain native head, selected epoch 19 | Frozen static representations and nearest-eight distance-distribution evidence |
| W0 | seed 20260831, selected epoch index 2 | 62,786-parameter setwise pose ranking |

Each atom receives 32 MDN features, 576 frozen-Generator interaction features and 103 geometry/chemistry features, concatenated in that order. W0 normalizes with its released training scaler, encodes atoms, pools the mean and maximum, and incorporates three statistics of the MDN baseline across the entire 40-pose ensemble. Its unbounded score is `success_logit − 0.25 × predicted_log_rmsd`. Higher is better; it is not a calibrated binding probability or an affinity estimate.

The Generator checkpoint contains all 463 expected state entries. The MDN static bundle contains the 205 entries used by its frozen static feature path, including the adapted head and distance reference. Unused dynamic scorer layers are not required for this path. `best_head.pt` is retained for training provenance; it cannot replace the static bundle by itself.

The W0 inference export preserves the selected model's parameters. `checkpoints/scorer/provenance.json` records the original selected-checkpoint hash and the released serialization. Original optimizer/training containers remain in the local research archive; the lightweight download includes the inference model and required normalization files.

## Limits

- The evaluation uses native-ligand-defined context and surface crops, a fixed receptor and heavy-atom pose RMSD. It does not establish pocket discovery, induced-fit receptor prediction or prospective affinity ranking.
- W0 requires K=40. Changing K, taking a subset, mixing ligands or changing atom ordering changes or invalidates its inputs.
- Generalizing to new chemistries, sites and conformational states requires additional evaluation. A high pose score alone does not establish binding or fluorescence.
- CUDA graph neighborhoods and the frozen runtime are part of the reproduction environment. CPU feature extraction is not supported.
- Earlier assembly checks covered one real complex. This publication pass did not run new model tests, benchmarks or retraining.
