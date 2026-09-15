# Data and preprocessing

The released target-domain snapshot contains **801 training, 89 validation and 128 test complexes**. `reproducibility/splits/` contains human-readable CSV lists and the original ordered split files. The four historical benchmark memberships overlap; their sizes must not be summed as independent observations.

## Filtering and duplicate control

The construction audit began with 15,464 PDB entries and 67,123 ligand instances. Source quality filters and exact benchmark-PDB exclusion retained 3,398 candidates. Sequence exclusion removed 1,151. Subsequent ligand/context quality control retained 1,787 instances from 1,287 receptors; structural exclusion removed 811, and 86 surface/graph failures were excluded, leaving 890 graph-ready training/validation instances.

Sequence exclusion used one-to-one multichain global matching, identity normalized by the larger full-chain length, and a 0.70 threshold. Structural exclusion used either directional RMscore ≥0.75 with shorter-receptor coverage ≥0.80. These comparisons address similarity to the benchmark. They are not a claim of 70% sequence disjointness between training and validation.

For the internal split, any shared nonempty exact key linked instances: parent PDB, selected sequence signature, canonical selected-chain sequences, standardized receptor hash, exact ligand-pose key or exact pocket fingerprint. Connected components remained indivisible. The final 338 components were assigned using seed 20260831 and exact subset-sum targeting 10% validation. Training and validation have no shared exact split key. Ligand identity was not an exclusion rule: 22 CCD codes and 25 recorded ligand-SMILES strings occur in both. This is not scaffold-disjoint validation.

The final instance selections are in `datasets/`; exact split-group membership and edges are in `reproducibility/membership/`. Ordered splits and subset weights are retained. Full upstream exclusion ledgers remain in the local research archive and are not bundled in the lightweight repository.

## Obtain and prepare structures

The public release contains identifiers and coordinate-free selection metadata in `datasets/`, rather than the prepared structures, surfaces or graph caches. Download the selected PDB/mmCIF entries and CCD definitions with `scripts/download_pdb.py`; its source URLs follow the [RCSB file download services](https://www.rcsb.org/docs/programmatic-access/file-download-services). Obsolete entries are sought in the original wwPDB obsolete archive without silently switching identifiers.

Use `nucleic_acid_instances.csv` to select each ligand instance and receptor chains; `selection_metadata.json.gz` records retained receptor residues and the inherited test-ligand atom names/order. Preserve the deposited ligand coordinates when applying CCD chemistry. Composite and BIRD ligands require their deposited inter-component connectivity. An ideal CCD conformer is a chemical template, not the native pose. The downloader retrieves source files; it does not itself reproduce the historical benchmark's structure processing.

Write a receptor PDB and coordinate-preserving ligand SDF to `data/prepared/<name>/`, named `<name>_protein_processed.pdb` and `<name>_ligand.sdf`. Surface generation also accepts the historical `<name>_protein.pdb` alias. The original CCD construction functions are preserved under `reproducibility/reference/data_preparation/` for method inspection and adaptation to the downloaded source paths.

Surface generation:

```bash
python surface/generate_surfaces_v2.py \
  --data_dir data/prepared --out_dir data/surfaces --target_kind na \
  --complexes_file datasets/test_complexes.txt \
  --pdb2pqr_bin /path/to/pdb2pqr --apbs_bin /path/to/apbs --msms_bin /path/to/msms
```

This step also requires the mesh-regularization backend described in the surface script. External APBS, PDB2PQR, MSMS and PyMesh tools are installed separately. Then run `scripts/prepare_graphs.py` as shown in the main README. The download, structure preparation, surface calculation and graph construction stages are distinct; no regenerated benchmark inputs are asserted to be byte-identical to the historical preparation.

The target scaler is fit on train801. All data-efficiency fractions reuse it. Charge median is −18.9804, robust scale 18.5898 and clipping range [−5,5]. Feature order is hydrogen-bonding propensity, hydrophobicity, charge, shape index, donor, acceptor, apolar, boundary. The frozen loader supplies shape index in [0,1] because it clips negative values; this differs from the signed mathematical definition.

Native-ligand-conditioned preparation uses a 30 Å receptor context, 15 Å pocket and 8 Å surface crop; target surfaces retain at most 512 ligand-proximal vertices. Native ligand heavy atoms are preserved and checked for CCD completeness. PDB2PQR/AMBER charges and APBS (0.15 M ionic strength), MSMS (probe 1.4 Å; density 4), mesh regularization and nearest-four-atom interpolation supply the surface features. Heavy-atom matching and finite-value checks are required. Tool versions and structure-repair routes can affect regenerated surfaces; prepared surfaces are not redistributed in this lightweight release.

`surface/generate_surfaces_v2.py --help` and the accompanying scripts expose the preparation implementation. External executables are not redistributed. For a new target, supply a prepared receptor, an ordered ligand SDF and its v2_full8 surface. `scripts/prepare_graphs.py` constructs a graph cache from the same named-complex directory convention; new-target preparation has not been validated by the release regression example. Use a new cache directory and inspect the retained membership before generation.

Graphs use centered model coordinates; exported SDFs use the original receptor frame. Raw graph `receptor.center_pos` is global and must be centered once, whereas nucleic anchor coordinates are already centered. Prepared graphs are explicitly marked. Reapplying scaling or centering changes predictions.

The source-protein Generator manifests record 4,622/260/236 training/validation/test identifiers from the inherited PDBbind v2020 processing. Independently recovered protein-MDN manifests record 4,622 training and 260 validation complexes; their lists, training metadata and reference/scaler files are retained in `reproducibility/protein_source/`. Source diffusion and source MDN weight identities are documented in the checkpoint registry; these pretraining weights are not part of the lightweight inference assets. Raw PDBbind data are not bundled. Original data providers' attribution and redistribution terms continue to apply; no dataset-wide license is asserted here.
