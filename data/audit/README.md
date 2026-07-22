# SurfNA dataset and model-provenance audit

These files were generated from the frozen SurfNA research workspace by
`scripts/build_dataset_audit.py`.

## Headline counts

- Protein-surface pretraining: 12596 nominal training entries, of which 10,294 cached graphs were actually loaded.
- Released nucleic-acid generator: 2904 nominal entries and 2,904 graphs actually loaded (882 original + 2022 independently PDB-mined high-confidence complexes).
- Released MDN scorer: 1240 nominal entries and 1,240 cached graphs actually loaded from a separate v5 split.
- Fixed generator development set: 132 nominal receptors, of which 128 graphs were actually loaded.
- Final Jiang evaluation: 220 component-set rows over 132 unique PDB receptors.

`model_provenance.csv` ties these counts to the exact public checkpoint hashes,
run identifiers, split names, loader/cache evidence, and best epochs. The 2,904
count applies specifically to the released generator run and must not be read
as the size of every SurfNA development experiment or of the MDN training set.
`mdn_trainval_manifest.csv` separately lists the 1,240 MDN training instances
and 24 validation instances; these correspond to 868 and 14 unique PDB
receptors, respectively.

## Leakage checks

`leakage_audit.csv` tests exact complex identifiers, PDB/base-receptor identity,
and exact full-receptor canonical nucleic-acid sequence signatures. The released
generator training split has zero exact complex-ID and PDB-receptor overlap with
the Jiang receptors. Its sequence audit finds 22 shared
exact signatures and is marked `REVIEW`; corresponding receptors are listed in
`exact_sequence_overlap_details.csv`.

The fixed development set used for generator checkpoint selection contains the
same 132 PDB receptor accessions as the Jiang component-set union
(132 shared). This is model-selection leakage even though
the structures were not used for gradient fitting. A strictly independent final
benchmark requires checkpoint selection on a receptor-disjoint validation set
and regeneration of the reported Jiang results.

The released MDN split is audited separately. It has zero exact complex-ID and
PDB-accession overlap with the development set, but 18
shared exact sequence signatures; those cases are listed in
`mdn_exact_sequence_overlap_details.csv`.

Sequence signatures are chain-order-independent SHA-256 digests. They detect
exact canonical sequence reuse but are **not** a homology-threshold or
structural-similarity audit, and modified nucleotides are ignored. Canonical
sequence extraction was unavailable for 0 train/validation and
0 development receptors.

The development list represents the same 132 receptor accessions underlying
the four Jiang component sets. The component sets contain overlapping receptors
and together contribute 220 evaluation rows.

No coordinate files or private cluster paths are included in the public
manifests.
