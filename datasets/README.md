# Structure identifiers and sample selections

`nucleic_acid_instances.csv` identifies the 1,018 ligand instances from 699 distinct PDB entries: train801 (499 entries), val89 (72), test128 (128). The PDB entry sets are disjoint across these splits. Different ligand instances in one entry are retained where allowed by the recorded selection criteria.

- `*_pdb_ids.txt`: unique entry identifiers for downloading.
- `*_complexes.txt`: instance names in the released ordering. Formal test sampling order is also recorded in `reproducibility/splits/original/`.
- `nucleic_acid_instances.csv`: split; ligand CCD code, label/auth chain IDs, residue, model and alternate location where recorded; selected receptor chains; chemistry; original prepared-file hashes; benchmark membership.
- `selection_metadata.json.gz`: retained receptor residue tuples `[chain, residue_number, insertion_code, residue_name]`, plus original test-ligand atom mappings and CCD definition hashes. This is coordinate-free metadata, not a structure archive.
- `summary.json`: counts of instances and distinct PDB entries.

Blank model/alternate-location fields for inherited benchmark structures mean the original preparation record did not specify them; they do not assert model 1 or a particular alternate conformer. `ligand_residues_json` supplies the deposited target residue identifiers. Composite ligands may span multiple residues or CCD components.

`reference_*_sha256` values refer to historical prepared structures, not to the complete mmCIF files downloaded from the PDB. Current PDB/CCD files can differ from those used in the original preparation. Download hashes and retrieval dates are recorded separately by the downloader. Do not substitute a replacement PDB ID for an obsolete entry without documenting a revised dataset.

Protein pretraining lists are separately recorded in `reproducibility/splits/protein_generator_split.csv` and `reproducibility/protein_source/`. Raw PDBbind data must be obtained from the provider under its access terms.
