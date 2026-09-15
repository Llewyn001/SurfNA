# Installation

Use Python 3.10 and Linux with an NVIDIA GPU for molecular graph inference. The release was tested with PyTorch 2.1.2 + CUDA 12.1, PyG 2.5.0 and the versions below. A clean installation of these commands has not yet been tested; validation used the recorded existing environment.

```bash
conda create -n surfna-v2 python=3.10.16 -y
conda activate surfna-v2
python -m pip install torch==2.1.2 --index-url https://download.pytorch.org/whl/cu121
python -m pip install torch-scatter==2.1.2+pt21cu121 torch-sparse==0.6.18+pt21cu121 torch-cluster==1.6.3+pt21cu121 -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
python -m pip install -r requirements-runtime.txt
python -m pip install -e . --no-deps
```

Run scripts from the release root. Keep the `runtime/`, `configs/`, `checkpoints/` and `reproducibility/` directories alongside the editable package. A standalone wheel that embeds these assets is not provided.

`configs/environment_validated.json` records the verification environment. `configs/environment_recorded.json` instead records the earlier data-preparation environment (PyTorch 2.2.2); it is not an alternative claim of numerical equivalence.

CUDA is required for Generator and MDN feature extraction. Capped radius-neighborhood construction differs between CPU and CUDA in the frozen graph implementation, and CPU-generated model features did not reproduce the archived ranking. No CPU fallback is made. Applying W0 to already extracted features is supported on CPU and has a separate regression check.

Prepared graph and checkpoint assets contain Python/PyTorch serialization. Load only the release assets after checking their hashes. Surface regeneration additionally needs MSMS, PDB2PQR, APBS and PyMesh; these executables and any separately licensed tools are not bundled. The lightweight release provides PDB identifiers and selection metadata. Readers obtain structures from the PDB and run preparation locally.
