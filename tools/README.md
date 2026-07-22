# External Surface Tools

SurfNA surface generation expects a `tools/transfer/` directory containing:

- `msms` and `pdb_to_xyzrn`
- APBS 1.5 compatible executable, usually `conda_apbs_1.5/bin/apbs`
- PDB2PQR executable or a conda environment that provides `pdb2pqr`

These third-party binaries are not redistributed because their licenses and
installation methods differ. Install them from their official distributions
and collect the executables under `tools/transfer/`, or point SurfNA at an
existing installation.

You can also point `TOOLS_DIR` at an existing installation when running:

```bash
TOOLS_DIR=/path/to/tools/transfer bash scripts/run_surface_generation.sh
```

The checked-in inference checkpoints do not require these tools when a receptor
surface PLY and pocket PDB have already been prepared.
