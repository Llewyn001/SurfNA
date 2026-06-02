# External Surface Tools

SurfNA surface generation expects a `tools/transfer/` directory containing:

- `msms` and `pdb_to_xyzrn`
- APBS 1.5 compatible executable, usually `conda_apbs_1.5/bin/apbs`
- PDB2PQR executable or a conda environment that provides `pdb2pqr`

The research workspace keeps these binaries under
`/public/home/luoyuxuan/SurfNA/tools/transfer`. They are not copied into the
source release automatically because binary redistribution depends on each
tool's license. For internal runs, copy that directory to this location:

```bash
rsync -a 4090_2:/public/home/luoyuxuan/SurfNA/tools/transfer ./tools/
```

You can also point `TOOLS_DIR` at an existing installation when running:

```bash
TOOLS_DIR=/path/to/tools/transfer bash scripts/run_surface_generation.sh
```
