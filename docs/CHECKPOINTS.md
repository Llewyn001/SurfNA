# Checkpoint downloads

The lightweight release distributes selected inference models separately from Git. The default set is sufficient for a new docking run after preparing the input structures. The optional repeat set supplies the other two Generator training repetitions used in Figure 2.

| Asset | Download size | Contents |
|---|---:|---|
| [surfna-v2-default-checkpoints.tar.gz](https://github.com/Llewyn001/SurfNA/releases/download/v2.0.0-rc1/surfna-v2-default-checkpoints.tar.gz) | 106.8 MB | seed0 EMA350 Generator; native MDN static bundle and selected head; W0; distance reference and scalers |
| [surfna-v2-benchmark-repeats-checkpoints.tar.gz](https://github.com/Llewyn001/SurfNA/releases/download/v2.0.0-rc1/surfna-v2-benchmark-repeats-checkpoints.tar.gz) | 141.6 MB | seed1 EMA500 and seed2 EMA400 Generator checkpoints; use the same MDN/W0 as the default set |

Use `python scripts/download_checkpoints.py --set default` and optionally `--set benchmark-repeats`. Each archive expands directly into the repository root. `checkpoints/assets.json` records both archive and individual-file hashes. Do not mix a V1 MDN ranker with the V2 W0 pipeline.

The required SO(3)/torsion numerical lookup tables are generated locally with `python scripts/precompute_diffusion_tables.py`. They are not a dataset and are not included in these checkpoint archives.

The complete checkpoint registry also documents source-pretraining, data-efficiency and intermediate-epoch models from the local research archive. Those entries are explicitly marked as outside this lightweight public release. Their presence in the registry does not mean they can be downloaded here. No neighboring epoch is substituted for them.
