# Release validation

Validated on 19 October 2026 using Linux, NVIDIA H20, Python 3.10.16,
PyTorch 2.1.2+cu121, PyG 2.5.0 and the package versions pinned in environment.yml.

- Strict isolated Generator loading: passed (19,015,265 parameters).
- Exact MDN static-path tensor loading: passed (205 frozen tensors).
- Strict Scorer loading: passed (62,786 parameters).
- Synthetic candidate permutation-invariance test: passed.
- Single private known-site input through prepare, generate, all three feature
  families and rank: passed, 40 candidates, 20 steps.
- Combined features: 40 x 27 x 711, all finite.
- Final SDF: 40 readable finite poses in descending score order.
- Public checkpoint tensor values: identical to the selected source models.

Raw structures, generated poses, private graph caches and raw logs are excluded
from this distribution. This is an integration smoke test, not a repeat of the
complete paper benchmark. A clean installation from environment.yml and a fresh
installation of the external surface tools have not been tested on a new host.

The source/model archives were also extracted into a fresh directory. Installing
all three model files, verifying hashes, loading every model on CPU and reranking
the saved features passed. All 40 ranks matched the GPU run; maximum score
difference was 7.15255737e-07. This reused the
existing dependency environment, rather than installing a new Conda environment.
