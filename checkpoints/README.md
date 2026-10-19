# Matching model files

Install `surfna-v2-minimal-weights.tar.gz` with:

```bash
python scripts/install_weights.py --archive /path/to/surfna-v2-minimal-weights.tar.gz
python scripts/verify_weights.py
```

The archive contains the current validation-selected Generator, repaired Scorer
and required frozen MDN static feature model. `MODEL_CARD.json` pins the exact
public-file hashes and the original selected-checkpoint hashes. Checkpoint run
metadata has been removed without changing model tensors. Weight binaries are
distributed separately from Git source files. The small MDN reference JSON is a
fitted model distribution, not a structural dataset.

Earlier `v2.0.0-rc1` assets use a different Generator and Scorer. They are retained
as historical releases and must not be mixed with this minimal distribution.
