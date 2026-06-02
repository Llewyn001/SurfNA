# Checkpoints

The default SurfNA inference weights are included here. You can replace them or
pass absolute paths through the run scripts.

Expected generator directory:

```text
generator/
  model_parameters.yml
  best_inference_epoch_model.pt
```

Expected MDN scorer directory:

```text
mdn_scorer/
  model_parameters.yml
  best_model.pt
```

The released inference command uses the generator to sample poses and the MDN
scorer to rank them. The current paper default is 40 poses per complex.
