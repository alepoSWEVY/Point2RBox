# Evidence and Comparison

## Comparability gate

Before comparing results, verify that runs match on:

- code baseline except the declared factor
- dataset version, split, classes, filtering, and evaluator
- training schedule, optimizer, batch size, precision, and hardware assumptions
- augmentation and image scale
- initialization or pretraining
- checkpoint selection rule
- seed policy and number of repeats

If any item differs, label the comparison confounded and explain the likely effect.

## Evidence hierarchy

1. Completed evaluation on the intended split with saved artifacts
2. Repeated runs with declared seeds and summary statistics
3. One completed run
4. Short-run trend or proxy metric
5. Smoke test
6. Untested hypothesis

Match claim strength to evidence. Use mean and standard deviation for repeated runs. Report failures and excluded runs with reasons. Avoid cherry-picking classes, checkpoints, or seeds after observing results.

## Result statement

State:

- baseline and modified run IDs
- absolute metric values and delta
- repeat count and variability
- compute/runtime difference
- whether the planned protocol was followed
- limitations and unresolved confounders
