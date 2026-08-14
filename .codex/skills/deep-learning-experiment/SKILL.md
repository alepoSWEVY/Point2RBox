---
name: deep-learning-experiment
description: Plan, launch, record, compare, or audit reproducible deep-learning experiments. Use for PyTorch, MMEngine, MMCV, MMDetection, MMRotate, Point2RBox, DOTA, training configs, seeds, datasets, checkpoints, work directories, evaluation metrics, loss curves, or baseline-versus-method comparisons.
---

# Deep Learning Experiment

## Workflow

1. Define the hypothesis, comparison target, success metric, and stopping rule.
2. Establish the baseline identity: code revision or diff, resolved config, dataset split/version, environment, seed, hardware, checkpoint, and evaluation command.
3. Read `references/run-record.md` and create a complete record before a long run.
4. Isolate the experiment in a new config or explicit override and a unique work directory. Preserve the baseline defaults and artifacts.
5. Run preflight checks: config resolution, dataset availability, class mapping, checkpoint compatibility, output path, GPU capacity, and resume/load semantics.
6. Validate in stages: import/config, one batch, loss/backward, short run, then full run.
7. Record the exact command, timestamps, logs, final/best checkpoint selection rule, metrics, failures, and deviations.
8. Compare only like-for-like runs using `references/evidence-and-comparison.md`.

## Non-negotiable rules

- Do not overwrite baseline configs, checkpoints, logs, or work directories.
- Do not compare runs with silent differences in dataset split, schedule, image scale, augmentation, evaluator, seed policy, or checkpoint selection.
- Do not call a run reproducible when the resolved config, code state, seed, environment, and data identity are missing.
- Do not report the best observed result without stating the selection procedure and run count.
- Do not claim an improvement from an incomplete, failed, or incomparable run.
- Separate observed results from interpretation and proposed explanations.

## MMRotate checks

- Record angle convention and version-sensitive transforms.
- Record dataset classes, annotation path, split, filtering rules, and evaluation protocol.
- Save or print the resolved config because `_base_` inheritance and command-line overrides can hide effective values.
- Distinguish `load_from` initialization from `resume` state restoration.
- Record distributed launcher, GPU count, batch size per GPU, gradient accumulation, precision mode, and effective batch size.

## References

- `references/run-record.md`: mandatory experiment record template.
- `references/evidence-and-comparison.md`: comparability and claims policy.
