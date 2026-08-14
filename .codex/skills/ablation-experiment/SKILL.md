---
name: ablation-experiment
description: Design, execute, audit, or summarize controlled ablation studies for machine-learning research. Use when isolating model components, loss terms, hyperparameters, SAM or pseudo-label modules, augmentation choices, teacher-student mechanisms, or preparing trustworthy ablation tables for a paper.
---

# Ablation Experiment

## Workflow

1. Convert each proposed component into a falsifiable hypothesis with a mechanism and primary metric.
2. Freeze the shared baseline: code, data, schedule, initialization, evaluator, compute budget, seed policy, and checkpoint selection.
3. Read `references/design-template.md` and define factors, levels, controls, and the minimum experiment matrix.
4. Prefer one-factor-at-a-time comparisons for attribution. Add interaction rows only when the research question requires them.
5. Assign unique config names, run IDs, and work directories. Never reuse result directories.
6. Randomize or balance execution order when machine load or time could bias results.
7. Record failed and partial runs; do not silently remove them.
8. Use repeated seeds when expected gains are close to normal run-to-run variance.
9. Summarize results with `references/result-table.md` and separate observations from explanations.

## Design rules

- Change one intended factor per attribution comparison.
- Keep training budget equal unless compute is itself the factor; then report both quality and cost.
- Include the untouched baseline and the complete proposed method.
- Include removal studies for components already present in the full method when that better tests necessity.
- Predefine the primary metric and checkpoint selection rule.
- Do not tune the baseline less carefully than the proposed method.
- Do not infer synergy without testing the relevant interaction.
- Do not claim causality when multiple factors changed together.
- Do not hide negative, neutral, unstable, or failed results.

## Point2RBox-oriented checks

- Isolate candidate generation, matching/selection, loss weighting, teacher updates, and inference changes.
- Keep annotation regime, SAM masks/features, preprocessing cache, and proposal filtering fixed unless they are the tested factor.
- Track overall mAP/AP50 and per-class AP when class sensitivity is plausible.
- Report parameter count, FLOPs or latency, memory, and preprocessing cost when the component adds meaningful computation.

## References

- `references/design-template.md`: hypothesis and matrix design.
- `references/result-table.md`: reporting and interpretation template.
