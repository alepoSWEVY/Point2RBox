---
name: debugging-pytorch
description: Diagnose root causes in PyTorch, MMEngine, MMCV, MMDetection, MMRotate, and Point2RBox training or inference. Use for exceptions, NaN or Inf losses, CUDA out-of-memory, shape/device/dtype mismatches, dataloader failures, config or registry errors, checkpoint incompatibility, distributed hangs, metric regressions, or unstable training.
---

# Debugging PyTorch

## Workflow

1. Preserve the exact error, traceback, command, resolved config, environment, data sample identity, code state, and last known good state.
2. Reproduce with the smallest faithful case. Do not change several variables before reproducing.
3. Classify the failure using `references/debug-playbook.md`.
4. Trace from the first incorrect state, not merely the final exception. Inspect shapes, dtype, device, ranges, finiteness, gradients, coordinate conventions, and config resolution at the nearest boundary.
5. Form one falsifiable hypothesis and choose the cheapest discriminating check.
6. Change one variable at a time. Keep diagnostic instrumentation narrow and removable.
7. Implement a fix only after evidence identifies the root cause or strongly excludes alternatives.
8. Add a targeted regression check when practical, then rerun the original reproduction.
9. Report root cause, evidence, fix, verification, and remaining uncertainty separately.

## Guardrails

- Do not mask errors with broad `try/except`, empty tensors, skipped batches, disabled assertions, or silent fallbacks.
- Do not treat lower batch size as the root-cause fix for a memory leak or unexpected tensor expansion.
- Do not disable mixed precision, distributed training, augmentation, or validation permanently just because doing so hides the symptom.
- Do not assume every CUDA error occurs at the line where it surfaces; account for asynchronous execution.
- Do not declare a metric regression fixed from a successful forward pass.
- Do not leave verbose hooks, anomaly detection, synchronous CUDA mode, or sample dumps enabled in normal training.

## Evidence standard

A root-cause conclusion must explain both the observed failure and why the fix removes it. If only a workaround is available, label it as a workaround and state what remains unknown.

## Reference

- `references/debug-playbook.md`: failure-class routing and targeted checks.
