# PyTorch/MMRotate Debug Playbook

Choose the branch matching the first reliable symptom.

## Config or registry

- Print or save the resolved config and command-line overrides.
- Verify custom module imports and exact registry/type names.
- Check installed MMEngine/MMCV/MMDetection/MMRotate compatibility.
- Distinguish missing config keys from keys ignored by the active class.

## Data pipeline

- Fetch one deterministic sample with worker count zero.
- Inspect keys, shapes, dtype, ranges, empty annotations, class IDs, and metadata.
- Visualize transformed boxes/masks when geometry is involved.
- Check angle convention, flips, resizing, padding, and coordinate frames.

## Shape, dtype, or device

- Log contracts immediately before the failing operation.
- Check broadcasting, batch/proposal dimensions, indexing, contiguity, and empty tensors.
- Check CPU/GPU placement, autocast promotion, and integer-versus-float inputs.

## NaN/Inf or unstable loss

- Find the first non-finite activation, target, loss component, or gradient.
- Check denominators, logs, square roots, exponentials, normalization counts, invalid boxes, and empty matches.
- Compare FP32 and mixed precision only as a diagnostic.
- Inspect each loss term and gradient norm before clipping.
- Use anomaly detection briefly on the minimal reproduction.

## CUDA out of memory

- Record allocated/reserved/peak memory at consistent steps.
- Determine whether failure is immediate, input-size dependent, or grows by iteration.
- Check retained graphs, stored tensors with gradients, repeated model copies, proposal explosion, and evaluation accumulation.
- Use batch-size reduction only after identifying whether capacity or leakage is responsible.

## Distributed failure or hang

- Identify the last collective reached by each rank.
- Check rank-dependent branches, uneven dataloader lengths, skipped backward paths, and unused parameters.
- Reproduce on one GPU to separate model logic from collective behavior, without treating success as proof of distributed correctness.

## Metric regression

- Verify evaluator, dataset split, class order, checkpoint, inference config, and post-processing first.
- Compare predictions on identical samples between last-good and current code.
- Bisect config/code deltas and separate training regression from evaluation regression.

## Checkpoint loading

- Distinguish intentional new/missing keys from accidental name or shape drift.
- Inspect key prefixes, class count, head dimensions, and strictness.
- Do not suppress mismatch warnings without explaining each mismatch.

## Final diagnosis record

Record reproduction, first bad state, hypothesis, discriminating evidence, root cause, fix/workaround, regression check, and unresolved risks.
