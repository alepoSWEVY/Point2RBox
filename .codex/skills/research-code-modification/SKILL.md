---
name: research-code-modification
description: Safely modify research code with minimal scope and evidence-based validation. Use for implementing or reviewing changes in PyTorch, MMEngine, MMCV, MMDetection, MMRotate, Point2RBox, model heads, losses, datasets, configs, hooks, or training scripts, especially in a dirty workspace or an experiment repository where reproducibility matters.
---

# Research Code Modification

## Workflow

1. State the exact requested behavior and the smallest likely edit surface.
2. Inspect repository instructions, Git status, relevant diffs, config inheritance, registries, call sites, tensor contracts, and existing tests before editing.
3. Read `references/change-boundary.md` before changing code.
4. Read `references/workspace-safety.md` when the tree is dirty, generated artifacts are present, or Git operations are involved.
5. Trace real definitions for fields, shapes, dtypes, devices, coordinate conventions, and config keys. Never invent an API or output field.
6. Make the smallest coherent change. Preserve public names, registry names, defaults, checkpoint compatibility, and unrelated behavior unless the task requires otherwise.
7. Validate from narrow to broad using `references/validation-and-reporting.md`.
8. Report changed files, verified behavior, unverified items, assumptions, and residual risks.

## Guardrails

- Do not refactor, rename, reformat, reorder imports, or reorganize nearby code for cleanliness.
- Do not modify datasets, configs, model code, and evaluation together unless the requested behavior crosses those boundaries.
- Do not add compatibility fallbacks that hide a broken contract.
- Do not add or upgrade dependencies unless existing project capabilities are insufficient and the user approves the scope.
- Do not overwrite checkpoints, logs, work directories, cached annotations, or user changes.
- Do not claim accuracy, stability, speed, or memory improvement without measurements from a comparable run.
- Treat a successful import, forward pass, or short run as a smoke check, not proof of convergence or metric improvement.

## Research-specific checks

- Preserve tensor shape, dtype, device, gradient flow, and distributed-training assumptions.
- Check angle units/ranges, rotated-box conventions, coordinate frames, image scale factors, and augmentation transforms when geometry changes.
- Check config inheritance and registry resolution before adding new modules or parameters.
- Keep experimental behavior behind an explicit config option when the baseline must remain reproducible.
- Prefer a local assertion or targeted regression test when it protects a discovered contract.

## References

- `references/change-boundary.md`: scope and refactor rules.
- `references/workspace-safety.md`: Git and experiment-artifact safety.
- `references/validation-and-reporting.md`: validation ladder and completion format.
