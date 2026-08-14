# Workspace Safety

Assume modified files, untracked files, checkpoints, logs, cached data, and work directories belong to the user.

## Required behavior

- Inspect Git status and relevant diffs before editing.
- Preserve unrelated changes and adapt around them when safe.
- Stage or commit only when explicitly requested, and include only intended files.
- Treat `work_dirs/`, checkpoints, TensorBoard/W&B logs, generated proposals, pseudo-labels, and cached annotations as experiment records.
- Choose a new explicit output directory for a new experiment.
- Confirm exact paths before deleting, moving, or overwriting artifacts.

## Forbidden behavior

- Do not reset, clean, revert, or discard user work without explicit approval.
- Do not assume the branch or worktree is disposable.
- Do not overwrite a baseline checkpoint or reuse its directory for a modified run.
- Do not regenerate large outputs unless regeneration is part of the task.
- Do not change shared environment or dependency lock files during a narrow fix.

If existing edits conflict with the requested change, inspect and preserve them where possible. Stop and ask only when safe coexistence is impossible.
