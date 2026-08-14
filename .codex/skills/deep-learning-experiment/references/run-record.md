# Experiment Run Record

Complete the fields that affect interpretation. Mark unavailable fields explicitly instead of silently omitting them.

```yaml
experiment_id:
date:
hypothesis:
baseline_id:
changed_factor:

code:
  repository:
  revision:
  dirty_diff_summary:

configuration:
  config_path:
  resolved_config_path:
  command:
  seed:
  deterministic_flags:

data:
  dataset:
  version_or_hash:
  train_split:
  val_test_split:
  classes:
  annotation_path:
  filtering_or_sampling:

runtime:
  environment_or_container:
  pytorch_cuda_versions:
  mmengine_mmcv_mmdet_mmrotate_versions:
  gpu_model_and_count:
  precision:
  batch_size_per_gpu:
  effective_batch_size:
  work_dir:

initialization:
  checkpoint_path:
  checkpoint_hash:
  load_from_or_resume:

evaluation:
  protocol:
  command:
  primary_metric:
  secondary_metrics:
  checkpoint_selection_rule:

outcome:
  status:
  best_checkpoint:
  metrics:
  runtime_and_memory:
  anomalies:
  deviations_from_plan:
  conclusion:
```

Prefer a content hash or immutable identifier for data and checkpoints when paths can change.
