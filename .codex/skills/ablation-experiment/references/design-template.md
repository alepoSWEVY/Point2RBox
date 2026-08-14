# Ablation Design Template

```text
Research question:
Baseline:
Primary metric:
Secondary metrics:
Expected effect size:
Seed policy:
Compute budget:
Checkpoint selection rule:

Factor A:
- Levels:
- Mechanistic hypothesis:
- Exact config/code delta:

Factor B:
- Levels:
- Mechanistic hypothesis:
- Exact config/code delta:

Controls held fixed:
- Dataset/split/evaluator:
- Initialization:
- Schedule/optimizer/batch size:
- Augmentation/image scale:
- Hardware/precision:

Planned rows:
- Baseline
- Baseline + A
- Baseline + B
- Baseline + A + B (only if interaction matters)

Decision rule:
Known confounders:
```

Use the smallest matrix that answers the hypotheses. For many levels, use a screening stage before expensive repeated training. Do not add rows that cannot change the conclusion.
