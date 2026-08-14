---
name: paper-reading-analysis
description: Analyze computer-vision and machine-learning papers for understanding, reproduction, comparison, and research ideation. Use for papers about weak or point supervision, oriented object detection, remote sensing, Point2RBox, MMRotate, SAM, teacher-student learning, losses, matching, datasets, or when mapping a paper's method into an implementation plan.
---

# Paper Reading Analysis

## Workflow

1. Use the paper and official supplementary material or code as primary evidence when available.
2. Separate author claims, directly observed evidence, and your own inference.
3. Read `references/analysis-template.md` and fill only sections supported by the available source.
4. Trace the method from inputs through supervision, modules, objectives, training, and inference. Record tensor or data contracts when implementation matters.
5. Extract experimental controls: datasets, splits, metrics, baselines, schedule, initialization, augmentations, seeds, and compute.
6. Identify assumptions, missing details, likely reproduction risks, and claims not isolated by ablation.
7. Map the method to Point2RBox/MMRotate only after the paper's method is clear; distinguish direct correspondence from proposed adaptation.
8. Cite page, section, equation, figure, table, or code location for consequential claims when the source permits.

## Evidence rules

- Do not invent equations, implementation details, hyperparameters, results, or author motivations.
- Do not treat the abstract as sufficient evidence for implementation.
- Do not merge details from related papers without labeling the source.
- Do not call an idea novel until relevant prior work has been checked.
- Do not claim that a paper proves a mechanism when results only show correlation.
- Mark unclear or absent details explicitly and propose a verification path.

## Research output

Conclude with:

- the method's essential contribution in one paragraph
- the minimum reproducible pipeline
- the strongest evidence and main limitation
- implementation mapping to relevant project modules/configs
- prioritized experiments that can falsify the proposed adaptation

## Reference

- `references/analysis-template.md`: structured reading and reproduction template.
