# Paper Analysis Template

## Citation and scope

- Title, authors, venue/year, version
- Primary sources consulted
- Task and supervision setting

## Problem and contribution

- Problem addressed
- Failure mode of prior methods
- Claimed contributions
- Actual novelty relative to cited closest work

## Method trace

- Inputs and annotations
- Backbone/features
- Candidate or pseudo-label generation
- Assignment/matching/selection
- Losses and weighting
- Training-only components
- Inference path and outputs
- Key equations, shapes, thresholds, and coordinate conventions

## Experimental protocol

- Datasets, splits, classes, and metrics
- Baselines and fairness of comparison
- Initialization, schedule, augmentation, seeds, and hardware
- Main results, ablations, runtime, and memory

## Evidence audit

- Claim -> supporting figure/table/experiment
- Missing control or confounder
- Robustness and statistical evidence
- Negative or boundary cases

## Reproduction map

- Required code/data/checkpoints
- Underspecified details
- Likely version or framework risks
- Minimal reproduction sequence
- Sanity checks and expected intermediate outputs

## Point2RBox/MMRotate adaptation

- Corresponding modules and configs
- Required interface changes
- Preserved baseline behavior
- New hypothesis and falsification experiment
- Expected cost and failure modes

Label every adaptation not stated by the paper as an inference or proposal.
