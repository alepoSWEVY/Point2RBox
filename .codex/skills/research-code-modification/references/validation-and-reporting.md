# Validation and Reporting

## Validation ladder

Run the smallest relevant levels first and stop on failure:

1. Syntax, import, and config parsing
2. Targeted unit or regression test
3. Synthetic tensor forward/loss/backward check
4. One-batch data-pipeline and model smoke test
5. Short training/evaluation smoke run
6. Full experiment only when requested and resources permit

For geometry changes, include representative empty, boundary, rotated, and multi-scale cases. For loss changes, check finite values and gradients. For distributed changes, do not infer multi-GPU correctness from a single-GPU run.

## Evidence labels

- **Verified:** directly observed by a completed check.
- **Partially verified:** a narrower proxy passed, but the target condition was not fully exercised.
- **Not verified:** not run or unavailable.
- **Hypothesis:** expected behavior that still needs measurement.

## Completion format

Report:

- Changed: files and behavior
- Verified: exact checks and results
- Not verified: full training, target dataset, multi-GPU, or other missing evidence
- Risks/assumptions: compatibility or experimental caveats
- Next step: only when a concrete follow-up is needed

Never turn a smoke test into a claim of metric gain, convergence, robustness, or speedup.
