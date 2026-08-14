# Ablation Result Reporting

## Table fields

| Run ID | A | B | Seed(s) | Primary metric | Per-class/secondary | Params/FLOPs | Train time | Memory | Status |
|---|---:|---:|---|---:|---|---:|---:|---:|---|

Add mean +/- standard deviation when repeated seeds are available. Show absolute values and deltas from the matching baseline. Mark incomplete or protocol-deviating runs instead of blending them into the main comparison.

## Interpretation checklist

- Does each attribution comparison differ by only one declared factor?
- Is the gain larger than plausible run-to-run variance?
- Is the effect consistent across seeds/classes or concentrated in a subset?
- Does added compute explain part of the gain?
- Was checkpoint selection identical?
- Does the interaction row support or contradict the claimed synergy?

## Claim language

- Use **improves** only for completed comparable evaluations.
- Use **is associated with** when confounding remains.
- Use **suggests** for limited or noisy evidence.
- Use **did not show a reliable gain** for neutral or unstable results.
- Use **not evaluated** when evidence is absent.
