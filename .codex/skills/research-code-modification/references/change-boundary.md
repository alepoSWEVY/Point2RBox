# Change Boundary

## Highest priority

Apply the minimal-change principle before elegance, generality, consistency, or future-proofing.

## Before editing

1. Define the requested observable behavior.
2. Locate the narrowest implementation and its direct callers.
3. Inspect local conventions and the resolved config path.
4. Verify actual inputs, outputs, tensor contracts, and registry names.
5. Identify the smallest validation that can fail before the change and pass after it.

## Allowed edits

- Change only files required for the requested behavior.
- Perform a small local refactor only when the task cannot be implemented safely without it.
- Keep defaults unchanged when adding an experimental option, unless changing the default is explicit.
- Explain any necessary expansion beyond the initially expected scope.

## Prohibited expansion

- Do not clean up unrelated style, comments, typing, imports, naming, or directory structure.
- Do not extract helpers or abstractions merely because code could be reused later.
- Do not duplicate a module or config without a clear experiment-isolation purpose.
- Do not silently change loss reduction, normalization, sampling, augmentation, evaluation, or checkpoint-loading semantics.
- Do not guess framework APIs, config keys, output dictionaries, or annotation fields.

## Contract checklist

- Input and output shapes, including batch and proposal dimensions
- dtype, device, autocast, and gradient requirements
- coordinate system, angle convention, units, and scale factors
- distributed reduction and synchronization behavior
- registry/import visibility and config inheritance
- state-dict keys and checkpoint compatibility
