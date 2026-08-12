## What changed

<!-- One or two sentences. What does this do, and why? -->

## Mathematics

<!-- Delete this section if no formula, sign convention, or coordinate
     convention moved. Otherwise: cite the reference or give the derivation,
     and say which test would fail under the old behaviour. -->

## Checks

- [ ] `pytest tests/ -q` passes
- [ ] `JAX_ENABLE_X64=1 pytest tests/ -q` passes
- [ ] `ruff check src/ tests/ examples/` and `ruff format --check src/ tests/ examples/` are clean
- [ ] Shapes, units, and sign conventions are stated in any docstring I touched

## AI assistance

<!-- Disclose it if you used it; that is the policy, not a mark against the PR.
     See CONTRIBUTING.md. State what you used it for and confirm you can
     explain every line you are submitting. -->
