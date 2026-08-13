# Contributing to HAM

Thank you for your interest in HAM. Bug reports, mathematical corrections,
new metrics/manifolds/solvers, and documentation improvements are all welcome.

## Branch model

`main` is the framework: geometry, solvers, training, and the examples. It has
no domain-specific code and no application data dependencies.

Worked applications live on their own branches, each of which is `main` plus one
application subtree:

| Branch | Contains |
| :--- | :--- |
| `app/wildfire` | `experiments/wildfire/`, the `ham.data` raster loaders, the `wildfire` extra |
| `app/robot-arm` | `experiments/arm/` and its Stage-D theory note |

Send library changes to `main`, and application changes to the relevant `app/*`
branch. When `main` moves, the application branches rebase onto it.

## Getting started

```bash
git clone https://github.com/hubibala/HAM.git
cd HAM
pip install -e ".[dev]"      # core + pytest, ruff, matplotlib, jupyter
```

## Quality gates

`main` is protected: changes land through a pull request, and these checks must
be green before it can merge.

| Check | What CI runs | Run it locally |
| :--- | :--- | :--- |
| Lint | `ruff check --no-cache src/ tests/ examples/` | `ruff check src/ tests/ examples/` |
| Format | `ruff format --check --no-cache src/ tests/ examples/` | `ruff format src/ tests/ examples/` |
| Tests | `pytest tests/ -v --timeout=300`, on Python 3.10 and 3.11 × `JAX_ENABLE_X64` ∈ {0, 1} | `pytest tests/ -q` and `JAX_ENABLE_X64=1 pytest tests/ -q` |
| Types | `mypy src/ham` | `mypy src/ham` |
| Build | `uv build` then `twine check dist/*` | `uv build` |

The whole suite takes several minutes, so while iterating on one module, run
just its test file and lint the files you touched. Run the full dual-precision
suite before opening the PR.

To have the lint and format gates run before each commit instead of after the
push:

```bash
pip install pre-commit && pre-commit install
```

## What makes a good contribution

- **Bug reports**: a minimal reproducible example and the observed vs expected
  behaviour. For numerical issues, include the precision (float32/float64) and
  a tolerance-aware comparison.
- **Mathematical changes**: anything touching a formula needs a literature
  reference (or a short derivation in the PR) and a test that would fail under
  the old behaviour. Sign conventions and coordinate conventions must be stated
  explicitly.
- **New geometry** (metrics, manifolds, solvers): implement against the
  `FinslerMetric` / `Manifold` interfaces, validate against a known closed-form
  case where one exists, and add an entry to the API docs.

## Style

- Docstrings: one-line imperative summary first; state shapes, units, and
  sign/coordinate conventions explicitly; use `r"""` whenever a docstring
  contains a backslash; keep `Args:`/`Returns:` blocks only where they add
  signal.
- Comments describe present behaviour and constraints — not edit history.
- Formatting and import order are enforced by ruff; don't hand-format.

## AI-assisted contributions

AI coding tools are welcome here, under the same terms as in the wider
scientific Python ecosystem (NumPy, SciPy, scikit-learn):

- **Disclose** the use of AI assistance in your PR description.
- **Understand what you submit.** You must be able to explain and defend every
  line; "the model wrote it" is not an answer in review.
- **Your effort must exceed the review effort.** Do not open PRs that outsource
  thinking to the maintainers — run the tests, check the math, trim the output.

Low-effort or unverified AI-generated PRs and issues will be closed without
detailed review.

## License

By contributing, you agree that your contributions are licensed under the MIT
License that covers the project.
