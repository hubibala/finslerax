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

## Running tests and lint

```bash
python -m pytest tests/ -q                       # full suite (float32)
JAX_ENABLE_X64=1 python -m pytest tests/ -q      # full suite (float64)
ruff check src/ tests/ examples/                 # must be clean
```

CI runs the suite on Python 3.10/3.11 in both precisions plus the lint gate;
a green matrix is required to merge. When iterating on a single module, run
just its test file — the full suite takes several minutes.

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
