# Releasing finslerax

Releases are built from a published GitHub release and uploaded to PyPI through
Trusted Publishing. No long-lived PyPI token is stored in the repository.

## One-time setup

Before the first PyPI release, create a pending Trusted Publisher for the
`finslerax` project with these exact values:

- Owner: `hubibala`
- Repository: `finslerax`
- Workflow: `pypi-publish.yml`
- Environment: `pypi`

Create the matching `pypi` environment in the GitHub repository settings. Do
not publish a GitHub release until both sides are configured: publishing the
release starts the upload workflow immediately.

## Release checklist

1. Choose a version newer than every existing tag. Never move or reuse a public
   tag. The historical `v1.0.0` tag belongs to the pre-rename HAM code, so the
   first `finslerax` distribution is `v1.1.0`.
2. Set the same version in `pyproject.toml` and
   `src/finslerax/__init__.py`, then run `uv lock`.
3. Run `uv run ruff check src/ tests/ examples/`, `uv run ruff format --check
   src/ tests/ examples/`, `uv run mypy src/finslerax`, and the test suite in
   both precisions.
4. Run `uv build` and `uvx twine check dist/*`. Inspect the wheel to confirm it
   contains only `finslerax`, metadata, and the license; inspect the source
   archive for the expected package source, tests, README, metadata, and license.
5. Merge the release pull request and wait for every `main` check to pass.
6. Publish a GitHub release from the matching tag, for example `v1.1.0`. The
   publish workflow rejects a tag that differs from the package version.
7. Confirm the workflow succeeded, install the released wheel in a fresh
   environment, and verify `python -c "import finslerax; print(finslerax.__version__)"`.
