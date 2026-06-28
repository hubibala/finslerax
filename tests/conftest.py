"""Pytest configuration for the HAM test suite.

Numerical precision is controlled by JAX's native ``JAX_ENABLE_X64`` flag (read
at ``import jax``; see ``ham.utils.config``) — nothing is set here, so the suite
adapts to whatever precision the environment selected. CI runs it twice, with
``JAX_ENABLE_X64=0`` and ``=1``, so float64 support cannot silently rot.

Shared precision-aware assertion/tolerance helpers live in ``tests/_precision.py``.
"""

from ham.utils.config import default_dtype, x64_enabled


def pytest_report_header(config):
    """Surface the active precision in the pytest header (handy in CI logs)."""
    mode = "float64 (x64 ON)" if x64_enabled() else "float32 (x64 off)"
    return f"HAM precision: {mode} [default dtype: {default_dtype()}]"
