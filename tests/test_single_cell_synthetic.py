"""Validation suite for the synthetic single-cell experiment.

Run: ``JAX_PLATFORMS=cpu pytest tests/test_single_cell_synthetic.py``.

Covers the plan's claims with analytic, sign/axis, cross-solver, and
recovery-floor checks (PLAN §4, §13): the exact Hodge split, the Zermelo sign
discipline (with-flow cheaper), directionality monotone in the flux κ, the
LAP↔Randers correspondence (H3), sparseVFC drift recovery under noise, the
mild-wind navigability cap, the dataset contract, and amortized-vs-exact
geodesic fidelity (H4).
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.single_cell_synthetic.baselines import cellrank_fate
from experiments.single_cell_synthetic.drift import SparseVFC, helmholtz_hodge_rbf
from experiments.single_cell_synthetic.evaluate import (
    directionality_score,
    global_cosine,
    mean_cosine,
    path_discrepancy,
)
from experiments.single_cell_synthetic.generator import GeneratorConfig, generate
from experiments.single_cell_synthetic.landscape import (
    Landscape,
    least_action_path,
    om_action,
)
from experiments.single_cell_synthetic.metric import (
    FlatSea,
    build_true_metric,
    navigable_wind_scale,
)
from experiments.single_cell_synthetic.solvers import exact_geodesic, train_rfm

jax.config.update("jax_platform_name", "cpu")

RNG = np.random.default_rng(0)
_GRID = jnp.asarray(RNG.uniform([-2.5, -1.5], [2.5, 1.5], size=(300, 2)), jnp.float32)


# ---------------------------------------------------------------------------
# Ground-truth landscape: the exact Hodge split
# ---------------------------------------------------------------------------
def test_hodge_split_is_exact():
    """grad_part + sol_part must equal the drift bit-for-bit."""
    ls = Landscape(kappa=1.0)
    for p in _GRID[:20]:
        g, s = ls.hodge_split(p)
        assert jnp.allclose(g + s, ls.drift(p), atol=1e-5)


def test_hodge_identities():
    """The solenoidal part is divergence-free; the gradient part is curl-free."""
    ls = Landscape(kappa=1.3)
    div_sol = jax.vmap(lambda p: ls.divergence(p, lambda q: ls.kappa * ls.curl_flux(q)))(_GRID)
    curl_grad = jax.vmap(lambda p: ls.scalar_curl(p, lambda q: -ls.grad_potential(q)))(_GRID)
    assert float(jnp.max(jnp.abs(div_sol))) < 1e-4
    assert float(jnp.max(jnp.abs(curl_grad))) < 1e-4


def test_flux_vanishes_at_kappa_zero():
    ls0 = Landscape(kappa=0.0)
    sol = jax.vmap(lambda p: ls0.hodge_split(p)[1])(_GRID)
    assert float(jnp.max(jnp.abs(sol))) == 0.0


# ---------------------------------------------------------------------------
# Onsager–Machlup least action (H3) + irreversibility
# ---------------------------------------------------------------------------
def test_least_action_forward_cheaper_than_reverse():
    """Forward (down the developmental flow) action << reverse — irreversibility."""
    ls = Landscape(kappa=1.0)
    term = ls.terminal_states()
    x0 = jnp.array([-2.0, 0.0])
    _, sf = least_action_path(ls, x0, term[1], n_steps=24, iters=800)
    _, sr = least_action_path(ls, term[1], x0, n_steps=24, iters=800)
    assert float(sf) < float(sr)
    # the minimiser beats the straight-line action
    straight = om_action(jnp.linspace(x0, term[1], 25), ls, 0.15)
    assert float(sf) < float(straight)


def test_lap_randers_correspondence():
    """The Randers geodesic recovers the true min-action path shape (H3)."""
    ls = Landscape(kappa=1.0)
    term = ls.terminal_states()
    x0 = jnp.array([-2.0, 0.0], jnp.float32)
    metric = build_true_metric(ls, _GRID, margin=0.8)
    geo = exact_geodesic(metric, x0, term[1], n_steps=32)
    lap, _ = least_action_path(ls, x0, term[1], n_steps=32)
    assert path_discrepancy(np.asarray(geo.xs), np.asarray(lap)) < 0.6


# ---------------------------------------------------------------------------
# Zermelo sign discipline + directionality (H1)
# ---------------------------------------------------------------------------
def test_with_flow_is_cheaper():
    """Moving with the drift costs less than against it (sign discipline)."""
    ls = Landscape(kappa=1.0)
    metric = build_true_metric(ls, _GRID, margin=0.8)
    x = jnp.array([0.5, 0.2])
    f = ls.drift(x)
    assert float(metric.metric_fn(x, 0.1 * f)) < float(metric.metric_fn(x, -0.1 * f))


def test_directionality_monotone_in_kappa():
    """On the transverse commitment route, directionality is ~1 at κ=0 and
    increases with κ (only the non-conservative flux can break symmetry there)."""
    TA = jnp.array([0.8, -0.8], jnp.float32)
    TB = jnp.array([0.8, 0.8], jnp.float32)
    dirs = []
    for k in [0.0, 0.5, 1.0, 2.0]:
        m = build_true_metric(Landscape(kappa=k), _GRID, margin=0.8)
        dirs.append(directionality_score(m, TA, TB, n_steps=20))
    assert abs(dirs[0] - 1.0) < 0.05  # symmetric at κ=0
    assert dirs[1] < dirs[2] < dirs[3]  # monotone increasing
    assert dirs[3] > 1.1


# ---------------------------------------------------------------------------
# Mild-wind navigability cap (§5.2)
# ---------------------------------------------------------------------------
def test_navigable_wind_scale_keeps_corridor_navigable():
    ls = Landscape(kappa=2.0)
    sea = FlatSea(2)
    scale = navigable_wind_scale(ls.drift, sea, _GRID, margin=0.85)

    def wnorm(z):
        w = scale * ls.drift(z)
        return jnp.dot(w, jnp.dot(sea(z), w))

    assert float(jnp.max(jax.vmap(wnorm)(_GRID))) < 1.0


# ---------------------------------------------------------------------------
# Drift reconstruction + Hodge flux recovery (H1/H2)
# ---------------------------------------------------------------------------
def test_sparsevfc_recovers_drift_under_noise():
    """sparseVFC recovers the clean drift direction despite noise + flipped arrows."""
    ls = Landscape(kappa=1.0)
    z = np.asarray(_GRID)
    f = np.asarray(jax.vmap(ls.drift)(_GRID))
    v = f + 0.3 * f.std() * RNG.standard_normal(f.shape)
    flip = RNG.random(len(z)) < 0.15
    v[flip] = -v[flip]
    vf = SparseVFC.fit(z, v, n_control=80, reg=1e-3)
    pred = np.asarray(jax.vmap(vf)(_GRID))
    assert mean_cosine(pred, f) > 0.9


def test_flux_recovery_increases_with_kappa():
    """The recovered solenoidal part's cosine to truth grows with κ."""
    cosines = []
    for k in [0.0, 1.0, 2.0]:
        ls = Landscape(kappa=k)
        z = np.asarray(_GRID)
        f = np.asarray(jax.vmap(ls.drift)(_GRID))
        v = f + 0.25 * f.std() * RNG.standard_normal(f.shape)
        vf = SparseVFC.fit(z, v, n_control=80, reg=1e-3)
        hf = helmholtz_hodge_rbf(vf, z, n_control=80, reg=1e-2)
        sol_pred = np.asarray(jax.vmap(hf.sol_part)(_GRID))
        sol_true = np.asarray(jax.vmap(lambda p, ls=ls: ls.hodge_split(p)[1])(_GRID))
        cosines.append(global_cosine(sol_pred, sol_true) if k > 0 else 0.0)
    assert cosines[1] > 0.15
    assert cosines[2] > cosines[1]


def test_hodge_kernels_are_orthogonal_subspaces():
    """The curl-free kernel has zero curl; the div-free kernel zero divergence."""
    ls = Landscape(kappa=1.0)
    z = np.asarray(_GRID)
    f = np.asarray(jax.vmap(ls.drift)(_GRID))
    hf = helmholtz_hodge_rbf(SparseVFC.fit(z, f, n_control=80), z, n_control=80)

    def curl(fn, p):
        J = jax.jacobian(fn)(p)
        return J[1, 0] - J[0, 1]

    def div(fn, p):
        J = jax.jacobian(fn)(p)
        return J[0, 0] + J[1, 1]

    c = jax.vmap(lambda p: curl(hf.grad_part, p))(_GRID)
    d = jax.vmap(lambda p: div(hf.sol_part, p))(_GRID)
    assert float(jnp.max(jnp.abs(c))) < 1e-3
    assert float(jnp.max(jnp.abs(d))) < 1e-3


# ---------------------------------------------------------------------------
# Generator / dataset contract
# ---------------------------------------------------------------------------
def test_dataset_contract():
    ls = Landscape(kappa=1.0)
    ds = generate(ls, GeneratorConfig(n_cells=400, n_clones=80, n_genes=60,
                                      substeps=20, seed=7))
    assert ds.X_counts.shape == (ds.n_cells, 60)
    assert ds.X_counts.dtype.kind == "i" and ds.X_counts.min() >= 0
    assert (ds.X_counts == 0).mean() > 0.0  # dropout produced zeros
    assert ds.X_pca.shape == (ds.n_cells, 2)
    assert ds.velocity_pca.shape == (ds.n_cells, 2)
    assert ds.has_oracle
    assert ds.triples.shape[1] == 3 and ds.triples.shape[0] > 0
    # exact Hodge attached to every cell
    assert np.abs(ds.true_grad + ds.true_sol - ds.true_drift).max() < 1e-4
    assert set(np.unique(ds.time_point)).issubset(set(range(4)))


# ---------------------------------------------------------------------------
# Baselines + amortization (H4)
# ---------------------------------------------------------------------------
def test_cellrank_absorption_is_stochastic():
    ls = Landscape(kappa=1.0)
    ds = generate(ls, GeneratorConfig(n_cells=400, n_clones=80, n_genes=60,
                                      substeps=20, seed=8))
    late = ds.time_point == ds.time_point.max()
    m0 = late & (ds.true_state[:, 1] < 0)
    m1 = late & (ds.true_state[:, 1] > 0)
    ab = cellrank_fate(ds.X_pca, ds.velocity_pca, [m0, m1], k=15)
    assert np.allclose(ab.sum(1), 1.0, atol=1e-3)
    assert ab.min() >= -1e-6 and ab.max() <= 1.0 + 1e-6


def test_amortized_rfm_matches_exact_geodesic():
    """The amortized Randers-Flow-Matching path matches the exact BVP (H4)."""
    ls = Landscape(kappa=1.0)
    metric = build_true_metric(ls, _GRID, margin=0.8)
    term = np.asarray(ls.terminal_states())
    x0 = np.array([-2.0, 0.0], np.float32)
    pairs = np.stack([np.tile(x0, (12, 1)),
                      np.linspace(term[0], term[1], 12)], axis=1).astype(np.float32)
    interp, _hist = train_rfm(metric, pairs, dim=2, key=jax.random.PRNGKey(0),
                             steps=400, batch=12, n_quad=16)
    z0, z1 = jnp.asarray(x0), jnp.asarray(term[1])
    g = exact_geodesic(metric, z0, z1, n_steps=24)
    r = interp.path(z0, z1, n_steps=24)
    assert path_discrepancy(np.asarray(r), np.asarray(g.xs)) < 0.3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
