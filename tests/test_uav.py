"""Validation suite for the UAV energy-ledger experiment (experiments/uav).

Run: ``pytest tests/test_uav.py``. Mirrors the validation ladder in
``spec/uav_energy_gauge_PLAN.md`` §4 — the rungs that exist before real data:

    U2  — estimator math on synthetic flights: exact recovery at zero noise
          (the model is linear-exact for quadratic drag + uniform wind), and
          tolerance recovery under noise + model mismatch; spatial (vortex)
          wind through the recover_form_lsq-style atoms; pooled Schur fit.
    U3  — direction-balance / even-odd orthogonality diagnostics.
    U6  — negative control: a fake east-west potential recovers ≈ 0.

Plus the structural guarantees the ladder rests on: feature parity under
segment reversal, the schema contract, and the directed-energy metric gating a
ledger-free model at ≤ 0. Pure numpy/pandas — no JAX, precision-independent.
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.uav import (
    EVEN_NAMES,
    PowerModel,
    directed_energy_r2,
    direction_balance,
    even_features,
    even_odd_leakage,
    fit_log,
    fit_per_log,
    fit_pooled,
    fleet_predictor,
    fleet_wind_cosine,
    implied_wind,
    implied_wind_field,
    k_consistency,
    k_series,
    make_vortex,
    negative_control,
    odd_features,
    reverse_segments,
    split_segments,
    synthesize_fleet,
    validate_table,
    wind_field_cosine,
)


# ===========================================================================
# Structural guarantees — schema contract and feature parity
# ===========================================================================
def test_schema_contract():
    """validate_table accepts the generator's output and rejects broken tables."""
    df, _ = synthesize_fleet(2, seed=0)
    validate_table(df)

    with pytest.raises(ValueError, match="missing columns"):
        validate_table(df.drop(columns=["energy"]))
    bad = df.copy()
    bad.loc[bad.index[0], "energy"] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        validate_table(bad)
    bad = df.copy()
    bad.loc[bad.index[0], "length"] = 0.0  # shorter than its chord
    with pytest.raises(ValueError, match="chord"):
        validate_table(bad)


def test_feature_parity_under_reversal():
    """Even features are invariant and odd features negate under segment reversal."""
    df, _ = synthesize_fleet(2, seed=1)
    rev = reverse_segments(df)
    np.testing.assert_allclose(even_features(rev), even_features(df), rtol=0, atol=0)
    np.testing.assert_allclose(odd_features(rev), -odd_features(df), rtol=0, atol=0)


# ===========================================================================
# U2 — estimator math on synthetic flights (the bridge to real data)
# ===========================================================================
def test_u2_exact_recovery_zero_noise():
    """Quadratic drag + uniform wind is linear-exact: lstsq recovers everything.

    The whole convention chain in one assert: k = mg(1/η↑+η↓)/2 on dz,
    s = mg(1/η↑−η↓)/2 on |dz|, c₂ on len²/dt, b = −2c₂W on dxy, and the
    |W|² remainder folded into the dt coefficient.
    """
    model = PowerModel()
    W = np.array([3.0, -2.0])
    df, truth = synthesize_fleet(6, model=model, wind=tuple(W), noise=0.0, seed=2)

    fits = fit_per_log(df, ridge=0.0)
    for fit in fits.values():
        assert abs(fit.k - model.k) / model.k < 1e-6
        np.testing.assert_allclose(fit.wind_form, truth.wind_form, rtol=1e-6)
        c = dict(zip(EVEN_NAMES, fit.even))
        assert abs(c["length2_over_dt"] - model.quad_w) / model.quad_w < 1e-6
        assert abs(c["abs_dz"] - model.s_even) / model.s_even < 1e-6
        expected_c0 = model.hover_w + model.quad_w * float(W @ W)
        assert abs(c["dt"] - expected_c0) / expected_c0 < 1e-6
        np.testing.assert_allclose(implied_wind(fit), W, rtol=1e-6)


def test_u2_noisy_recovery_with_model_mismatch():
    """Noise + linear-drag + induced-power mismatch: ledger and wind survive.

    The mismatch terms are even (they depend on speed, not direction), so they
    load the nuisance bracket and leave the odd channel recoverable.
    """
    model = PowerModel(lin_w=1.5, induced_w=30.0)
    W = np.array([4.0, 1.5])
    df, _ = synthesize_fleet(25, model=model, wind=tuple(W), noise=0.02, seed=3)

    fits = fit_per_log(df)
    ks = k_series(fits)
    assert abs(ks.mean() - model.k) / model.k < 0.05
    assert k_consistency(ks) < 0.10  # no mass jitter injected -> tight ledger

    w_hat = np.mean([implied_wind(f) for f in fits.values()], axis=0)
    assert np.dot(w_hat, W) / (np.linalg.norm(w_hat) * np.linalg.norm(W)) > 0.95
    assert abs(np.linalg.norm(w_hat) - np.linalg.norm(W)) / np.linalg.norm(W) < 0.2
    assert fleet_wind_cosine(w_hat, df) > 0.95


def test_u2_pooled_schur_matches_and_pools():
    """The Schur-pooled fit shares the wind, keeps per-log ledgers, and predicts."""
    model = PowerModel()
    W = np.array([2.0, 3.0])
    # n_legs matches the plan's ≥5-min filter (~75+ segments per flight);
    # shorter flights leave the per-log ledger visibly noisier.
    df, truth = synthesize_fleet(
        12, model=model, wind=tuple(W), noise=0.02, mass_cv=0.10,
        n_legs=(10, 16), seed=4
    )

    pooled = fit_pooled(df, share_k=False, wind="uniform")
    # Per-log ledgers track the injected mass jitter log by log (each k̂ comes
    # from one flight's ~50 noisy segments, so ~5-10% estimation noise rides
    # on top of the 10% injected spread).
    k_hat = pooled.k_by_log.sort_index()
    k_true = truth.k_by_log.sort_index()
    assert float(np.max(np.abs(k_hat - k_true) / k_true)) < 0.15
    assert float(np.mean(np.abs(k_hat - k_true) / k_true)) < 0.08
    assert float(np.corrcoef(k_hat, k_true)[0, 1]) > 0.6
    cv_hat, cv_true = k_consistency(k_hat), k_consistency(k_true)
    assert abs(cv_hat - cv_true) < 0.05
    # The shared wind matches the truth in physical units.
    w_hat = implied_wind(pooled)
    assert np.dot(w_hat, W) / (np.linalg.norm(w_hat) * np.linalg.norm(W)) > 0.98

    # Held-out prediction through the pooled model.
    train, test = split_segments(df, test_frac=0.3, seed=0)
    pooled_tr = fit_pooled(train, share_k=False, wind="uniform")
    pred = pooled_tr.predict(test)
    obs = test["energy"].to_numpy(float)
    ss = 1.0 - np.sum((obs - pred) ** 2) / np.sum((obs - obs.mean()) ** 2)
    assert ss > 0.99


def test_u2_spatial_vortex_wind():
    """A rotational wind field is recovered by the RBF form atoms (Stage C core)."""
    vortex = make_vortex(center=(40.0, -30.0), strength=4.0, radius=150.0)
    df, _ = synthesize_fleet(30, wind=vortex, noise=0.01, seed=5)

    g = np.linspace(-250.0, 250.0, 5)
    C1, C2 = np.meshgrid(g, g, indexing="ij")
    centers = np.stack([C1.ravel(), C2.ravel()], axis=-1)
    pooled = fit_pooled(df, wind="spatial", centers=centers, width=180.0)
    w_fn = implied_wind_field(pooled)
    assert wind_field_cosine(w_fn, df) > 0.8


# ===========================================================================
# Directed-energy metric — scores the odd channel, gates a ledger-free model
# ===========================================================================
def test_directed_energy_r2_gates_ledger():
    """Full model scores high; the same nuisance without any odd channel ≤ 0."""
    df, _ = synthesize_fleet(20, wind=(2.0, -1.0), noise=0.02, seed=6)
    train, test = split_segments(df, test_frac=0.4, seed=1)

    full = fit_per_log(train)
    r2_full, n_pairs = directed_energy_r2(test, fleet_predictor(full))
    assert n_pairs > 30
    # Pair differences carry √2 the segment energy noise, so at 2% hover noise
    # the ceiling sits below 1; the Stage B real-data bar is 0.7.
    assert r2_full > 0.75

    class _EvenOnly:
        def __init__(self, fit):
            self.fit = fit

        def predict(self, sub):
            return even_features(sub) @ self.fit.even

    even_only = {lid: _EvenOnly(f) for lid, f in full.items()}
    r2_even, _ = directed_energy_r2(test, fleet_predictor(even_only))
    assert r2_even <= 0.0


# ===========================================================================
# U3 — direction balance and even/odd orthogonality
# ===========================================================================
def test_u3_balance_and_orthogonality():
    """Reversal-balanced sets have exactly zero leakage; imbalance is detected."""
    df, _ = synthesize_fleet(6, wind=(1.0, 1.0), noise=0.02, seed=7)

    balanced = pd.concat([df, reverse_segments(df)], ignore_index=True)
    assert even_odd_leakage(balanced) < 1e-12

    # Every mission is an altitude round trip (up to the dropped partial
    # segments), so the ledger's channel is near-balanced; a climb-only subset
    # is maximally imbalanced and the diagnostic says so.
    bal = direction_balance(df)
    assert float(bal["vertical"].mean()) < 0.15
    climb_only = df[df["dz"] > 1.0]
    bal_climb = direction_balance(climb_only)
    assert float(bal_climb["vertical"].min()) > 0.99
    assert even_odd_leakage(climb_only) > even_odd_leakage(balanced)


# ===========================================================================
# U6 — negative control: a fake east-west potential recovers ≈ 0
# ===========================================================================
def test_u6_negative_control():
    """|k'| on dx stays below 10% of the true ledger k (windless fleet)."""
    model = PowerModel()
    df, _ = synthesize_fleet(20, model=model, wind=(0.0, 0.0), noise=0.02, seed=8)
    k_fake = negative_control(df)
    assert float(np.abs(k_fake).median()) < 0.10 * model.k
    # And the genuine ledger, fitted the same way, is far from zero.
    ks = k_series(fit_per_log(df, odd=("dz",)))
    assert float(ks.median()) > 5.0 * float(np.abs(k_fake).median())


# ===========================================================================
# Estimator plumbing
# ===========================================================================
def test_fit_log_odd_channel_selection():
    """Ledger-only fits ignore wind columns; k stays correct without them."""
    model = PowerModel()
    df, _ = synthesize_fleet(4, model=model, wind=(0.0, 0.0), noise=0.0, seed=9)
    one = df[df["log_id"] == df["log_id"].iloc[0]]
    fit = fit_log(one, odd=("dz",), ridge=0.0)
    assert abs(fit.k - model.k) / model.k < 1e-6
    assert fit.wind_form is None
    with pytest.raises(ValueError, match="unknown odd channels"):
        odd_features(df, ("dz", "speed"))
