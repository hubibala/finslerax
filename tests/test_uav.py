"""Validation suite for the UAV energy-ledger experiment (experiments/uav).

Run: ``pytest tests/test_uav.py``. Mirrors the validation ladder in
``spec/uav_energy_gauge_PLAN.md`` §4 — the rungs that exist before real data:

    U0  — ingest integrity: NED→ENU frame/sign round-trip, ∫V·I dt energy
          balance and charge-within-capacity, densified length ≥ chord.
    U1  — segmentation: cruise-band / accel-cap / AGL-floor gates behave and
          the survivorship census is honest.
    U2  — estimator math on synthetic flights: exact recovery at zero noise
          (the model is linear-exact for quadratic drag + uniform wind), and
          tolerance recovery under noise + model mismatch; spatial (vortex)
          wind through the recover_form_lsq-style atoms; pooled Schur fit.
    U3  — direction-balance / even-odd orthogonality diagnostics.
    U6  — negative control: a fake east-west potential recovers ≈ 0.

Plus the structural guarantees the ladder rests on: feature parity under
segment reversal, the schema contract, and the directed-energy metric gating a
ledger-free model at ≤ 0. Pure numpy/pandas — no JAX, precision-independent.
The U0/U1 rungs drive the real ingest pipeline with a synthetic PX4-convention
raw generator (no pyulog, no files — offline and deterministic).
"""

import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from experiments.uav import (
    EVEN_NAMES,
    IngestConfig,
    PowerModel,
    RawFlightSpec,
    align_flight,
    apply_filters,
    directed_energy_r2,
    direction_balance,
    energy_balance,
    even_features,
    even_odd_leakage,
    fit_log,
    fit_per_log,
    fit_pooled,
    fleet_predictor,
    fleet_wind_cosine,
    implied_wind,
    implied_wind_field,
    ingest_flight,
    ingest_logs,
    k_consistency,
    k_series,
    ledger_conditioned,
    load_corpus,
    make_vortex,
    negative_control,
    odd_features,
    reverse_segments,
    segment_track,
    simulate_raw_flight,
    split_segments,
    synthesize_fleet,
    valid_current,
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
# U0 — ingest integrity (frame/sign, energy balance, densified geometry)
# ===========================================================================
def _box_flight(*, log_id="sim", airframe="quad_a", seed=0, **spec_kw):
    """A closed altitude round trip: climb, out-and-back square, descend."""
    wps = np.array([
        [0.0, 0.0, 0.0],     # launch
        [0.0, 0.0, 40.0],    # climb
        [120.0, 0.0, 55.0],  # east + climb
        [120.0, 90.0, 45.0], # north + descend
        [0.0, 90.0, 55.0],   # west + climb
        [0.0, 0.0, 40.0],    # south + descend (back over launch)
        [0.0, 0.0, 0.0],     # land
    ])
    return simulate_raw_flight(
        RawFlightSpec(waypoints=wps, **spec_kw),
        log_id=log_id, airframe=airframe, seed=seed,
    )


def test_u0_ned_enu_frame_roundtrip():
    """align_flight undoes PX4 NED: climb→dz>0, east→+x, north→+y, wind aligned."""
    spec = RawFlightSpec(
        waypoints=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 30.0],   # pure climb
                            [100.0, 0.0, 30.0],                   # pure east
                            [100.0, 60.0, 30.0]]),                # pure north
        wind=(3.0, -2.0),
    )
    raw = simulate_raw_flight(spec)
    track = align_flight(raw)
    # Up is recovered (launch at u=0, climb to +30).
    assert track["u"].iloc[0] == pytest.approx(0.0, abs=1e-6)
    assert track["u"].max() == pytest.approx(30.0, abs=0.5)
    # East leg (interior, away from the corners) increases e with n≈const.
    seg = segment_track(track, raw, IngestConfig(trim_s=0.0, seg_dt=2.0))
    east_leg = seg[(seg["mid_x"] > 20) & (seg["mid_x"] < 80)]
    assert (east_leg["dx"] > 0).all()
    assert east_leg["dy"].abs().max() < 1e-6
    # Wind columns carry (east, north) as given.
    np.testing.assert_allclose(
        [track["wind_e"].iloc[0], track["wind_n"].iloc[0]], [3.0, -2.0], atol=1e-6
    )


def test_u0_climb_has_positive_dz_and_more_energy():
    """A climb segment has dz>0 and outdraws the matched descent (sign of the ledger)."""
    raw = _box_flight(hover_current=20.0, climb_current_gain=5.0)
    seg = segment_track(align_flight(raw), raw, IngestConfig(trim_s=0.0, seg_dt=2.0))
    climb = seg[seg["dz"] > 2.0]
    descend = seg[seg["dz"] < -2.0]
    assert len(climb) > 0 and len(descend) > 0
    # Specific power (energy per second) is higher climbing than descending.
    assert (climb["energy"] / climb["dt"]).mean() > (descend["energy"] / descend["dt"]).mean()


def test_u0_energy_balance_and_capacity():
    """∫V·I dt over segments matches the whole-flight integral; charge ≤ capacity."""
    raw = _box_flight()
    raw.capacity_ah = 10.0
    track = align_flight(raw)
    bal = energy_balance(track)
    # Segment energies sum to the whole-flight integral (trim_s=0, full span).
    seg = segment_track(track, raw, IngestConfig(trim_s=0.0, seg_dt=4.0))
    assert seg["energy"].sum() == pytest.approx(bal["energy_j"], rel=0.02)
    # 20 A baseline over a ~1-2 min flight is well under a 10 Ah pack.
    assert bal["charge_ah"] < raw.capacity_ah
    _, census = ingest_flight(raw, IngestConfig(min_duration=10.0, agl_floor=0.0))
    assert census["charge_within_capacity"]


def test_u0_densified_length_exceeds_chord():
    """Densified segment length is ≥ the straight-line chord (tunneling guard)."""
    raw = _box_flight()
    seg = segment_track(align_flight(raw), raw, IngestConfig(trim_s=0.0, seg_dt=8.0))
    chord = np.sqrt(seg[["dx", "dy", "dz"]].pow(2).sum(axis=1))
    assert (seg["length"] >= chord - 1e-6).all()
    # A segment spanning a corner turn is strictly longer than its chord.
    assert (seg["length"] > chord + 0.1).any()


# ===========================================================================
# U1 — segmentation gates and survivorship census
# ===========================================================================
def test_u1_filters_and_census():
    """Cruise-band, AGL-floor and accel-cap gates each remove the right segments."""
    raw = _box_flight(speed=12.0)
    seg = segment_track(align_flight(raw), raw, IngestConfig(trim_s=0.0, seg_dt=3.0))

    # AGL floor removes the low takeoff/landing segments.
    kept, census = apply_filters(seg, IngestConfig(agl_floor=10.0, cruise_speed=(0.0, 99.0)))
    assert (kept["agl_min"] >= 10.0).all()
    assert 0.0 < census["kept_frac"] <= 1.0
    assert census["pass_agl"] < 1.0  # some low segments existed to drop

    # Cruise band (on 3-D speed) excludes hover and acro while keeping a fast
    # climb — a crafted set makes the three cases explicit.
    crafted = pd.DataFrame({
        "speed": [0.4, 12.0, 30.0],       # hover, cruise/climb, acro
        "accel_max": [0.1, 0.5, 0.5],
        "agl_min": [40.0, 40.0, 40.0],
    })
    kept_c, cen_c = apply_filters(crafted, IngestConfig(cruise_speed=(3.0, 22.0), agl_floor=0.0))
    assert list(kept_c["speed"]) == [12.0]
    assert cen_c["pass_cruise"] == pytest.approx(1 / 3)

    # Census counts are consistent.
    assert census["n_in"] == len(seg)
    assert census["n_out"] == len(kept)


def test_u1_ingest_logs_batch_and_bad_log():
    """ingest_logs pools good flights and records a failed parse without dying."""

    def parser(name):
        if name == "bad":
            raise ValueError("corrupt ulog")
        return _box_flight(log_id=name)

    cfg = IngestConfig(min_duration=10.0, agl_floor=0.0, cruise_speed=(0.0, 99.0))
    pooled, census = ingest_logs(["logA", "logB", "bad"], cfg, parser=parser)

    assert set(pooled["log_id"]) == {"logA", "logB"}
    validate_table(pooled)
    assert (census["log_id"] == "bad").any()
    bad_row = census[census["log_id"] == "bad"].iloc[0]
    assert "corrupt ulog" in str(bad_row["error"])


def test_u1_short_flight_skipped():
    """A flight under min_duration is dropped whole with a reason in the census."""
    short = simulate_raw_flight(
        RawFlightSpec(waypoints=np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 10.0],
                                          [20.0, 0.0, 10.0]]), speed=10.0)
    )
    seg, census = ingest_flight(short, IngestConfig(min_duration=600.0))
    assert len(seg) == 0
    assert census["skipped"] == "too_short"


def test_u1_invalid_current_skipped():
    """A log with no power module (PX4 current_a = -1 sentinel) is skipped, not fit.

    A large fraction of the public corpus lacks a current sensor; the energy
    ledger is meaningless there, so the flight is dropped with a reason.
    """
    raw = _box_flight()
    raw.topics["battery_status"]["current_a"] = -1.0  # the no-power-module sentinel
    assert not valid_current(align_flight(raw))
    seg, census = ingest_flight(raw, IngestConfig(min_duration=10.0, agl_floor=0.0))
    assert len(seg) == 0
    assert census["skipped"] == "invalid_current"


def test_u4_real_fixture_ledger():
    """The Stage-A gates, frozen on REAL PX4 flights (plan §4: U4 on fixture logs).

    ``real_segments.csv`` holds 5 CC-BY flights (3 physical vehicles) from the
    2026-07 corpus pull, preprocessed by this very pipeline. The frozen claims:
    every flight is ledger-conditioned, climb costs more (k>0) in all of them,
    and the deterministic LSQ reproduces the k values recorded at freeze time.
    """
    fix = pd.read_csv(pathlib.Path(__file__).parent / "fixtures" / "uav" / "real_segments.csv")
    validate_table(fix)
    assert fix["log_id"].nunique() == 5
    assert ledger_conditioned(fix).all()

    ks = k_series(fit_per_log(fix))
    assert (ks > 0).all()  # climb costs more, on real hardware
    frozen = {"04fc37c2": 10.387, "0f773ce7": 18.291, "0193dffe": 6.247,
              "02f37fcd": 1.836, "12a68b66": 3.035}
    for lid, k in ks.items():
        np.testing.assert_allclose(k, frozen[str(lid)[:8]], rtol=1e-3)
    # Same-vehicle flights agree far better than the cross-fleet spread.
    same_vehicle = ks[[l for l in ks.index if str(l)[:8] in ("04fc37c2", "0f773ce7")]]
    assert k_consistency(same_vehicle) < 0.5


def test_load_corpus_seam(tmp_path):
    """The real-data seam: a directory of .ulg files loads via the injected parser."""
    for name in ("alpha", "beta"):
        (tmp_path / f"{name}.ulg").write_bytes(b"")  # content ignored by the fake parser
    cfg = IngestConfig(min_duration=10.0, agl_floor=0.0, cruise_speed=(0.0, 99.0))
    df = load_corpus(tmp_path, cfg, parser=lambda p: _box_flight(log_id=p.stem))
    assert set(df["log_id"]) == {"alpha", "beta"}
    validate_table(df)
    # A mis-set path fails loudly rather than silently falling back to synthetic.
    with pytest.raises(FileNotFoundError, match="files under"):
        load_corpus(tmp_path / "empty")


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
    """The ledger term carries the directed-energy signal the even nuisance cannot."""
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
    # The even nuisance keeps at most residual drag correlation (the exact level
    # is data-dependent); the ledger term must dominate the held-out skill.
    assert r2_full - r2_even > 0.3
    assert r2_even < r2_full


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
def test_ledger_conditioned_gate():
    """k is identifiable only where a flight actually changes altitude.

    Real logs are dominated by constant-altitude cruise/hover where the ledger
    coefficient blows up; the gate is decidable from the geometry before any fit.
    """
    df, _ = synthesize_fleet(6, wind=(0.0, 0.0), noise=0.02, n_legs=(10, 16), seed=2)
    assert ledger_conditioned(df).mean() >= 0.5  # climbing missions are identifiable
    flat = df.copy()
    flat["dz"] = 0.05  # a constant-altitude fleet carries no vertical signal
    assert not ledger_conditioned(flat).any()


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
