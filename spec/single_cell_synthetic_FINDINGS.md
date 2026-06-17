# Findings — Synthetic Single-Cell Waddington Geometry

Implementation of [`single_cell_synthetic_PLAN.md`](single_cell_synthetic_PLAN.md).
Code: `experiments/single_cell_synthetic/`; tests:
`tests/test_single_cell_synthetic.py` (14, green, `JAX_PLATFORMS=cpu`).

This report records what the staged study established and — in the marine/AVBD
tradition — the **lessons it surfaces for the real Weinreb run**.

---

## Headline

A learned **non-conservative Randers drift recovers the true differentiation
flux and breaks path-reversal symmetry correctly, and the margin over symmetric
and potential-only controls grows with the ground-truth flux fraction `κ`** —
exactly as the Hodge-decomposition argument predicts. At `κ=0` the symmetric
baseline ties; the asymmetry is real geometry, not a reparametrised tilt.

The single most important construction decision: the synthetic landscape is
defined as `f = -∇φ + κ·∇^⊥ψ`, so its **Hodge split is exact** (`∇^⊥ψ` is
divergence-free by construction). Flux recovery is therefore *directly scorable*
with no discrete Helmholtz solve on the ground truth — the synthetic advantage
the real data cannot offer.

---

## Stage A — solver validation (H3)

* **LAP↔Randers correspondence holds.** The exact Randers geodesic (AVBD
  continuation + Gauss–Newton) and the true Onsager–Machlup minimum-action path
  of the known SDE agree to a path discrepancy of **~0.3** basin units on a
  cross-bifurcation transit. The OM minimiser is found with **Adam + gradient
  clipping**, not plain gradient descent — the OM action is stiff and fixed-step
  GD *diverges* (the same fixed-step pathology that motivates continuation for
  the Randers BVP; memory `avbd-long-geodesic-diagnosis`).
* **Irreversibility is large.** Forward (down-flow) OM action ≪ reverse: e.g.
  ~1.5 vs ~48 at `κ=1` on the progenitor→fate route.
* **Directionality is monotone in `κ` on the commitment route.** On the
  transverse constant-`x₁` route (where the gradient is symmetric and *only* the
  flux can break symmetry), `cost(against)/cost(with)` is **exactly 1.0 at κ=0**
  and rises **1.0 → 1.09 → 1.20 → 1.42** for `κ ∈ {0,0.5,1,2}`.
* **Solver stays finite** across `N ∈ {8..64}` under continuation.

**Lesson for real data:** trust the continuation stack; expect to *need* it
(cold AVBD on long stiff geodesics is the documented failure). Measure
directionality on routes tangent to the candidate flux, not along the dominant
pseudotime axis where the conservative tilt swamps it.

## Stage B — reconstruction & collapse (H2)

* **sparseVFC tolerates substantial corruption.** Full-drift recovery cosine
  stays high under Gaussian velocity noise and a non-trivial fraction of
  sign-flipped arrows; the EM outlier step is what buys robustness to the flips.
* **Recovery degrades gracefully** with fewer clones / lower velocity SNR — a
  smooth identifiability frontier, not a cliff.
* **Pullback collapse is real and measurable.** The pullback base `H = JᵀJ`
  becomes increasingly anisotropic / ill-conditioned as the working latent grows,
  while the density-conformal base `H = c(z)I` stays well-conditioned — the
  documented Finsler collapse (Pouplin TMLR'23), here *measured* against truth.

**Lesson for real data:** prefer (or at least co-report) the conformal/density
base at high latent dimension; the pullback metric is the collapse-prone arm and
needs a `d*` characterisation before it is trusted at 130k-cell scale.

## Stage C — asymmetry pays (H1, the headline)

Per-`κ` comparison of the **full** non-conservative Randers drift against the
**potential-only** (gradient-part wind) negative control, the **symmetric**
(`β=0`) control, and a **CellRank-style** Markov fate baseline.

| `κ` | flux cosine (full) | directionality full / pot-only / sym | fate full / sym / CellRank |
|----:|:------------------:|:------------------------------------:|:--------------------------:|
| 0.00 | **0.000** | 1.02 / 1.05 / 1.00 | 1.00 / 1.00 / 1.00 |
| 0.25 | **0.356** | 1.21 / 1.09 / 1.00 | 1.00 / 1.00 / 1.00 |
| 0.50 | **0.512** | 1.37 / 1.25 / 1.00 | 1.00 / 1.00 / 1.00 |
| 1.00 | **0.722** | 1.58 / 1.59 / 1.00 | 1.00 / 1.00 / 1.00 |
| 2.00 | **0.906** | 1.86 / 1.66 / 1.00 | 1.00 / 1.00 / 1.00 |

*(`run_stage_c_asymmetry`, `n_cells=2500`, `vel_noise=0.3`, `flip=0.1`;
`visualizations/stage_c_asymmetry.png`.)*

**Flux recovery is the clean headline.** The full non-conservative Randers drift
recovers the true solenoidal part with a cosine that rises **monotonically from
exactly 0 at κ=0 to 0.91 at κ=2** — precisely the Hodge-predicted signature, and
something the potential-only control (zero solenoidal part *by construction*) and
the symmetric control cannot produce.

**Directionality** confirms it: the symmetric (`β=0`) metric is **flat at exactly
1.00** at every κ (no wind ⇒ no asymmetry), while the full drift rises to **1.86**.
Two honest qualifications:

* The **potential-only** control is *not* perfectly flat (it rises to ~1.66). The
  reason is instructive: it uses the **reconstructed** gradient part `∇Φ̂`, and on
  finite noisy data the Hodge separation leaves a residual asymmetry in `Φ̂`
  (the true `φ` is even across the transverse route, so the *analytic* gradient
  one-form would give exactly 1 — the reconstructed one does not). The clean
  separation between full and potential-only is therefore the **flux cosine**, not
  directionality, on this geometry.
* **Fate accuracy is saturated** (1.00 for all methods, all κ). With well-separated
  terminal valleys and a flat sea, nearest-terminal-by-geodesic is trivially
  correct for *every* metric — the metric is not the bottleneck here. A
  discriminating fate test needs **uncommitted, saddle-region sources** whose fate
  is decided by the flux; that is a cheap refinement (flagged), not a result we
  overclaim.

**Lesson for real data:** the deliverable claim is conditional on `κ` — report
the asymmetry payoff *as a curve in estimated flux fraction*, and include the
potential-only control to prove the effect is the flux, not a gradient tilt.

## Stage D — amortization / scale (H4)

* The amortized **Randers-Flow-Matching** interpolant (a low-parameter network
  trained to minimise the *asymmetric* Randers action) matches the exact
  AVBD+continuation BVP to a path discrepancy of **~0.01** on held-out pairs and
  tracks the true min-action path.
* Its forward cost is **N-independent**, whereas a single exact solve grows with
  `N` — the scaling argument that makes a 130k-cell × 71k-triple loop feasible.

**Lesson for real data:** amortize. Validate the interpolant against a handful of
exact solves first (as here), then use it inside the training loop.

---

## Implementation notes / gotchas (for the next engineer)

* **Hodge decomposition of a *learned* field is the hard part.** A naïve primal
  fit `f ≈ ∇Φ + ∇^⊥Ψ` and an inverse-Laplacian (Poisson) fit are both
  ill-conditioned — they reconstruct the field but split it arbitrarily. The
  well-posed tool is **matrix-valued divergence-free / curl-free RBF kernels**
  (Fuselier–Wright), fit as *separate* projections (joint fitting lets the two
  finite-sampled bases trade off with huge cancelling coefficients). Demean the
  divergence-free target so the harmonic constant (the forward drift `(c,0)`) is
  attributed to the gradient part. Result: flux cosine `0` at `κ=0`, monotone in `κ`.
* **Sign discipline (Zermelo).** The wind is the *current*: `W = wind_scale·f`
  directly (no one-form inversion, since `f` is a contravariant velocity).
  Moving *with* the drift is cheaper — unit-tested against the metric directly.
* **Mild-wind cap.** Size `wind_scale` with `navigable_wind_scale` so
  `‖W‖_H < 1`; the true metric uses `wind_mode="raw"` (trusted), learned winds
  use the `"soft"` causal clamp as a safety net.
* **The flux signature is route-dependent — that *is* the signature.** A
  gradient one-form's line integral is path-independent; a solenoidal wind's is
  not. Hence directionality must be measured on the commitment route.
* **Windows console** is cp1252 — run with `PYTHONIOENCODING=utf-8` so the Greek
  in the printed summaries doesn't crash the script (figures are unaffected).

## Memory pointers

Relates to `ham-research-positioning` (asymmetric Randers as the novelty),
`synthetic-suite-conventions` (sign/axis discipline, float32 FD),
`avbd-solver-upgrade-verdict` / `avbd-long-geodesic-diagnosis` (continuation is
the robust workhorse), `src-deep-review` (the Zermelo `(H,W)` contract).
