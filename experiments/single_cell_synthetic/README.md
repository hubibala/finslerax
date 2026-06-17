# HAM Synthetic Single-Cell Waddington Geometry

A **ground-truth precursor** to the real Weinreb Randers-metric experiment: a
branching Waddington developmental landscape with a **tunable non-conservative
flux** (the `κ` axis) whose *entire* geometry is known by construction. We embed
it into high-dimensional "gene space" through a nonlinear decoder with
negative-binomial count noise, sample it **destructively** at discrete days
(never the same cell twice), attach **clonal barcodes** and **noisy RNA
velocity**, then run HAM's learned **Randers** (asymmetric Finsler) metric
pipeline on it and measure recovery against truth.

This is a **dimension-/dataset-agnostic frame** (Landscape / Generator / Dataset
/ Embedding / Metric+Drift / Solvers / Baselines / Evaluate). The synthetic
generator is instantiated here; a real `research/weinreb` loader that satisfies
the same `SingleCellDataset` contract is the drop-in next target.

> Design rationale, SOTA anchoring, and the full staged plan live in
> [`spec/single_cell_synthetic_PLAN.md`](../../spec/single_cell_synthetic_PLAN.md).
> Findings & lessons-for-real-data: [`spec/single_cell_synthetic_FINDINGS.md`](../../spec/single_cell_synthetic_FINDINGS.md).

---

## The framing (what is actually novel)

Differentiation is **irreversible** (forward ≠ reprogramming) — the Wang
landscape-**plus-flux** picture: a gradient potential alone is *reversible*; the
*non-conservative* part is the irreversibility. Every single-cell geometry SOTA
we surveyed (Metric Flow Matching, Riemannian pullback, PACE) is **symmetric**,
so HAM's **learned asymmetric Randers drift** is a genuine gap-fill — *if* it can
be shown to capture something a symmetric method provably cannot.

The catch is the **Hodge decomposition** `f = -∇φ + (curl/harmonic)`. A Randers
metric `F(x,v) = √(vᵀH v) + β·v` with an **exact** (gradient) one-form `β = dφ`
only adds an endpoint-dependent cost `∫dφ` — it tilts cost but does **not** bend
geodesic *shape* irreversibly. Only the **non-exact** part of `β` changes
geometry in a way no potential and no symmetric metric can reproduce.

**Consequence baked into the design:** the synthetic ground truth carries a
*tunable* conservative→non-conservative knob `κ`, and the headline is a
**function of `κ`**, not a single number. At `κ=0` the symmetric baseline ties;
as `κ` grows, only the asymmetric drift keeps up.

## The ground truth (known by construction)

| Quantity | Model | Role |
|---|---|---|
| Potential `φ(x)` | `-c·x₁ + b·(x₂⁴/4 - s(x₁)·x₂²/2)`, `s(x₁)=s₀·tanh((x₁-x_b)/w)` | pitchfork: single progenitor well → double fate well past `x_b` |
| Conservative drift | `-∇φ` | reversible roll downhill toward a fate |
| Flux `f_curl(x)` | `∇^⊥ψ`, `ψ` a Gaussian bump at the saddle | divergence-free circulation at the commitment point |
| Total drift | `f = -∇φ + κ·∇^⊥ψ` | the SDE drift; `κ` is the irreversibility knob |
| **Exact Hodge split** | `(-∇φ, κ·∇^⊥ψ)` | scorable ground truth for flux recovery — *no* discrete Helmholtz solve |
| Diffusion | isotropic `D` | sets the Onsager–Machlup action scale |
| Gene map | fixed random smooth MLP `x→rates`, NB counts + dropout | forces the real preprocessing path; PCA can't trivially invert |
| RNA velocity | latent drift pushed through count→PCA, then corrupted | per-dim noise + steady-state bias + sign-flipped arrows (the H2 SNR axis) |

The **mild-wind cap** `‖W‖_H < 1` (navigability) is sized by
`navigable_wind_scale`, exactly as `experiments/marine` does.

## Scientific claims

- **H1 (asymmetry pays).** Learned non-conservative Randers drift recovers the
  true flux, breaks symmetry correctly (directionality `>1` forward), and beats
  the symmetric (`β=0`) and CellRank-Markov baselines — **monotone in `κ`**.
- **H2 (identifiability/collapse).** Recovery is a measurable function of velocity
  SNR, lineage sampling, and latent dim `d`; the pullback base `H=JᵀJ` collapses
  beyond some `d*` while the conformal base degrades gracefully.
- **H3 (LAP↔Randers).** The Randers geodesic recovers the true
  Onsager–Machlup / Freidlin–Wentzell minimum-action path of the known SDE.
- **H4 (amortization/scale).** An amortized interpolant minimizing the *asymmetric
  Randers action* (Randers Flow Matching) matches the exact BVP geodesic at far
  lower per-pair cost — the enabler for real 130k-cell scale.

## Results (reproduced by the scripts)

* **Stage A — solver validation.** On the *true* Randers metric, the AVBD +
  continuation + Gauss–Newton geodesic recovers the analytic Onsager–Machlup
  minimum-action path (path discrepancy **~0.3** basin units) and stays finite
  across path resolutions `N ∈ {8..64}`. On the **transverse commitment route**
  (constant `x₁`, where the gradient is symmetric and *only* the flux can break
  symmetry), directionality `cost(against)/cost(with)` is **exactly 1.0 at κ=0**
  and rises **1.0 → 1.09 → 1.20 → 1.42** with `κ` — the cleanest isolation of the
  non-conservative signal.
* **Stage B — reconstruction & collapse.** sparseVFC recovers the clean drift and
  degrades smoothly with velocity noise (cosine **0.99→0.92** as noise grows) and
  flipped-arrow fraction (**0.99→0.55** at 45% flips), and with fewer cells
  (**0.87→0.99**, 100→1500 cells). The pullback base `H=JᵀJ` becomes sharply
  ill-conditioned as the working latent dim grows — `log₁₀ cond` **0.7→4.8** over
  `d ∈ {2..32}` (the documented Finsler collapse) — while the density-conformal
  base stays at `cond≈1` throughout.
* **Stage C — asymmetry pays (headline).** As `κ` grows `{0,.25,.5,1,2}`, the
  **full** Randers drift's flux-recovery cosine rises monotonically **0.00 → 0.36
  → 0.51 → 0.72 → 0.91** and its directionality **1.02 → 1.86**, while the
  **symmetric** `β=0` control is flat at exactly **1.00** — the Hodge-predicted
  separation. (Fate accuracy is saturated at 1.0 for all methods in this
  well-separated regime; flux recovery and directionality are the discriminating
  signals. See the findings report for the full table and honest qualifications.)
* **Stage D — amortization.** The amortized Randers-Flow-Matching interpolant
  matches the exact AVBD+continuation BVP (path discrepancy **~0.01**) at an
  **N-independent forward cost**, while a single exact solve grows with `N` — the
  scaling argument for real data.

Figures land in `visualizations/`.

## Architecture (the frame)

```
landscape.py   Ground-truth Waddington landscape: φ, tunable curl flux, drift,
               diffusion, EXACT Hodge split, Onsager–Machlup min-action utilities.
generator.py   SDE (Euler–Maruyama), destructive day-snapshots, power-law clonal
               barcodes, nonlinear decoder, NB+dropout counts, noisy RNA velocity.
dataset.py     SingleCellDataset — the contract a real Weinreb loader satisfies.
embedding.py   normalize→log1p→z-score→PCA (+ optional AE decoder for pullback).
drift.py       sparseVFC RKHS reconstruction (EM outlier rejection) + mesh-free
               matrix-valued (Fuselier–Wright) Helmholtz–Hodge decomposition.
metric.py      Randers construction: base arms (flat / pullback JᵀJ / conformal)
               × wind (full f̂ / potential-only / β=0), navigable wind sizing.
solvers.py     Exact BVP geodesic (continuation) + amortized Randers-Flow-Matching.
baselines.py   symmetric-Riemannian, potential-only, CellRank-Markov fate, Dynamo LAP.
evaluate.py    recovery cosine, directionality, lineage, fate, collapse-d, LAP match.
run_stage_{a,b,c,d}_*.py   the four staged studies.
```

## Reproducing

```bash
JAX_PLATFORMS=cpu python -m experiments.single_cell_synthetic.run_stage_a_solver
JAX_PLATFORMS=cpu python -m experiments.single_cell_synthetic.run_stage_b_reconstruct
JAX_PLATFORMS=cpu python -m experiments.single_cell_synthetic.run_stage_c_asymmetry
JAX_PLATFORMS=cpu python -m experiments.single_cell_synthetic.run_stage_d_amortize
JAX_PLATFORMS=cpu pytest tests/test_single_cell_synthetic.py
```

All scales are config-driven (`GeneratorConfig`) and CPU-friendly. On Windows set
`PYTHONIOENCODING=utf-8` so the consoles print Greek/maths in the summaries.

## Honest caveats

- **Mild-wind cap.** The causal clamp distorts truly non-navigable
  (super-vehicle) drift; we keep the corridor navigable and map the regime.
- **Synthetic decoder ≠ real gene regulation.** A random smooth MLP is *not* gene
  programs; conclusions are about **geometry recovery**, not biology. The realism
  is in the count statistics, destructive sampling, and velocity noise.
- **Velocity noise model is a model.** We emulate documented pathologies
  (steady-state bias, flipped arrows), not the full richness of real velocity.
- **The flux signature is route-dependent — by definition.** Directionality on
  the *developmental axis* is dominated by the conservative tilt; the flux's
  irreversibility shows up on routes **tangent to the commitment circulation**
  (constant-`x₁` transverse route). This path-dependence *is* the non-conservative
  signature (a gradient one-form's line integral is path-independent).
- **`κ` is ours.** Real flux fraction is unknown; the `κ`-sweep is a controlled
  demonstration, and the real run must *estimate* `κ`, not assume it.
- **Local minima.** The Randers BVP is non-convex; numbers are local optima from
  warm-started continuation, reported as such.
- **Probabilistic/heteroscedastic velocity supervision is deferred** (cheap
  future add: inverse-variance-weighted drift fitting).
