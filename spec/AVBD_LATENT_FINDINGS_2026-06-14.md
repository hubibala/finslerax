# Why AVBD struggles on long latent geodesics — controlled findings (2026-06-14)

**Question.** The AVBD BVP solver recovers a clean geodesic for a ~60° latent
"rotation" but fails for longer paths in high dimension. *Why*, exactly — and
how do we build an honest, rigorous latent-geodesic demo around the real
behaviour?

**Method.** We removed the KDE/EBM training confound by using an *analytic*
ring-confinement conformal metric, where the ground-truth geodesic between two
ring points is the connecting arc in **any** ambient dimension D:

> G(z) = exp(α·c(z))·I,  c(z) = ((r−R)² + |z_⊥|²)/w²,
> r = √(z₀²+z₁²) (radius in the ring plane), z_⊥ = all other coordinates.

Cheap on the ring, expensive off it. Scripts: `spec/avbd_solver_study/diag_avbd*.py`,
`bench_*.py`, `test_vbd_newton.py`, `proto_ae_latent.py`, `build_notebook.py`.

## Findings

**F1 — Dimension is NOT the bottleneck (the headline).**
Fixing arc=120°, N=24, α=4: iterations-to-converge and final path-RMSE are
*identical to 4 decimals* for D ∈ {2, 8, 32, 128, 256} (iters_to_2%=200,
RMSE=0.0267, off-plane=0.0000 throughout). The block-tridiagonal adjoint and the
per-vertex update are D-agnostic in their *dynamics*; D only changes per-iteration
cost and the geometry of the cold-start void. Calling this a "high-dimensional"
difficulty is misleading — **the cost axes are path length N and metric stiffness.**

**F2 — O(N²) critical slowing down (the fundamental limit).**
Relaxing a pure low-frequency bump on a Euclidean path (no confinement confound)
is heat diffusion on the 1D path Laplacian. Sweeps to decay the bump to 5%:

| N        | 8   | 16  | 32   | 64   | 128   |
|----------|-----|-----|------|------|-------|
| τ(sweeps)| 128 | 512 | 2048 | 8192 | 32768 |

Exactly 4× per doubling of N → **slope d log τ / d log N = 2.00**. Mechanism: a
randomized Gauss-Seidel sweep moves information ~1 vertex along the chain, so a
global deformation across N vertices needs O(N²) sweeps. Longer geodesics need
more vertices (to keep discretization density and resolve the longer detour), and
pay quadratically for them.

**F3 — Stiffness-induced divergence (the cliff).**
Cold brute force at a stiff setting (arc=150°, N=32, D=64, α=8, w=0.3) **diverges
within the first 50 sweeps to E≈5×10³³** and stays pinned (gradient clipping
prevents NaN but not the blow-up); final path-RMSE 0.65, max|r−R|=0.95 — fully off
the ring. Root cause: the per-vertex update is a **single fixed-step clipped
gradient step** (block Gauss-Seidel *gradient descent*), not the local Newton /
quadratic solve of textbook Vertex Block Descent. A fixed σ that is stable in the
flat region is unstable in the exp(α·c) void.

**F4 — Energy is a misleading convergence monitor.**
The energy gap is dominated by the *fast radial snap* onto the ring; the *slow
tangential redistribution* (the N² mode) barely moves energy but dominates path
error. Energy reaches a 1% gap almost immediately while the path is still
relaxing. **"Energy converged" ≠ "geodesic found"** — monitor path displacement.

## The principled fix (and why it works)

**Numerical continuation + multilevel coarse-to-fine.** On the hard setting where
cold N=64 diverges (RMSE 0.66):

| strategy                                   | iters | RMSE   | wall |
|--------------------------------------------|-------|--------|------|
| cold N=64                                  | 3000  | 0.66 ✗ | 0.5s |
| multilevel 8→16→32→64 (warm-start)         | 1600  | 0.048  | 2.3s |
| multilevel + α-annealing (full recipe)     | 1300  | 0.016  | 1.6s |

- **Multilevel** is the textbook cure for F2: low-frequency modes are resolved
  cheaply on a coarse path (small N → small N²) and only refined locally when
  upsampled. It converts the O(N²) wall into ~O(N log N).
- **α-annealing** (solve a gentle metric, warm-start a stiffer one) defuses F3:
  every solve starts near its own solution, so the fixed-step update never sees
  the destabilising void.
- Both use the existing `AVBDSolver.solve(init_path=...)` warm-start arg — no
  solver internals change.

## Solver-internals study — second-order steps (scripts `bench_*.py`)

We then asked whether the solver *internals* can be elevated, not just wrapped. The
mechanistic root of F2/F3 is that the per-vertex update is a single **fixed-step
clipped gradient step** (block Gauss-Seidel *gradient descent*) — *not* the Newton
step of textbook Vertex Block Descent. Three candidates were benchmarked:

| approach | iters vs N | stiff cold start | verdict |
|----------|-----------|------------------|---------|
| GS-GD (current) | O(N²) critical slowing | diverges (E~10³³) | cheap/iter; the F2/F3 baseline |
| per-vertex Newton (textbook VBD) | — | **diverges (RMSE 1.16, NaN)** | negative result; local conformal subproblem is non-convex |
| **global block-tridiag Gauss-Newton (LM)** | **N-independent (~27–30 iters, N=16→256)** | stalls cold, robust warm | **ship as second-order option** |

- The **global Gauss-Newton** step assembles the block-tridiagonal Hessian of the
  path energy and solves `(H+μI)dx=-g` via the *same* Thomas sweep AVBD already uses
  in its adjoint. Because the step is global, low-frequency modes are resolved at
  once → the O(N²) wall vanishes (validated: in-basin iters ≈ flat in N *and* in D).
  Cost is O(N·D³)/iter; ~3× the per-iter wall-clock of fused GS-GD at moderate N, so
  it wins at large N / high precision, not on cheap sweeps.
- Why per-vertex Newton fails but global GN works: GN does a **trust-region line
  search on the *total* energy** (LM damping), the globalisation an undamped local
  Newton step lacks. (Recompile gotcha: pass μ as a `jnp` array, not a Python float,
  or `filter_jit` recompiles every line-search trial — 1.2 s → 12 ms per iter.)
- Newton is only *locally* convergent: a cold straight chord that dives into a stiff
  void breaks it too. Continuation remains the required globaliser; GN is the polisher.

## Shipped to the library (no existing test broken; 8 new tests, suite green)

- `ham.solvers.GaussNewtonGeodesic` — fully jit/vmap-friendly LM global Newton solver
  (`lax.fori_loop`; same `Trajectory` interface as `AVBDSolver`).
- `ham.solvers.solve_continuation` + `resample_path` — generic warm-started
  multilevel / annealing driver; stages accept any solver with the `solve(...,
  init_path=)` signature (AVBD or GN).
- Tests: `tests/test_gauss_newton.py`.

## Positioning vs the EBM-metric paper (arXiv:2505.18230, Syrota/Hauberg)

They derive the *same* class of conformal metric (G ∝ E or 1/p from an EBM) and
sidestep the optimisation entirely by **learning a low-parameter neural
interpolant** φ_η that perturbs the straight line, trained per endpoint-pair with
finite differences. That dodges F2/F3 by construction but (a) needs training per
metric, (b) restricts the path to the interpolant's parameterisation. HAM solves
the *full* BVP for *any* metric with no training; the honest price is the N²/
stiffness behaviour above, which continuation + multilevel pay down to seconds.
Their rotated-characters task (0–360°, 64-D autoencoder) is the natural template
for our hero scene.

## Delivered demo — `examples/notebooks/demo_high_dim_latent_geodesics.ipynb`

Rebuilt (`build_notebook.py`) around the honest story, on a **genuine trained
autoencoder** (784→64, rotated asymmetric shapes, recon MSE 7e-5):

1. Learned latent: the rotation orbit is a closed loop (PCA captures 75% in 2-D).
2. Linear interpolation ghosts (double shape).
3. EBM/KDE conformal metric `G=exp(α(E−E0))I` (midpoint ~10¹⁴× stiffer).
4. **Live diagnostics:** F1 N²-slowing (τ=128/512/2048/8192, slope **2.00**),
   F2 D-independence (RMSE 0.0266–0.0268 for D=2→256), F3 divergence (E→10³⁵).
5. **Recovery:** cold AVBD diverges on a 150° rotation (E~10²²); `solve_continuation`
   (multilevel + anneal + GN polish) hugs the data (dist 1.0 vs linear 6.8) in ~4s;
   arc-length sweep shows cold AVBD is erratic, continuation robust everywhere.
6. `GaussNewtonGeodesic` N-independence panel (fixed budget: GN RMSE flat in N, GS
   degrades).
7. Limitations + neural-interpolant contrast + per-vertex-Newton negative result.
