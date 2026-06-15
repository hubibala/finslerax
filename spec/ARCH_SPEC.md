# ARCH_SPEC.md — Software Architecture of HAM

**Version:** 1.2.0
**Date:** June 2026
**Dependencies:** JAX, Equinox, Optax

## 1. Design Philosophy

`HAM` (distributed as `hamtools`) is a JAX-native library for learning and
manipulating Finsler geometries. Unlike static differential-geometry libraries
(e.g. `geomstats`), it prioritizes **generative** and **learning** use-cases
where the metric is a neural network rather than an analytical formula.

**Core principles**

1. **Metric-first design.** The `FinslerMetric` object is the single source of
   truth. It defines the cost $F(x,v)$ and the energy $E = \tfrac12 F^2$;
   everything else (geodesics, curvature, transport) is *auto-differentiated*
   from it.
2. **Implicit dynamics.** Christoffel symbols are never written out. The
   geodesic spray is obtained by solving a small linear system built from
   `jax.grad` / `jax.hessian` of the energy.
3. **PyTree-native.** Every metric, solver, and network is an `eqx.Module`,
   hence a JAX PyTree. The same object passes through `jax.jit`, `jax.grad`, and
   `jax.vmap` whether its parameters are constants or trainable weights.
4. **Single-point methods, external batching.** Core methods operate on a single
   $(x, v)$; callers `vmap` over batches. This keeps the math readable and lets
   JAX choose the parallelization.

---

## 2. Core Abstractions

### 2.1 The Manifold (topology)

`Manifold` defines the domain $\mathcal{M}$ and how to stay on it. It does *not*
define distance — that is the metric's job. Concrete manifolds live in
`ham.geometry.manifolds` (`EuclideanSpace`, `Sphere`, `Torus`, `Hyperboloid`,
`Paraboloid`) and `ham.geometry.mesh` (`TriangularMesh`).

```python
class Manifold(eqx.Module):
    @property
    def ambient_dim(self) -> int: ...      # dimension of the embedding space
    @property
    def intrinsic_dim(self) -> int: ...    # dimension of the manifold itself

    def project(self, x):        ...       # ambient point  -> nearest point on M
    def to_tangent(self, x, v):  ...       # ambient vector -> T_x M
    def random_sample(self, key, shape): ...

    def retract(self, x, v):   return self.project(x + v)   # 1st-order exp
    def exp_map(self, x, v):   ...                          # exact exp (defaults to retract)
    def log_map(self, x, y):   ...                          # inverse exp (tangent secant)
    def tangent_norm(self, x, v): ...                       # ambient/Minkowski norm in T_x M
    def tangent_dot(self, x, u, v): ...                     # inner product in T_x M
```

`tangent_norm` / `tangent_dot` give losses and solvers a *manifold-agnostic*
inner product (Euclidean in the ambient space by default, Minkowski on the
hyperboloid), so geometry-aware code never reaches into a specific manifold's
private helpers.

### 2.2 The Finsler Metric (geometry)

`FinslerMetric` is the heart of the library. It is an **`eqx.Module`** (not a
bare ABC), so subclasses are automatically valid PyTrees. A subclass implements
exactly one method — `metric_fn` — and inherits all derived geometry.

```python
class FinslerMetric(eqx.Module):
    manifold: Manifold
    spray_reg: float = eqx.field(static=True, default=PSD_EPS)

    @abstractmethod
    def metric_fn(self, x, v) -> Array:    # the cost F(x, v); 1-homogeneous in v
        ...

    def energy(self, x, v):        return 0.5 * self.metric_fn(x, v) ** 2
    def inner_product(self, x, v, w1, w2): ...   # w1ᵀ g(x,v) w2, g = Hess_v(E)
    def spray(self, x, v):         ...           # geodesic spray Gⁱ (implicit solve)
    def geod_acceleration(self, x, v): return -2.0 * self.spray(x, v)
    def arc_length(self, gamma):   ...           # midpoint-rule path length
```

**Derived geometry, for free:**

| Method | Returns | Mechanism |
| :--- | :--- | :--- |
| `energy` | $E = \tfrac12 F^2$ | algebraic |
| `inner_product` | $w_1^\top g(x,v)\, w_2$ | `jax.hessian(energy)` |
| `spray` | $G^i(x,v)$ | linear solve of the Euler–Lagrange system |
| `geod_acceleration` | $\ddot x = -2G$ | scales the spray |

The spray solve adds a **trace-scaled Tikhonov term** $\varepsilon\cdot
(\mathrm{tr}\,H / D)\,I$ to the velocity-Hessian before inverting, regularizing
near-degenerate directions (e.g. the Randers boundary) without distorting
well-conditioned metrics. See `spec/MATH_SPEC.md` § 6.1.

### 2.3 Asymmetric metrics (Randers base)

`AsymmetricMetric(FinslerMetric)` is the base for all Randers-type metrics. It
adds one abstract method, `zermelo_data(x) -> (H, W, λ)`, returning the
navigation triple (sea metric, wind, causality scalar $\lambda = 1 - \|W\|_H^2$).

Consumers branch on `isinstance(metric, AsymmetricMetric)` to access the wind
field — a `jit`-safe replacement for the fragile `hasattr(..., '_get_zermelo_data')`
duck-typing that earlier code relied on (Python-level `hasattr` cannot be traced
inside `jax.jit`).

---

## 3. The Metric Hierarchy

Concrete metrics live in `ham.geometry.zoo`; learnable ones in `ham.models`.

| Class | Module | `metric_fn(x, v)` |
| :--- | :--- | :--- |
| `Euclidean` | `zoo` | $\|v\|$ |
| `Riemannian` | `zoo` | $\sqrt{v^\top G(x)\, v}$ |
| `Randers` | `zoo` | Zermelo formula (§ 5 of MATH_SPEC) |
| `DiscreteRanders` | `zoo` | anisotropic mesh metric via differentiable face weights |
| `SegmentQuadratureMetric` | `zoo` | Gauss-quadrature segment cost (high-accuracy arc length) |
| `NeuralRanders`, `NeuralRiemannian` | `models.learned` | $H, W$ are neural networks |
| `PullbackRanders`, `PullbackRiemannian` | `models.learned` | $H = J^\top J$ from a decoder Jacobian |
| `DataDrivenPullbackRanders` | `models.learned` | pullback sea + kernel wind from data |
| `EnergyBasedRanders` | `models.learned` | wind from the gradient of a scalar energy field |
| `KernelWindField`, `PseudotimeRanders` | `models.learned` | kernel / pseudotime-driven winds |
| `CovariateConditionedRanders` | `models.wildfire` | wind conditioned on local terrain covariates |

### 3.1 The Randers specialization

`Randers` manages the Zermelo data $(H, W)$ so the result is always a valid,
strongly-convex Finsler norm:

```python
class Randers(AsymmetricMetric):
    def __init__(self, manifold, h_net, w_net, epsilon=1e-5, use_wind=True): ...
    def zermelo_data(self, x):   # symmetrize H, project & causally squash W, return (H, W, λ)
```

The wind is squashed with a **$C^1$ map** applied at *all* magnitudes,
`W_safe = (1-ε)·tanh(‖W‖_H)·W/‖W‖_H`, guaranteeing $\|W_{\text{safe}}\|_H <
1-\varepsilon$ everywhere. An earlier gated squash introduced a discontinuity at
$\|W\|_H = 0.5$ that violated Finsler regularity (review finding **W-RAND**); the
smooth version replaced it.

---

## 4. Solvers and Transport

A geodesic *problem* (the spray ODE / minimizing action) is decoupled from the
*method* used to solve it.

### 4.1 Initial-value: `ExponentialMap` (`solvers/geodesic.py`)

Integrates $\ddot x^i + 2G^i(x,\dot x) = 0$ by RK4. After each composite step the
position is projected back to the manifold and the velocity to its tangent space
to counter numerical drift. `shoot(metric, x0, v0, t_max)` returns the endpoint
$\mathrm{Exp}_{x_0}(t_{\max} v_0)$ memory-efficiently via `lax.fori_loop`.

### 4.2 Boundary-value: `AVBDSolver` (`solvers/avbd.py`)

**Augmented Vertex Block Descent.** Discretizes the path into $N{+}1$ vertices
and minimizes the discrete action $\sum_i E(x_i, v_i)$ via randomized
Gauss–Seidel sweeps; equality constraints $c(x)=0$ are handled with an Augmented
Lagrangian Method. Fully differentiable w.r.t. metric parameters.

- **`parallel=True`** replaces sequential sweeps with a 2-colored Gauss–Seidel
  sweep — even/odd vertices update simultaneously under `vmap` for GPU speedups,
  preserving convergence between color groups.
- **`implicit_diff=True`** uses an O(1)-memory analytical adjoint (the same
  block-tridiagonal structure as § 4.3) for the backward pass.
- **`init_path`** warm-starts from a supplied path, enabling numerical
  continuation (annealing a stiff metric by chaining solves).

Reference: Giles, Diaz & Yuksel, *Augmented Vertex Block Descent*, SIGGRAPH 2025.

### 4.3 Boundary-value, global: `GaussNewtonGeodesic` (`solvers/gauss_newton.py`)

Where AVBD takes *local* gradient steps (and suffers the $O(N^2)$ critical
slowing of 1-D Laplacian relaxation on long/stiff paths), this solver takes a
*global* damped-Newton (Levenberg–Marquardt) step over the whole path. The path
energy has a **block-tridiagonal** Hessian, so each step costs $O(N D^3)$ via the
block-Thomas algorithm, and low-frequency deformations converge in a number of
iterations **independent of $N$**. This is the recommended workhorse for long
latent-space geodesics (see `spec/AVBD_LATENT_FINDINGS_2026-06-14.md`).

### 4.4 Arrival times: the Eikonal family (`solvers/eikonal.py`, …)

For *all-pairs-from-a-source* problems, solving the anisotropic Eikonal PDE is
far cheaper than shooting many geodesics. HAM solves the **dual Randers /
Zermelo arrival-time PDE**

$$(\nabla T - B)^\top G^{-1} (\nabla T - B) = 1, \qquad T(\text{source}) = 0,$$

with $G, B$ derived from the metric's Zermelo data, by the **Fast Sweeping
Method** (alternating directional Gauss–Seidel sweeps to steady state). Three
backends share the Godunov stencil:

| Solver | Domain |
| :--- | :--- |
| `EikonalSolver` | dense 2-D Cartesian grid |
| `VolumetricEikonalSolver` | dense 3-D Cartesian grid (full anisotropic stencil) |
| `MeshEikonalSolver` | unstructured triangulation |

All provide O(1)-memory implicit gradients w.r.t. $(G, B)$ via `jax.custom_vjp`
(an adjoint fixed-point iteration at the converged solution), so an Eikonal solve
can sit inside a training loss (`ArrivalTimeLoss`, `DenseArrivalTimeLoss`).

### 4.5 Supporting solver utilities

- `geodesic_learning.GeodesicLearningSolver` — optimizes the entire path at once
  with Adam (used by energy-based models).
- `continuation` — arc-length resampling/reparametrization and `solve_continuation`
  for parameter homotopy.
- `graph_init` — kNN-graph + Dijkstra warm-starts for BVP solvers in latent space.
- `coloring` — chain / greedy / mesh vertex colorings for parallel sweeps.

### 4.6 Parallel transport: `BerwaldConnection` (`geometry/transport.py`)

Transports a vector along a curve using the **Berwald connection**, the unique
connection induced by the geodesic spray. Its coefficients are the velocity
Hessian of the spray, $^B\Gamma^i_{jk} = \partial^2 G^i / \partial v^j \partial
v^k$, so transport is again an ODE driven by auto-differentiation. See
`spec/MATH_SPEC.md` § 3.

### 4.7 Curvature (`geometry/curvature.py`)

`sectional_curvature`, `flag_curvature_sample` (the Finsler generalization),
`riemann_curvature_tensor`, and `scalar_curvature` — all computed from the
nonlinear connection derived from the spray.

---

## 5. Module Structure

```text
src/ham/
├── geometry/
│   ├── manifold.py          # Manifold ABC (eqx.Module)
│   ├── manifolds/           # EuclideanSpace, Sphere, Torus, Hyperboloid, Paraboloid
│   ├── mesh.py              # TriangularMesh   (mesh_adjacency.py: half-edge adjacency)
│   ├── metric.py            # FinslerMetric & AsymmetricMetric + auto-diff geometry
│   ├── zoo/                 # Euclidean, Riemannian, Randers, DiscreteRanders, SegmentQuadrature
│   ├── transport.py         # BerwaldConnection
│   └── curvature.py         # flag / sectional / Riemann / scalar curvature
├── models/
│   ├── learned.py           # Neural / pullback / energy-based / kernel metrics
│   └── wildfire.py          # CovariateConditionedRanders, LocalTerrainCNN, SPD/‖B‖ projections
├── nn/
│   ├── networks.py          # VectorField, PSDMatrixField, RandomFourierFeatures
│   ├── ebm.py               # ScalarEnergyField, PseudotimePotential, QuadraticHead
│   └── kde.py               # GaussianKDEEnergy
├── solvers/
│   ├── geodesic.py          # ExponentialMap (IVP)
│   ├── avbd.py              # AVBDSolver (BVP, block descent, implicit-diff)
│   ├── gauss_newton.py      # GaussNewtonGeodesic (global BVP)
│   ├── geodesic_learning.py # GeodesicLearningSolver
│   ├── eikonal.py / mesh_eikonal.py / volumetric_eikonal.py   # arrival-time PDEs
│   ├── continuation.py / graph_init.py / coloring.py          # warm-starts, homotopy
├── training/
│   ├── pipeline.py          # HAMPipeline, TrainingPhase
│   ├── losses.py            # geometry-aware loss components
│   └── losses_ebm.py        # contrastive divergence, denoising score matching
├── data/                    # dataset loaders (Weinreb, sim2real)
├── sim/                     # analytic vector fields (Rossby–Haurwitz, vortices)
├── utils/                   # numerics (safe_norm, epsilons), device, config
└── vis/                     # plotting, isosurfaces, marching cubes

examples/        # runnable demo scripts + notebooks
experiments/     # larger applications (marine navigation, …)
spec/            # MATH_SPEC.md, ARCH_SPEC.md, solver studies
tests/           # 333 tests across 32 modules
```

---

## 6. Generative Modeling & Training

The library's headline application is **latent-geometry representation
learning**: a VAE whose latent space carries a *learned Finsler metric*, so that
geodesics in latent space model directed processes (cell differentiation, fronts,
flows).

### 6.1 The generative model contract

Losses and the pipeline operate on a model exposing `encode`, `decode`,
`project_control` (lift an ambient velocity into the latent tangent space), and a
`metric` / `manifold`. A bare metric does **not** satisfy this contract — train a
standalone metric with a plain Equinox/Optax loop instead (see README § 3).

### 6.2 Loss components (`training/losses.py`)

`LossComponent` is an `eqx.Module` with a `weight`, a `name`, and a
`__call__(model, batch, key) -> scalar`. Phases sum the weighted outputs. The
library ships, among others:

- **Generative:** `ReconstructionLoss`, `KLDivergenceLoss`.
- **Geometry alignment:** `ZermeloAlignmentLoss`, `VelocityDirectionAlignmentLoss`,
  `ContrastiveAlignmentLoss`, `WindAssistedTrajectoryAlignmentLoss`.
- **Dynamics consistency:** `GeodesicSprayLoss`, `EulerLagrangeResidualLoss`,
  `AVBDPathEnergyLoss`, `LongTrajectoryAlignmentLoss`.
- **Regularizers / priors:** `MetricAnchorLoss`, `MetricSmoothnessLoss`,
  `WindThermodynamicLoss`, `KinematicPriorLoss`.
- **Arrival-time matching:** `ArrivalTimeLoss`, `DenseArrivalTimeLoss`.
- **Flow / action matching:** `FinslerActionMatchingLoss`, `FinslerianFlowMatchingLoss`.
- **EBM** (`losses_ebm.py`): `ContrastiveDivergenceLoss`, `DenoisingScoreMatchingLoss`, `MSELoss`.

### 6.3 `TrainingPhase`

A declarative description of one stage: `name`, `epochs`, an
`optax.GradientTransformation`, a list of losses, a `filter_spec` (a PyTree mask
marking trainable vs. frozen leaves, consumed by `eqx.partition`), and
`requires_pairs` (lineage pair/triple batching).

### 6.4 `HAMPipeline`

Executes phases in sequence. For each phase it partitions the model via
`filter_spec`, initializes the optimizer on the trainable partition, runs vmapped
mini-batch gradient descent (`fit(dataset, phases, batch_size, …)`), and
recombines for the next phase. The model is mutated in place; the returned model
and `self.model` are the same object. Phases with `requires_pairs=True` are
skipped (with a printed notice) when no lineage data is present.

---

## 7. Implementation Status

### Completed & validated

1. **Geometry core** — `metric.py`, `zoo/`, `manifolds/`, `mesh.py`: complete and
   exercised by the test suite (333 tests / 32 modules). `FinslerMetric`
   auto-differentiates the energy to the spray; `Randers` / `DiscreteRanders`
   implement Zermelo navigation; curvature and Berwald transport are validated on
   `Sphere`, `Torus`, `Hyperboloid`, and triangular meshes.
2. **Geodesic solvers** — `ExponentialMap` (IVP), `AVBDSolver` (BVP, with
   parallel + implicit-diff modes), and `GaussNewtonGeodesic` (global BVP).
3. **Eikonal solvers** — grid, volumetric (3-D), and mesh arrival-time PDEs with
   implicit gradients; used inside arrival-time training losses.
4. **Parallel transport** — Berwald connection verified for norm preservation on
   the sphere and non-trivial Randers holonomy.
5. **Training pipeline** — `HAMPipeline` with per-phase freezing, lineage-triple
   batching, and the modular loss library above.
6. **Applications** (`examples/`, `experiments/`):
   - *Marine navigation* (`experiments/marine/`) — time-dependent Zermelo routing
     for a 3-D underwater glider on a Randers metric + differentiable eikonal
     solver.
   - *Wildfire* (`models/wildfire.py`, `data/wildfire.py`) — front propagation
     with terrain-covariate-conditioned Randers metrics.
   - *Single-cell* (`data/` Weinreb loaders) — generative latent-geometry models
     of hematopoietic differentiation.

### Known limitations

7. **Curved-manifold VAEs.** Joint training of the full generative pipeline on
   strongly curved latent manifolds (`Sphere`, `Hyperboloid`) remains numerically
   sensitive; the flat `EuclideanSpace` latent is the recommended default for
   biological applications. Integrating exact `cosh`/`sinh` maps with deep
   learning loops can still trigger solver collapse (see `spec/MATH_SPEC.md`
   § 4.1).
