# HAM — Differentiable Finsler Geometry in JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-green.svg)](https://github.com/google/jax)

**HAM** is a JAX-native library for **learnable Finsler geometry**. You define a
cost function $F(x, v)$ — the price of moving through point $x$ in direction $v$ —
and HAM auto-differentiates everything that follows: geodesics, the geodesic
spray, curvature, and parallel transport. Because metrics are
[Equinox](https://github.com/patrick-kidger/equinox) modules, $F$ can be a neural
network, and the whole pipeline is differentiable end-to-end.

Finsler geometry generalizes Riemannian geometry by dropping the requirement
that cost be symmetric: travelling **east** can be cheaper than travelling
**west**. That single relaxation is what lets HAM model wind, ocean currents,
cell-differentiation arrows, and wildfire spread as *geometry* rather than as
bolted-on vector fields.

```python
from ham.geometry import Randers, EuclideanSpace
from ham.solvers import AVBDSolver
import jax.numpy as jnp

# A plane with a steady eastward wind — moving with it is cheaper.
metric = Randers(EuclideanSpace(dim=2),
                 h_net=lambda x: jnp.eye(2),          # the "sea": flat Riemannian metric
                 w_net=lambda x: jnp.array([0.3, 0.0]))  # the "wind": a drift field

traj = AVBDSolver(iterations=50).solve(
    metric, jnp.array([0., 0.]), jnp.array([1., 1.]), n_steps=20)

print(metric.arc_length(traj.xs),          # downwind cost  ≈ 1.19
      metric.arc_length(traj.xs[::-1]))     # upwind cost    ≈ 1.83
```

---

## Why HAM?

- **Metric-first design.** Define $F(x, v)$; the geodesic spray, fundamental
  tensor $g_{ij}$, Berwald connection, and flag curvature all follow from
  `jax.grad` / `jax.hessian`. You never hand-code a Christoffel symbol.
- **Implicit dynamics.** The Euler–Lagrange equations are solved as a small
  linear system per step, avoiding the $O(N^3)$ blow-up of explicit connection
  coefficients.
- **Asymmetric (Randers) metrics done right.** A built-in Zermelo
  parameterization keeps the wind field causal ($\|W\|_h < 1$) and the metric
  strongly convex, with a $C^1$ squashing function (no discontinuity at the
  boundary).
- **Four ways to find a geodesic.** Shoot from initial conditions
  (`ExponentialMap`), relax a boundary-value path locally (`AVBDSolver`) or
  globally (`GaussNewtonGeodesic`), or solve the arrival-time PDE on a grid/mesh
  (`EikonalSolver` family).
- **Learnable metrics.** Neural Riemannian/Randers metrics, decoder-pullback
  metrics for latent spaces, energy-based and kernel wind fields — all trainable.
- **A declarative training pipeline.** Compose multi-phase schedules with
  per-phase parameter freezing and a library of geometry-aware losses.

---

## Installation

```bash
git clone https://github.com/hubibala/HAM.git
cd HAM
pip install -e ".[dev]"          # core + dev tooling (pytest, ruff, matplotlib, …)
```

The distribution is named **`hamtools`**; you import it as **`ham`**.

```python
import ham
ham.__version__   # '1.1.0'
```

Optional extras:

| Extra | Installs | For |
| :--- | :--- | :--- |
| `dev` | pytest, ruff, mypy, matplotlib, jupyter, plotly | development & examples |
| `gpu` | `jax[cuda12]` | NVIDIA GPU acceleration |
| `wildfire` | Pillow, rasterio | the wildfire terrain application |
| `bio` | anndata, scanpy, scvelo, pandas | single-cell / RNA-velocity data loaders (`ham.utils.download_*`) |

> The core install carries only the geometry/solver stack (JAX, Equinox, Optax,
> NumPy, SciPy). The single-cell data loaders are gated behind `[bio]`; calling
> them without it raises a clear install hint.

> **Requires JAX ≥ 0.4.** For GPU/TPU builds, follow the
> [JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html);
> the `gpu` extra covers the common CUDA 12 case.

---

## ⚡ Quickstart

### 1. Shoot a geodesic on a sphere

Integrate the geodesic spray ODE from an initial position and velocity (the
*exponential map*). Starting at the equator and shooting north for a quarter
turn lands on the pole:

```python
import jax.numpy as jnp
from ham.geometry import Sphere, Euclidean
from ham.solvers import ExponentialMap

sphere = Sphere(intrinsic_dim=2, radius=1.0)
metric = Euclidean(sphere)                  # round metric induced from the ambient norm

x0 = jnp.array([1.0, 0.0, 0.0])             # on the equator
v0 = jnp.array([0.0, 0.0, 1.0])             # unit velocity, pointing north

shooter = ExponentialMap(max_steps=200)
x_final = shooter.shoot(metric, x0, v0, t_max=jnp.pi / 2)
# x_final ≈ [0, 0, 1]  — the north pole (arc length |v0|·t_max = π/2)
```

> A unit-speed geodesic travels `t_max` radians along a great circle, so
> reaching the pole from the equator needs `t_max = π/2`, not `1.0`.

### 2. An asymmetric Randers metric

A Randers metric is a Riemannian "sea" $h$ plus a drifting "wind" $W$. Travel is
cheaper downwind, so forward and backward arc length differ:

```python
import jax.numpy as jnp
from ham.geometry import EuclideanSpace, Randers
from ham.solvers import AVBDSolver

manifold = EuclideanSpace(dim=2)
metric = Randers(manifold,
                 h_net=lambda x: jnp.eye(2),            # flat sea
                 w_net=lambda x: jnp.array([0.3, 0.0]))  # constant eastward wind

solver = AVBDSolver(iterations=50)
traj = solver.solve(metric, jnp.array([0., 0.]), jnp.array([1., 1.]), n_steps=20)

L_fwd = float(metric.arc_length(traj.xs))
L_bwd = float(metric.arc_length(traj.xs[::-1]))
print(f"downwind: {L_fwd:.4f}, upwind: {L_bwd:.4f}")
# downwind: 1.1939, upwind: 1.8305  — wind helps eastward travel
```

### 3. Learn a metric from data

A neural metric is an `eqx.Module`, so you train it with a standard Equinox +
Optax loop. Minimizing the Finsler energy of observed `(position, velocity)`
pairs teaches the metric to make the directions data actually moves in *cheap* —
i.e. it recovers the underlying wind/drift:

```python
import jax, jax.numpy as jnp, optax, equinox as eqx
from ham.geometry import EuclideanSpace
from ham.models.learned import NeuralRanders

key = jax.random.PRNGKey(42)
metric = NeuralRanders(EuclideanSpace(dim=8), key, hidden_dim=64, depth=3)

opt = optax.adam(1e-3)
opt_state = opt.init(eqx.filter(metric, eqx.is_array))

@eqx.filter_jit
def step(m, X, V, state):                          # X: (B, 8) points, V: (B, 8) velocities
    def loss_fn(m):
        return jnp.mean(jax.vmap(m.energy)(X, V))   # make observed motion low-cost
    loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
    updates, state = opt.update(grads, state, m)
    return eqx.apply_updates(m, updates), state, loss

# for X, V in batches:
#     metric, opt_state, loss = step(metric, X, V, opt_state)
```

See [`examples/demo_learned_wind.py`](examples/demo_learned_wind.py) for a full
runnable version (recovering a Rossby–Haurwitz wind on the sphere, with
smoothness regularization).

> **Higher-level pipeline.** For *generative* latent-geometry models — a VAE
> whose latent space carries a learned Randers metric — `ham.training.HAMPipeline`
> orchestrates multi-phase training (per-phase parameter freezing, lineage-aware
> batching, and a library of geometry-aware losses such as `ZermeloAlignmentLoss`
> and `EulerLagrangeResidualLoss`). Those losses expect a model exposing
> `encode` / `decode` / `metric`, not a bare metric — see
> [`spec/ARCH_SPEC.md`](spec/ARCH_SPEC.md) § 6.

---

## Core concepts

HAM separates **where** you are (topology) from **how costly** motion is
(geometry), then layers solvers on top.

| Layer | Abstraction | Concrete types |
| :--- | :--- | :--- |
| **Topology** | `Manifold` | `EuclideanSpace`, `Sphere`, `Torus`, `Hyperboloid`, `Paraboloid`, `TriangularMesh` |
| **Geometry** | `FinslerMetric` → `AsymmetricMetric` | `Euclidean`, `Riemannian`, `Randers`, `DiscreteRanders`, `SegmentQuadratureMetric` |
| **Learnable geometry** | (subclasses of the above) | `NeuralRanders`, `NeuralRiemannian`, `PullbackRanders`, `DataDrivenPullbackRanders`, `EnergyBasedRanders`, `KernelWindField`, … |
| **Geodesics** | initial- and boundary-value solvers | `ExponentialMap`, `AVBDSolver`, `GaussNewtonGeodesic`, `GeodesicLearningSolver` |
| **Arrival times** | anisotropic Eikonal PDE | `EikonalSolver` (grid), `MeshEikonalSolver`, `VolumetricEikonalSolver` (3D) |
| **Transport & curvature** | derived geometry | `BerwaldConnection`, `sectional_curvature`, `flag_curvature_sample`, `riemann_curvature_tensor` |

Every `FinslerMetric` is an `eqx.Module` (a JAX PyTree), so any metric — even a
neural one — can be passed straight through `jax.jit`, `jax.grad`, and `jax.vmap`.

The two faces of a geodesic problem:

- **Shooting (IVP).** Given a start point and velocity, integrate the spray ODE
  → `ExponentialMap`.
- **Connecting (BVP).** Given two endpoints, find the minimizing path →
  `AVBDSolver` (local block descent), `GaussNewtonGeodesic` (global, second-order,
  iteration count independent of path length), or the Eikonal solvers (full
  arrival-time field via fast sweeping).

For the mathematics behind these, see [`spec/MATH_SPEC.md`](spec/MATH_SPEC.md);
for the software design, [`spec/ARCH_SPEC.md`](spec/ARCH_SPEC.md).

---

## 📂 Repository structure

```text
src/ham/
├── geometry/
│   ├── manifold.py          # Manifold ABC
│   ├── manifolds/           # EuclideanSpace, Sphere, Torus, Hyperboloid, Paraboloid
│   ├── mesh.py              # TriangularMesh manifold (+ mesh_adjacency.py)
│   ├── metric.py            # FinslerMetric & AsymmetricMetric ABCs + auto-diff spray/energy
│   ├── zoo/                 # Euclidean, Riemannian, Randers, DiscreteRanders, SegmentQuadrature
│   ├── transport.py         # BerwaldConnection (parallel transport)
│   └── curvature.py         # flag / sectional / Riemann curvature
├── models/
│   ├── learned.py           # Neural, pullback, energy-based & kernel metrics
│   └── wildfire.py          # CovariateConditionedRanders, terrain CNN
├── nn/
│   ├── networks.py          # VectorField, PSDMatrixField, RandomFourierFeatures
│   ├── ebm.py               # ScalarEnergyField, PseudotimePotential
│   └── kde.py               # GaussianKDEEnergy
├── solvers/
│   ├── geodesic.py          # ExponentialMap (IVP, RK4)
│   ├── avbd.py              # AVBDSolver (BVP, vertex block descent)
│   ├── gauss_newton.py      # GaussNewtonGeodesic (global block-tridiagonal Newton)
│   ├── geodesic_learning.py # GeodesicLearningSolver (full-path Adam)
│   ├── eikonal.py           # EikonalSolver (2D grid fast sweeping)
│   ├── mesh_eikonal.py      # MeshEikonalSolver (unstructured triangulations)
│   ├── volumetric_eikonal.py# VolumetricEikonalSolver (3D grids)
│   ├── continuation.py      # arc-length resampling, numerical continuation
│   ├── graph_init.py        # kNN-graph geodesic warm-starts
│   └── coloring.py          # graph colorings for parallel sweeps
├── training/
│   ├── pipeline.py          # HAMPipeline, TrainingPhase
│   ├── losses.py            # geometry-aware loss components
│   └── losses_ebm.py        # contrastive divergence, score matching
├── data/  sim/  utils/  vis/  # loaders, analytic fields, numerics, plotting

examples/        # runnable demo scripts + Jupyter notebooks
experiments/     # larger applications (marine navigation, …)
spec/            # MATH_SPEC.md, ARCH_SPEC.md, solver studies
tests/           # 333 tests across 32 modules
```

---

## 📚 Examples

Runnable scripts live in [`examples/`](examples/); narrated walkthroughs with
plots live in [`examples/notebooks/`](examples/notebooks/).

| Topic | Script | Notebook |
| :--- | :--- | :--- |
| Geodesic shooting on curved surfaces | — | `demo_geodesic_shooting.ipynb`, `demo_curved_manifolds.ipynb` |
| Zermelo navigation / Randers winds | `demo_zermelo.py` | `demo_zermelo.ipynb` |
| Vortex wind field | `demo_vortex.py` | `demo_vortex.ipynb` |
| Learned wind from data | `demo_learned_wind.py` | `demo_learned_wind.ipynb` |
| Discrete (mesh) Zermelo metric | `demo_discrete_zermelo.py` | `demo_discrete_zermelo.ipynb` |
| Anisotropic Eikonal fronts | `demo_eikonal_fronts.py` | `demo_eikonal_fronts.ipynb` |
| Parallel transport & holonomy | — | `demo_parallel_transport.ipynb` |
| High-dimensional latent geodesics | — | `demo_high_dim_latent_geodesics.ipynb` |
| Generic neural Finsler metric | — | `demo_generic_finsler.ipynb` |

### Worked application: marine navigation

[`experiments/marine/`](experiments/marine/README.md) is a larger, end-to-end use
of the library: planning a time-optimal route for a buoyancy-driven underwater
glider through a time-varying, depth-stratified ocean current. It's a useful
reference for how the pieces fit together on a real problem.

Time-optimal navigation through a current is [Zermelo's
problem](https://arxiv.org/abs/2304.00478), whose solutions are the geodesics of
a Randers metric — so the experiment is built directly on HAM's `Randers` metric,
the differentiable `VolumetricEikonalSolver` (arrival-time field), and
`AVBDSolver` (route), plus a clock-threaded planner for the time-dependent case.
It runs in four stages:

| Stage | What it does | HAM components |
| :--- | :--- | :--- |
| **A — Forward planning** | arrival-time field + route through a frozen current | `VolumetricEikonalSolver`, `AVBDSolver` |
| **B — Reconstruction** | recover the current from passive drifter tracks (velocity regression) | `KernelWindField`, stream-function fit |
| **C — Time-dependent** | route through an *evolving* current (the stationary eikonal no longer applies) | time-lifted planner + differentiable penalties |
| **D — Closed-loop (MPC)** | receding-horizon replanning under a forecast that decays with lead time | warm-started re-solves |

The numbers are modest and reported with their caveats — e.g. diving to a
favorable deep layer saves ~10% over a surface-locked plan, and a time-aware
route is ~7% faster than a frozen-field one under a perfect forecast, with the
honest gap to closed-loop performance spelled out. See the
[experiment README](experiments/marine/README.md) for the full write-up,
reproduction commands, and limitations.

---

## 🧪 Tests

```bash
python -m pytest tests/ -q          # full suite (333 tests)
python -m pytest tests/test_metric.py tests/test_geodesic.py -v
```

| Module | Covers |
| :--- | :--- |
| `test_metric.py`, `test_zoo.py` | metric algebra, spray, energy |
| `test_geodesic.py`, `test_solver.py` | spray ODE, energy conservation |
| `test_avbd.py`, `test_gauss_newton.py` | BVP solvers, implicit differentiation |
| `test_eikonal_solver.py`, `test_mesh_eikonal.py`, `test_volumetric_eikonal.py` | arrival-time PDEs |
| `test_transport.py`, `test_curvature.py` | Berwald transport, curvature |
| `test_pipeline.py`, `test_learned_metric.py` | training pipeline, neural metrics |

> CI runs on CPU. If you hit GPU/accelerator initialization in a CPU-only
> environment, set `JAX_PLATFORMS=cpu`.

---

## 📝 Citation

```bibtex
@software{ham2026,
  author = {HAM Research Team},
  title  = {HAM: Differentiable Finsler Geometry in JAX},
  year   = {2026},
  url    = {https://github.com/hubibala/HAM}
}
```

## License

MIT — see [LICENSE](LICENSE).
