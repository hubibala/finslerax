# finslerax — Differentiable Finsler Geometry in JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-green.svg)](https://github.com/google/jax)

**finslerax** is a JAX-native library for learnable
Finsler geometry. You supply a cost function $F(x, v)$ — the price of moving
through point $x$ in direction $v$ — and the library derives what follows:
geodesics, the geodesic spray, curvature, and parallel transport. Metrics are
[Equinox](https://github.com/patrick-kidger/equinox) modules, so $F$ may be a
neural network and the whole pipeline stays differentiable end to end.

Finsler geometry drops the Riemannian requirement that cost be symmetric.
Travelling east can be cheaper than travelling west, which is what makes wind,
ocean currents, gravity in a robot's joint space, and spreading fronts
expressible as geometry.

```python
from finslerax.geometry import Randers, EuclideanSpace
from finslerax.solvers import AVBDSolver
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

**Documentation:** <https://finslerax-docs.vercel.app/>

---

## Author

finslerax is written by **Balázs Hubicska**, whose research is in Finsler
holonomy:

- B. Hubicska, V. S. Matveev & Z. Muzsnay, *Almost all Finsler metrics have
  infinite dimensional holonomy group*, **Journal of Geometric Analysis** (2020).
  [doi:10.1007/s12220-020-00517-9](https://doi.org/10.1007/s12220-020-00517-9)
- B. Hubicska & Z. Muzsnay, *Holonomy in the quantum navigation problem*,
  **Quantum Information Processing** 18:325 (2019).
  [doi:10.1007/s11128-019-2438-8](https://doi.org/10.1007/s11128-019-2438-8)
- B. Hubicska & Z. Muzsnay, *The holonomy group of projectively flat Randers
  two-manifolds of constant curvature* (2018).
  [arXiv:1805.05216](https://arxiv.org/abs/1805.05216),
  [doi:10.48550/arXiv.1805.05216](https://doi.org/10.48550/arXiv.1805.05216)

The transport and holonomy machinery in this library implements that line of
work rather than reproducing a textbook.

[issue](https://github.com/hubibala/finslerax/issues).

---

## Features

- Define $F(x, v)$ and the geodesic spray, fundamental tensor $g_{ij}$, Berwald
  connection and flag curvature follow from `jax.grad` and `jax.hessian`. No
  hand-coded Christoffel symbols.
- The Euler–Lagrange equations are solved as a small linear system per step,
  avoiding the $O(N^3)$ cost of forming connection coefficients explicitly.
- Randers metrics carry a Zermelo parameterization that keeps the wind causal
  ($\lVert W \rVert_h < 1$) and the metric strongly convex, using a $C^\infty$
  smooth-minimum that bends only near the boundary and leaves already-causal
  winds untouched.
- Four routes to a geodesic: shoot from initial conditions (`ExponentialMap`),
  relax a boundary-value path locally (`AVBDSolver`) or globally
  (`GaussNewtonGeodesic`), or solve the arrival-time PDE on a grid or mesh
  (the `EikonalSolver` family).
- Neural Riemannian and Randers metrics, decoder-pullback metrics for latent
  spaces, and energy-based and kernel wind fields are all trainable.
- Multi-phase training schedules with per-phase parameter freezing compose from
  a library of geometry-aware losses.

### The HAM research programme

The library grew out of a programme called HAM (*Holonomic Association Model*):
representing context as geometry. Parallel transport moves a representation
between contexts along a path, and holonomy — the path dependence of that
transport — is what the geometry remembers. Associations between states become
geodesics of a learned, possibly asymmetric, metric. HAM remains the name of that
research line; **finslerax** is the name of the software. The library is useful
well beyond the programme, but the transport and holonomy machinery it needed is
why every piece here exists.

---

## Prior art

The eikonal stack here follows **Gahtan, Shpund & Bronstein**, *Differentiable
Eikonal Wildfire Modelling* ([arXiv:2603.00035](https://arxiv.org/abs/2603.00035),
[code](https://github.com/BarakGahtan/differentiable-eikonal-wildfire), MIT).
Their algorithm and their reference implementation are what this was built from:
fast sweeping forward, adjoint fixed-point backward, with the metric produced by
a convolutional encoder over rasterized covariates. The JAX implementation and
the Finsler generalization are new here; the design is theirs.

Three parts of this repository are directly downstream of that paper and say so
in their docstrings:

- [`solvers/eikonal.py`](src/finslerax/solvers/eikonal.py) — the fast-sweeping
  anisotropic Godunov solver with `jax.custom_vjp` implicit gradients.
- [`models/covariate.py`](src/finslerax/models/covariate.py) — `LocalTerrainCNN`
  follows their §6 encoder architecture; `CovariateConditionedRanders` uses their
  $\lambda = 1$ Zermelo parametrization.
- [`training/losses.py`](src/finslerax/training/losses.py) — the arrival-time
  metric-recovery loss follows their §5.

The [wildfire application branch](https://github.com/hubibala/finslerax/tree/app/wildfire)
is a companion study to that paper and carries the full comparison.

Foundational Finsler material follows Bao, Chern & Shen, *An Introduction to
Riemann–Finsler Geometry* (Springer GTM 200, 2000); the anisotropic
fast-marching literature of Jean-Marie Mirebeau informs the eikonal design.

---

## Installation

```bash
git clone https://github.com/hubibala/finslerax.git
cd finslerax
pip install -e ".[dev]"          # core + dev tooling (pytest, ruff, matplotlib, …)
pip install -e ".[viz]"          # core + plotting (matplotlib, plotly) for finslerax.vis
```

The distribution is named **`finslerax`**; you import it as **`finslerax`**.

| Extra | Installs | For |
| :--- | :--- | :--- |
| `dev` | pytest, ruff, mypy, matplotlib, jupyter, plotly | development and examples |
| `viz` | matplotlib, plotly | the `finslerax.vis` plotting helpers |
| `gpu` | `jax[cuda12]` | NVIDIA GPU acceleration |

The core install carries only the geometry and solver stack: JAX, Equinox,
Optax, NumPy, SciPy. JAX ≥ 0.4 is required; for GPU and TPU builds follow the
[JAX installation guide](https://jax.readthedocs.io/en/latest/installation.html),
of which the `gpu` extra covers the common CUDA 12 case.

---

## Numerical precision

Precision is governed by JAX's own `jax_enable_x64` flag, which finslerax reads. The
default is **float32**. To run in float64, set the standard environment variable
before Python starts:

```bash
JAX_ENABLE_X64=1 python your_script.py
```

Equivalently, in-process before the first array is created:

```python
import jax
jax.config.update("jax_enable_x64", True)
import finslerax   # now float64
```

That switch flips the whole stack, because precision is decided at the
data-construction boundary and the solvers follow the dtype they are given.
Stability floors (`GRAD_EPS`, `PSD_EPS`, `TAYLOR_EPS`, …) scale with it. Query
the active setting through `finslerax.utils.config`:

```python
from finslerax.utils.config import x64_enabled, default_dtype, default_np_dtype
```

Reach for float64 on stiff or ill-conditioned solves: long AVBD geodesics, fine
eikonal grids, curvature and transport cancellation, tight VJP checks. Consumer
NVIDIA GPUs throttle FP64 to roughly 1/32–1/64 of FP32 throughput, so float32
remains the default and float64 is opt-in.

---

## Quickstart

### 1. Shoot a geodesic on a sphere

Integrate the geodesic spray ODE from an initial position and velocity — the
exponential map. Starting at the equator and shooting north for a quarter turn
lands on the pole:

```python
import jax.numpy as jnp
from finslerax.geometry import Sphere, Euclidean
from finslerax.solvers import ExponentialMap

sphere = Sphere(intrinsic_dim=2, radius=1.0)
metric = Euclidean(sphere)                  # round metric induced from the ambient norm

x0 = jnp.array([1.0, 0.0, 0.0])             # on the equator
v0 = jnp.array([0.0, 0.0, 1.0])             # unit velocity, pointing north

shooter = ExponentialMap(max_steps=200)
x_final = shooter.shoot(metric, x0, v0, t_max=jnp.pi / 2)
# x_final ≈ [0, 0, 1]  — the north pole (arc length |v0|·t_max = π/2)
```

A unit-speed geodesic travels `t_max` radians along a great circle, so reaching
the pole from the equator needs `t_max = π/2`, not `1.0`.

### 2. An asymmetric Randers metric

A Randers metric is a Riemannian "sea" $h$ plus a drifting "wind" $W$. Travel is
cheaper downwind, so forward and backward arc length differ:

```python
import jax.numpy as jnp
from finslerax.geometry import EuclideanSpace, Randers
from finslerax.solvers import AVBDSolver

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

A neural metric is an `eqx.Module`, so it trains in a standard Equinox and Optax
loop. Minimizing the Finsler energy of observed `(position, velocity)` pairs
makes the directions the data actually moves in cheap, which recovers the
underlying drift:

```python
import jax, jax.numpy as jnp, optax, equinox as eqx
from finslerax.geometry import EuclideanSpace
from finslerax.models.learned import NeuralRanders

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
```

[`examples/demo_learned_wind.py`](examples/demo_learned_wind.py) has a full
runnable version that recovers a Rossby–Haurwitz wind on the sphere with
smoothness regularization.

For generative latent-geometry models — a VAE whose latent space carries a
learned Randers metric — `finslerax.training.TrainingPipeline` orchestrates multi-phase
training with per-phase freezing and geometry-aware losses such as
`ZermeloAlignmentLoss` and `EulerLagrangeResidualLoss`. Those losses expect a
model exposing `encode`, `decode` and `metric` rather than a bare metric; see
[`spec/ARCH_SPEC.md`](spec/ARCH_SPEC.md) § 6.

---

## Core concepts

finslerax separates where you are (topology) from how costly motion is (geometry),
then layers solvers on top.

| Layer | Abstraction | Concrete types |
| :--- | :--- | :--- |
| Topology | `Manifold` | `EuclideanSpace`, `Sphere`, `Torus`, `Hyperboloid`, `Paraboloid`, `TriangularMesh` |
| Geometry | `FinslerMetric` → `AsymmetricMetric` | `Euclidean`, `Riemannian`, `Randers`, `DiscreteRanders`, `SegmentQuadratureMetric` |
| Learnable geometry | subclasses of the above | `NeuralRanders`, `NeuralRiemannian`, `PullbackRanders`, `PullbackRiemannian`, `KernelWindField`, … |
| Geodesics | initial- and boundary-value solvers | `ExponentialMap`, `AVBDSolver`, `GaussNewtonGeodesic`, `GeodesicLearningSolver` |
| Arrival times | anisotropic eikonal PDE | `EikonalSolver` (grid), `MeshEikonalSolver`, `VolumetricEikonalSolver` (3D) |
| Transport and curvature | derived geometry | `BerwaldConnection`, `sectional_curvature`, `flag_curvature_sample`, `riemann_curvature_tensor` |

Every `FinslerMetric` is an `eqx.Module`, hence a JAX PyTree, so any metric —
including a neural one — passes straight through `jax.jit`, `jax.grad` and
`jax.vmap`.

A geodesic problem comes in two forms. **Shooting** takes a start point and
velocity and integrates the spray ODE, via `ExponentialMap`. **Connecting**
takes two endpoints and finds the minimizing path, via `AVBDSolver` (local block
descent), `GaussNewtonGeodesic` (global and second-order, with an iteration
count independent of path length), or the eikonal solvers, which return a full
arrival-time field by fast sweeping.

The mathematics is in [`spec/MATH_SPEC.md`](spec/MATH_SPEC.md); the software
design is in [`spec/ARCH_SPEC.md`](spec/ARCH_SPEC.md).

---

## Repository structure

```text
src/finslerax/
├── geometry/     # Manifold and FinslerMetric ABCs, manifolds/, mesh, zoo/,
│                 # transport (Berwald connection), curvature (flag/sectional/Riemann)
├── models/       # learned.py: neural, pullback, energy-based, kernel metrics
│                 # covariate.py: CovariateConditionedRanders, terrain CNN
├── nn/           # VectorField, PSDMatrixField, RandomFourierFeatures, EBM, KDE
├── solvers/      # geodesic (IVP), avbd + gauss_newton + geodesic_learning (BVP),
│                 # eikonal / mesh_eikonal / volumetric_eikonal (arrival times),
│                 # continuation, graph_init, coloring (warm-starts)
├── training/     # TrainingPipeline, TrainingPhase, geometry-aware losses
└── sim/ utils/ vis/   # analytic fields, numerics, terrain, plotting

examples/        # runnable demo scripts + Jupyter notebooks
spec/            # MATH_SPEC.md, ARCH_SPEC.md
tests/           # 349 tests across 35 modules, run in both precisions
```

---

## Examples

Runnable scripts live in [`examples/`](examples/); narrated walkthroughs with
plots live in [`examples/notebooks/`](examples/notebooks/).

| Topic | Script | Notebook |
| :--- | :--- | :--- |
| Geodesic shooting on curved surfaces | — | `demo_geodesic_shooting.ipynb`, `demo_curved_manifolds.ipynb` |
| Zermelo navigation / Randers winds | `demo_zermelo.py` | `demo_zermelo.ipynb` |
| Vortex wind field | `demo_vortex.py` | `demo_vortex.ipynb` |
| Learned wind from data | `demo_learned_wind.py` | `demo_learned_wind.ipynb` |
| Discrete (mesh) Zermelo metric | `demo_discrete_zermelo.py` | `demo_discrete_zermelo.ipynb` |
| Anisotropic eikonal fronts | `demo_eikonal_fronts.py` | `demo_eikonal_fronts.ipynb` |
| Parallel transport and holonomy | — | `demo_parallel_transport.ipynb` |
| High-dimensional latent geodesics | — | `demo_high_dim_latent_geodesics.ipynb` |
| Generic neural Finsler metric | — | `demo_generic_finsler.ipynb` |

### Applications

End-to-end applications live on their own branches, so installing the library
never pulls in domain data loaders or their dependencies.

[**Wildfire spread**](https://github.com/hubibala/finslerax/tree/app/wildfire) models a
fire front as the unit-time level sets of an anisotropic Randers metric, with
terrain and fuel setting the symmetric part and wind the drift. It builds on
`CovariateConditionedRanders`, the differentiable `EikonalSolver`, and the
covariate-encoder training loop.

[**Robot-arm geodesics**](https://github.com/hubibala/finslerax/tree/app/robot-arm)
plans energy-optimal motion in configuration space. The arm's mass matrix is the
Riemannian metric, gravity enters as a Randers drift, obstacles fold into the
metric, and task constraints are enforced with augmented Lagrangian terms. It
builds on `AVBDSolver`, `GaussNewtonGeodesic`, continuation, and the eikonal
planners.

---

## Tests

```bash
python -m pytest tests/ -q                       # float32 (default)
JAX_ENABLE_X64=1 python -m pytest tests/ -q      # float64
```

Every test passes under both precisions. Precision-sensitive tolerances and
dtype checks adapt through `tests/_precision.py` (`tol()`,
`assert_default_dtype`), and the active precision is printed in the pytest
header. CI runs the full matrix: `JAX_ENABLE_X64` ∈ {0, 1} × Python 3.10/3.11,
on CPU. If you hit accelerator initialization in a CPU-only environment, set
`JAX_PLATFORMS=cpu`.

| Module | Covers |
| :--- | :--- |
| `test_metric.py`, `test_zoo.py` | metric algebra, spray, energy |
| `test_geodesic.py`, `test_solver.py` | spray ODE, energy conservation |
| `test_avbd.py`, `test_gauss_newton.py` | BVP solvers, implicit differentiation |
| `test_eikonal_solver.py`, `test_mesh_eikonal.py`, `test_volumetric_eikonal.py` | arrival-time PDEs |
| `test_transport.py`, `test_curvature.py` | parallel translation, curvature |
| `test_pipeline.py`, `test_learned_metric.py` | training pipeline, neural metrics |
| `test_invariants.py` | Finsler axioms and cross-cutting metric invariants |

---

## Development and AI disclosure

finslerax was developed with substantial assistance from AI coding tools (Anthropic's
Claude), used for implementation, tests and documentation throughout. The
mathematics is validated against the published literature and a numerical test
suite that runs in both float32 and float64, and every component has been
human-reviewed. Responsibility for correctness rests with the author, not the
tools. If you find an error, please open an issue.

Contributions are welcome, including AI-assisted ones, under the disclosure
policy in [CONTRIBUTING.md](CONTRIBUTING.md).

## Citation

```bibtex
@software{finslerax2026,
  author = {Hubicska, Bal\'azs},
  title  = {finslerax: Differentiable Finsler Geometry in JAX},
  year   = {2026},
  url    = {https://github.com/hubibala/finslerax}
}
```

## License

MIT — see [LICENSE](LICENSE).
