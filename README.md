# HAM — Differentiable Finsler Geometry in JAX

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-green.svg)](https://github.com/google/jax)

**HAM** is a differentiable library for **Finsler Geometry** built on JAX and Equinox. It treats geometric metrics as learnable modules — define a cost function $F(x, v)$, and the library auto-differentiates geodesics, sprays, curvature, and parallel transport.

## Key Features

- **Metric-First Design** — Define $F(x, v)$; geodesic sprays, fundamental tensors, and Berwald transport follow automatically via `jax.grad` / `jax.hessian`.
- **Implicit Dynamics** — Euler-Lagrange equations solved without expanding Christoffel symbols ($O(N^3)$ avoided).
- **Zermelo Navigation** — Built-in Randers metric with causality-preserving wind squasher ($\|W\|_h < 1$).
- **Two Geodesic Solvers** — Boundary-value (`AVBDSolver`) and initial-value (`ExponentialMap` via RK4).
- **Learnable Metrics** — Neural Riemannian / Randers metrics, pullback metrics from decoder Jacobians, and data-driven kernel wind fields.
- **Training Pipeline** — Declarative multi-phase training with per-phase parameter freezing, modular loss components, and lineage-aware batching.

## Installation

```bash
git clone https://github.com/hubibala/HAM.git
cd HAM
pip install -e ".[dev]"
```

> **Note:** Requires JAX ≥ 0.4.0. For GPU support, install JAX with CUDA following [JAX's installation guide](https://github.com/google/jax#installation).

---

## ⚡ Quickstart

### 1. Geodesic on a Sphere

Compute the shortest path between two points on $S^2$ using the geodesic spray ODE:

```python
import jax.numpy as jnp
from ham.geometry.surfaces import Sphere
from ham.geometry.zoo import Euclidean
from ham.solvers.geodesic import ExponentialMap

# Define the manifold and metric
sphere = Sphere(intrinsic_dim=2, radius=1.0)
metric = Euclidean(sphere)

# Shoot a geodesic from the equator toward the north pole
x0 = jnp.array([1.0, 0.0, 0.0])
v0 = jnp.array([0.0, 0.0, 1.0])  # initial velocity

shooter = ExponentialMap(step_size=0.01, max_steps=100)
x_final = shooter.shoot(metric, x0, v0)
# x_final ≈ [0, 0, 1] (north pole)
```

### 2. Randers Metric with Wind

Define an asymmetric Randers metric where a wind field $W(x)$ makes travel cheaper in one direction:

```python
import jax.numpy as jnp
from ham.geometry.surfaces import EuclideanSpace
from ham.geometry.zoo import Randers
from ham.solvers.avbd import AVBDSolver

# 2D plane with constant eastward wind
manifold = EuclideanSpace(dim=2)
h_net = lambda x: jnp.eye(2)               # flat Riemannian "sea"
w_net = lambda x: jnp.array([0.3, 0.0])    # wind blowing east

metric = Randers(manifold, h_net, w_net)

# Solve for the geodesic between two points
solver = AVBDSolver(iterations=50)
start = jnp.array([0.0, 0.0])
end   = jnp.array([1.0, 1.0])
traj  = solver.solve(metric, start, end, n_steps=20)

# Verify asymmetry: forward vs backward arc length
L_fwd = float(metric.arc_length(traj.xs))
L_bwd = float(metric.arc_length(traj.xs[::-1]))
print(f"Forward: {L_fwd:.4f}, Backward: {L_bwd:.4f}")
# Forward < Backward (wind helps eastward travel)
```

### 3. Learning a Metric with the Training Pipeline

Train a neural Randers metric to align its wind field with observed velocity data:

```python
import jax
import optax
import equinox as eqx
from ham.geometry.surfaces import EuclideanSpace
from ham.models.learned import NeuralRanders
from ham.training.pipeline import HAMPipeline, TrainingPhase
from ham.training.losses import ReconstructionLoss, KLDivergenceLoss

# Build a learnable metric
key = jax.random.PRNGKey(42)
manifold = EuclideanSpace(dim=8)
metric = NeuralRanders(manifold, key, hidden_dim=64, depth=3)

# Define training phases with parameter freezing
phase = TrainingPhase(
    name="WindAlignment",
    epochs=100,
    optimizer=optax.adam(1e-3),
    losses=[ReconstructionLoss(weight=1.0)],
    filter_spec=lambda m: jax.tree_util.tree_map(
        lambda x: True if eqx.is_array(x) else False, m
    ),
)

# pipeline = HAMPipeline(model).fit(dataset, [phase])
```

---

## 📂 Repository Structure

```
src/ham/
├── geometry/
│   ├── manifold.py       # Manifold abstract base class
│   ├── metric.py         # FinslerMetric ABC + auto-diff spray/energy
│   ├── surfaces.py       # Sphere, Torus, Hyperboloid, Paraboloid, EuclideanSpace
│   ├── zoo.py            # Euclidean, Riemannian, Randers, DiscreteRanders
│   ├── mesh.py           # Triangular mesh manifold
│   └── transport.py      # Berwald connection + parallel transport
├── models/
│   └── learned.py        # NeuralRiemannian, NeuralRanders, PullbackRanders,
│                          # DataDrivenPullbackRanders
├── nn/
│   └── networks.py       # VectorField, PSDMatrixField, RandomFourierFeatures
├── solvers/
│   ├── avbd.py           # BVP solver (Augmented Vertex Block Descent)
│   └── geodesic.py       # IVP solver (ExponentialMap via RK4)
├── training/
│   ├── pipeline.py       # HAMPipeline: multi-phase declarative training
│   └── losses.py         # Modular loss components (reconstruction, KL,
│                          # alignment, Euler-Lagrange residual, etc.)
├── bio/
│   ├── vae.py            # GeometricVAE with Zermelo control dynamics
│   └── data.py           # BioDataset (AnnData integration, lineage pairs)
├── sim/
│   └── fields.py         # Field abstractions
├── utils/
│   └── math.py           # safe_norm, numerical stability primitives
└── vis/
    └── hyperbolic.py     # Poincaré disk visualization
```

## 🧪 Running Tests

```bash
# Run the full test suite
python -m pytest tests/ -v

# Run individual test modules
python tests/test_metric.py        # Metric algebra
python tests/test_geodesic.py      # Spray ODE + energy conservation
python tests/test_transport.py     # Berwald parallel transport
python tests/test_pipeline.py      # Training pipeline
python tests/test_mesh_solver.py   # Mesh-constrained geodesics
```

## 📝 Citation

```bibtex
@software{ham2026,
  author = {HAM Research Team},
  title  = {HAM: Differentiable Finsler Geometry in JAX},
  year   = {2026},
  url    = {https://github.com/hubibala/HAM}
}
```
