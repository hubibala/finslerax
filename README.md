# HAMTools: Holonomic Association Models & Finsler Geometry

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![JAX](https://img.shields.io/badge/backend-JAX-green.svg)](https://github.com/google/jax)

**HAMTools** is a differentiable library for **Finsler Geometry**, built on JAX. It treats geometric metrics as learnable neural networks ("Generative Geometry"), enabling applications in Model-Based RL, Physics Discovery, and Representation Learning.

## Key Features

- **Metric-First Design:** Define $F(x, v)$, and the library automatically derives Geodesics, Sprays, and Curvature via Auto-Differentiation.
- **Implicit Dynamics:** Uses `jax.grad` and `jax.jvp` to solve Euler-Lagrange equations without expanding Christoffel symbols ($O(N^3)$ avoided).
- **Berwald Transport:** Native parallel transport for non-Riemannian (asymmetric) spaces.
- **The Zoo:** Verified implementations of **Euclidean**, **Riemannian**, and **Randers** (Zermelo) metrics.

## Installation

```bash
pip install -e .
```

⚡ Quick Start

1. Define a Metric

```python
import jax.numpy as jnp
from ham.geometry import Randers, Manifold

# Define the underlying space (e.g., a Plane)
class Plane(Manifold):
    @property
    def ambient_dim(self): return 2
    @property
    def intrinsic_dim(self): return 2
    def project(self, x): return x
    def to_tangent(self, x, v): return v
    def random_sample(self, key, shape): return jax.random.normal(key, shape + (2,))

# Define the fields (Neural Networks or Analytical)
h_net = lambda x: jnp.eye(2)             # The "Sea" (Riemannian)
w_net = lambda x: jnp.array([0.5, 0.0])  # The "Wind" (Drift)

# Instantiate the Geometry
metric = Randers(Plane(), h_net, w_net)
```

2. Solve for Geodesics

```python
from ham.solvers import AVBDSolver

solver = AVBDSolver(step_size=0.1)
start = jnp.array([0., 0.])
end   = jnp.array([1., 1.])

# Find the energy-minimizing path
traj = solver.solve(metric, start, end, n_steps=20)
print(f"Path Cost: {traj.energy}")
## 📂 Repository Structure

- `src/ham/bio/`: Domain-specific wrappers for single-cell biology (AnnData, Geometric VAE).
- `src/ham/geometry/`: Core manifold and metric definitions (`manifolds`, `meshes`, `zoo` of metric types).
- `src/ham/models/`: Neural implementations (e.g., `LearnedFinsler`).
- `src/ham/nn/`: Neural network building blocks.
- `src/ham/sim/`: Fields and simulation utilities.
- `src/ham/solvers/`: Geodesic solvers (Boundary-Value via `AVBD`, Initial-Value via `ExponentialMap`).
- `src/ham/utils/`: Core math utilities for numerical stability.
- `src/ham/vis/`: Visualization tools for manifolds.
- `tests/`: Comprehensive unit and integration test suite.
- `examples/`: Some geometrically motivated examples to test the geometric correctness and applicability of the framework. Early examples for the Weinreb data usecase to model cell differentiation processes in the latent space.

📝 Citation
If you use HAMTools in your research:

```

@software{hamtools2025,
author = {HAM Research Team},
title = {HAMTools: Differentiable Finsler Geometry in JAX},
year = {2025},
url = {[https://github.com/hubibala/ham](https://github.com/hubibala/ham)}
}

```

```
