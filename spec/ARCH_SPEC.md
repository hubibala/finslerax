# ARCH_SPEC.md: Software Architecture of HAMTools

**Version:** 1.1.0
**Date:** April 2026
**Dependencies:** JAX, Equinox, Optax

## 1. Design Philosophy

`HAMTools` is a JAX-native library for learning and manipulating Finsler geometries. Unlike existing libraries (like `geomstats`), it prioritizes **generative** use-cases where the metric is a learnable neural network, not a static analytical formula.

**Core Principles:**
1.  **Metric-First Design:** The `Metric` object is the single source of truth. It defines the energy $E(x,v)$; everything else (geodesics, curvature, transport) is auto-differentiated from it.
2.  **Implicit Dynamics:** We never manually implement Christoffel symbols. We use `jax.grad` and `jax.hessian` to solve the Euler-Lagrange equations dynamically.
3.  **Batch-First:** All operations assume a leading batch dimension `(B, ...)` to support massive parallel simulation of agents.

---

## 2. Core Abstractions

### 2.1. The Manifold (Topology)
Defines the domain $\mathcal{M}$ and its constraints. It does *not* define distance (that's the Metric's job).

```python
class Manifold(ABC):
    """
    Abstract base class for the topological domain.
    """
    @property
    @abstractmethod
    def ambient_dim(self) -> int: ...
    
    @property
    @abstractmethod
    def intrinsic_dim(self) -> int: ...

    @abstractmethod
    def project(self, x: jnp.ndarray) -> jnp.ndarray:
        """Projects a point from ambient space back onto the manifold (e.g., x / |x|)."""
        pass

    @abstractmethod
    def to_tangent(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Projects an ambient vector v onto T_x M."""
        pass
    
    @abstractmethod
    def random_sample(self, key: jax.random.PRNGKey, shape: tuple) -> jnp.ndarray:
        """Returns random points on the manifold."""
        pass
        
    def retract(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Retracts a tangent vector to a point on the manifold (first-order approx to exp)."""
        return self.project(x + v)

    def exp_map(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Computes the exact exponential map. Defaults to retract if un-overridden."""
        return self.retract(x, v)

    def log_map(self, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """Computes the inverse of the exponential map (tangent secant)."""
        # Implementation via tangent projection and norm scaling
        pass
```

### 2.2. The Finsler Metric (Geometry)
The heart of the library.

```python
class FinslerMetric(ABC):
    """
    Defines the geometry via the Finsler energy function.
    """
    def __init__(self, manifold: Manifold):
        self.manifold = manifold

    @abstractmethod
    def metric_fn(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        The fundamental Finsler cost function F(x, v).
        Must be 1-homogeneous in v.
        """
        pass

    def energy(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """Computes Lagrangian L = 0.5 * F(x, v)^2."""
        return 0.5 * self.metric_fn(x, v)**2

    def spray(self, x: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the Geodesic Spray G^i(x, v).
        Solves the linear system induced by Euler-Lagrange.
        Returns: Spray coefficients G^i.
        """
        # Implementation via jax.grad and linear solve (see MATH_SPEC)
        pass
        
    def inner_product(self, x: jnp.ndarray, v: jnp.ndarray, w1: jnp.ndarray, w2: jnp.ndarray) -> jnp.ndarray:
        """
        Computes <w1, w2>_v using the fundamental tensor g_ij(x, v).
        g_ij = Hessian_v(Energy)
        """
        pass
```


## 3. The Metric Hierarchy
This implementation uses inheritance to specialize behavior while keeping the Spray and Solver generic.

| Class | Description | metric_fn(x, v) implementation |
| :--- | :--- | :--- |
| Euclidean | Flat space | `norm(v)` |
| Riemannian | Curved, symmetric | `sqrt(v @ G(x) @ v)` |
| Randers | Asymmetric, wind | `sqrt(v @ M(x) @ v) + dot(beta(x), v)` |
| DiscreteRanders | Anisotropic Mesh Metric | Computed via differentiable target face weights |
| LearnedFinsler | Generic Neural Network | Defines $h_{net}$ and $w_{net}$ in `models/learned.py` |
### 3.1. The Randers Specialization
This class specifically manages the Zermelo data ($h$, $W$) to ensure convexity.

```python
class RandersMetric(FinslerMetric):
    def __init__(self, manifold, h_net: Callable, w_net: Callable):
        super().__init__(manifold)
        self.h_net = h_net # Outputs (B, D, D) positive definite
        self.w_net = w_net # Outputs (B, D)
        
    def get_zermelo_data(self, x):
        """
        Returns h(x) and W(x), enforcing |W|_h < 1.
        """
        # ... implementation details ...
```

## 4. Solvers and Transport
### 4.1. The Solver Interface
We separate the definition of the geodesic (ODE) from the method of finding it (Shooting vs. Relaxation).

```python
class GeodesicSolver(ABC):
    @abstractmethod
    def solve(self, metric: FinslerMetric, p_start, p_end, n_steps) -> Trajectory:
        """Finds the energy-minimizing path between two points."""
        pass
```

### 4.2. Implementation: AVBDSolver
The Augmented Vertex Block Descent solver.
- Input: FinslerMetric, Boundary Conditions.
- Method: Optimizes discrete path points $x_0, ..., x_N$ directly. Loss: $L = \sum E(x_i, v_i) + \text{ConstraintPenalties}$.
- Differentiability: Fully differentiable w.r.t metric parameters.

### 4.3. Parallel Transport (Berwald)
Implemented as an integrator on top of the metric.

```python
class BerwaldConnection(Connection):
    def parallel_transport(self, path_x: jax.Array, path_v: jax.Array, vec_start: jax.Array) -> jax.Array:
        """
        Transports vector vec_start along a trajectory using the Berwald connection.
        Differentiation: 
            Gamma = Hessian_v(metric.spray)
        Integration:
            dX/dt = -Gamma(x, v) * v * X
        """
        pass
```

### 4.4. Initial Value Solver (Exponential Map)
Implemented in `src/ham/solvers/geodesic.py` via standard Runge-Kutta 4 integration. Computes `Exp_x(v)` over `t` integrating the Spray dynamically and enforcing manifold projection to counteract ODE drift.

## 5. Module Structure

```text
src/ham/
├── geometry/
│   ├── manifold.py       # Abstract Base Class
│   ├── metric.py         # FinslerMetric ABC & AutoDiff Physics
│   ├── surfaces.py       # Sphere, Torus, Hyperboloid, Paraboloid, EuclideanSpace
│   ├── zoo.py            # Euclidean, Riemannian, Randers, DiscreteRanders
│   ├── mesh.py           # Triangular mesh manifold
│   └── transport.py      # Berwald transport integrator
├── models/
│   └── learned.py        # NeuralRiemannian, NeuralRanders, PullbackRanders,
│                          # PullbackRiemannian, DataDrivenPullbackRanders
├── nn/
│   └── networks.py       # VectorField, PSDMatrixField, RandomFourierFeatures
├── solvers/
│   ├── avbd.py           # BVP solver (Augmented Vertex Block Descent)
│   └── geodesic.py       # IVP solver (ExponentialMap via RK4)
├── training/
│   ├── pipeline.py       # HAMPipeline: multi-phase declarative training
│   └── losses.py         # Modular loss components
├── utils/
│   └── math.py           # safe_norm, numerical stability primitives
└── vis/
    └── hyperbolic.py     # Poincaré disk visualization

research/
├── weinreb/              # Hematopoiesis single-cell data application
└── wildfire/             # Wildfire front propagation simulation
```

## 6. Implementation Status

### Completed & Validated
1.  **Geometry Core:** `metric.py`, `zoo.py`, `surfaces.py`, `mesh.py` — structurally complete with 16 passing test modules.
    * `FinslerMetric` auto-differentiates the energy to correctly derive the geodesic spray.
    * `Randers` and `DiscreteRanders` models implement the Zermelo navigation formula.
2.  **Solvers:** AVBD (BVP) and ExponentialMap (IVP) solvers are validated on Sphere, Torus, Hyperboloid, and triangular meshes.
3.  **Parallel Transport:** Berwald connection verified — norm-preservation on Sphere and non-trivial Randers transport produce correct holonomy.
4.  **Training Pipeline:** `HAMPipeline` supports multi-phase training with per-phase parameter freezing, lineage-triple batching, and modular losses.
5.  **Applications (research folder):**
    *   **Weinreb (bio):** GeometricVAE + DataDrivenPullbackRanders trained on Weinreb hematopoiesis data. Experiments H1-H4 validate geometric topology, directional asymmetry, discriminative cost, and forward predictive simulation.
    *   **Wildfire (sim):** Applications to wildfire front propagation (W1-W3) using mesh metrics and vector fields.

### Known Limitations
6.  **Hyperboloid VAE:** Joint training on complex curved manifolds (Sphere, Hyperboloid) with the full VAE pipeline remains numerically sensitive. The flat `EuclideanSpace` latent space is the recommended default for biological applications.