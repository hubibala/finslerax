# Math Review: avbd (Augmented Vertex Block Descent Solver)

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** `src/ham/solvers/avbd.py`

## Summary

The core discrete geodesic energy formulation and block-coordinate descent scheme are **mathematically correct**. The solver properly minimizes $\sum_i E(x_i,\, \log_{x_i}(x_{i+1}))$ via Gauss-Seidel vertex sweeps with manifold-projected gradient descent. However, several declared algorithmic features (augmented Lagrangian dual updates, momentum, convergence tolerance) are **stub implementations** that silently degenerate, and the constraint-violation output is always hard-coded to zero. No sign, index, or formula errors were found in the implemented mathematics.

**Verdict:** Minor Issues — correct core math, incomplete ancillary features.

---

## Formula-by-Formula Audit

### 1. Discrete Energy Functional (`local_action`, lines 97–102)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2, § 2.1
- **Literature Reference:** Absil, Mahony & Sepulchre, *Optimization Algorithms on Matrix Manifolds*, §8.1 (discrete geodesic energy)
- **Implementation:**
  ```python
  def local_action(x_prev, x, x_next):
      v_in = metric.manifold.log_map(x_prev, x)
      v_out = metric.manifold.log_map(x, x_next)
      return metric.energy(x_prev, v_in) + metric.energy(x, v_out)
  ```
- **Verdict:** OK
- **Notes:** For an interior vertex $x_i$, the total discrete energy $\mathcal{E} = \sum_{k} E(x_k, \log_{x_k}(x_{k+1}))$ depends on $x_i$ through exactly two terms: $E(x_{i-1}, \log_{x_{i-1}}(x_i))$ and $E(x_i, \log_{x_i}(x_{i+1}))$. The `local_action` captures precisely these two terms. The energy definition $E(x,v) = \frac{1}{2}F^2(x,v)$ matches `spec/MATH_SPEC.md` § 1.2 and is implemented in `metric.py:37`. Minimizing discrete energy (rather than discrete length) correctly yields constant-speed discrete geodesics by the Cauchy–Schwarz inequality.

---

### 2. Vertex Gradient Descent on Manifold (`update_vertex`, lines 105–132)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.1 (strong convexity of $g_{ij}$)
- **Literature Reference:** Boumal, *An Introduction to Optimization on Smooth Manifolds*, §4.1 (Riemannian gradient descent via retraction)
- **Implementation:**
  ```python
  grad_f = jax.grad(loss_fn)(x)
  grad_tan = metric.manifold.to_tangent(x, grad_f)
  step = -self.step_size * grad_tan
  x_new = metric.manifold.retract(x, step)
  ```
- **Verdict:** OK
- **Notes:** The code computes the Euclidean gradient of the discrete energy, projects it onto $T_xM$ via `to_tangent`, and retracts. For submanifolds of $\mathbb{R}^N$ with the induced metric, orthogonal projection of the ambient gradient gives the Riemannian gradient. This is standard projected gradient descent and is correct. Note that the *optimiser's* metric (ambient Euclidean) differs from the *objective's* metric (Finsler); this is by design — the Finsler energy is encoded in the loss, not the step direction.

---

### 3. Block-Coordinate Descent (Gauss-Seidel sweep, lines 135–149)

- **Spec Reference:** N/A (algorithmic, not in MATH_SPEC)
- **Literature Reference:** Wright, *Coordinate Descent Algorithms*, Mathematical Programming 2015
- **Implementation:**
  ```python
  order = jax.random.permutation(step_key, jnp.arange(n_inner))
  full_order = order + 1

  def scan_body(curr_path_full, idx):
      new_node = update_vertex(curr_path_full, idx - 1, s)
      return curr_path_full.at[idx].set(new_node), None

  new_full, _ = jax.lax.scan(scan_body, full_path, full_order)
  ```
- **Verdict:** OK
- **Notes:** `jax.lax.scan` carries the updated `full_path` forward through the sweep, so when vertex $j$ is updated, it sees the already-updated vertex $i$ if $i$ was processed earlier. This is correct Gauss-Seidel behaviour with randomized coordinate ordering. The index arithmetic is correct: `order` ∈ $\{0,\ldots,n_{inner}-1\}$, `full_order` ∈ $\{1,\ldots,n_{inner}\}$, and `update_vertex(full_path, idx-1, s)` accesses `full_path[idx-1]` (left neighbour), `full_path[idx]` (current), `full_path[idx+1]` (right neighbour). Boundary points at indices 0 and $n_{steps}$ are never modified.

---

### 4. Total Energy Computation (lines 155–158)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2
- **Implementation:**
  ```python
  vels = jax.vmap(metric.manifold.log_map)(full_new[:-1], full_new[1:])
  total_E = jnp.sum(jax.vmap(metric.energy)(full_new[:-1], vels))
  ```
- **Verdict:** OK
- **Notes:** Computes $\mathcal{E} = \sum_{i=0}^{n-1} E(x_i, \log_{x_i}(x_{i+1}))$. Velocity vectors $v_i \in T_{x_i}M$ and energy evaluated at $(x_i, v_i)$. Consistent with the per-vertex `local_action`.

---

### 5. Augmented Lagrangian Penalty (lines 112–118)

- **Spec Reference:** N/A (not in MATH_SPEC; standard constrained optimisation)
- **Literature Reference:** Nocedal & Wright, *Numerical Optimization*, §17.3 (Augmented Lagrangian Method)
- **Implementation:**
  ```python
  c_val = jnp.stack([c(current_x) for c in constraints])
  lam = s.lambdas[idx]
  k = s.stiffness[idx]
  penalty = jnp.sum(lam * c_val + 0.5 * k * (c_val**2))
  ```
- **Verdict:** WARNING
- **Notes:** The penalty term $\sum_j (\lambda_j c_j + \frac{\mu_j}{2} c_j^2)$ matches the standard augmented Lagrangian for equality constraints $c_j(x) = 0$. However:
  1. `lambdas` is initialised to zero (line 87) and **never updated** — the dual ascent step $\lambda \leftarrow \lambda + \mu\, c(x)$ is entirely absent (line 152 contains a comment "Simplified: omitted").
  2. `stiffness` is initialised to 1.0 (line 88) and **never increased** — the adaptive penalty schedule $\mu \leftarrow \rho\, \mu$ is absent.
  3. As a consequence, the augmented Lagrangian degenerates to a pure quadratic penalty $\frac{1}{2}\|c(x)\|^2$, which cannot enforce constraints to arbitrary precision.

  **Recommended Action:** Either implement the dual update loop or document that constraint handling is a pure penalty method (and rename `lambdas` accordingly to avoid confusion).

---

### 6. Constraint Violation Output (lines 160, 200)

- **Spec Reference:** N/A
- **Implementation:**
  ```python
  # In step_fn (line 165):
  max_violation=0.0,
  # In output (line 200):
  constraint_violation=final_state.max_violation
  ```
- **Verdict:** WARNING
- **Notes:** `max_violation` is hard-coded to `0.0` in every iteration. The output `constraint_violation` therefore always reports zero regardless of actual constraint satisfaction. This is misleading for any downstream consumer that checks convergence.

  **Recommended Action:** Compute the actual maximum constraint violation: `max_violation = jnp.max(jnp.abs(c_vals))` when constraints are present, or `0.0` when no constraints are specified.

---

### 7. Momentum (declared but unused, lines 46, 85, 131)

- **Spec Reference:** N/A
- **Implementation:**
  ```python
  momentum: float = 0.5          # line 46, declared
  prev_path: jnp.ndarray         # line 22, stored in state
  # line 131: update is purely gradient, no momentum term
  step = -self.step_size * grad_tan
  ```
- **Verdict:** WARNING
- **Notes:** The `momentum` parameter and `prev_path` state field suggest a Polyak heavy-ball scheme $x_{k+1} = x_k - \alpha \nabla f + \beta (x_k - x_{k-1})$, but the update (line 131) is plain gradient descent. The stored `prev_path` is written (line 164) but never read. Dead code/parameters create a false API contract.

  **Recommended Action:** Either implement manifold-aware momentum (using parallel transport of the previous step, or a retraction-based scheme) or remove the `momentum` parameter and `prev_path` field.

---

### 8. Convergence Tolerance (declared but unused, lines 44–45, 170–171)

- **Spec Reference:** N/A
- **Implementation:**
  ```python
  tol: float = 1e-4              # line 44
  energy_tol: float = 1e-4       # line 46
  # line 170 (while_loop cond):
  def cond(s): return s.step < self.iterations
  ```
- **Verdict:** INFO
- **Notes:** `tol` and `energy_tol` are declared but never used in any convergence check. The `while_loop` condition (inference mode, line 170) is a pure step count, identical to the `scan` (train mode). The comment "convergence based, strictly for inference" is inaccurate.

  **Recommended Action:** Implement early stopping, e.g.: `return (s.step < self.iterations) & (jnp.abs(s.curr_energy - s.prev_energy) > self.energy_tol)`.

---

### 9. Gradient Clipping (lines 126–128)

- **Spec Reference:** `spec/MATH_SPEC.md` § 6.1 (numerical stability)
- **Implementation:**
  ```python
  grad_norm = safe_norm(grad_tan)
  clip_value = 10.0
  grad_tan = jnp.where(grad_norm > clip_value, grad_tan * (clip_value / grad_norm), grad_tan)
  ```
- **Verdict:** OK
- **Notes:** Standard gradient norm clipping. The threshold of 10.0 is a magic number but reasonable. Uses `safe_norm` (which handles the $v=0$ case via $\sqrt{\max(\|x\|^2, \epsilon)}$). This is mathematically benign — it bounds the step size without altering the descent direction.

---

### 10. Linear Initialisation with Noise (lines 67–76)

- **Spec Reference:** N/A
- **Implementation:**
  ```python
  linear_path = (1 - t) * p_start + t * p_end
  noise = jax.random.normal(k1, shape=linear_path.shape) * 1e-4
  linear_path = linear_path + noise
  path_guess = jax.vmap(metric.manifold.project)(linear_path)
  ```
- **Verdict:** OK
- **Notes:** Linear interpolation in ambient space followed by manifold projection is standard for BVP initialisation. The $O(10^{-4})$ noise perturbation prevents degenerate zero-velocity segments when $p_{start} = p_{end}$, which would cause $F(x,0)$ and $\nabla_v E$ singularities per `spec/MATH_SPEC.md` § 6.1. Projection after noise addition ensures the initialisation lies on $M$.

---

### 11. Output Velocity Convention (lines 193–194)

- **Spec Reference:** `spec/MATH_SPEC.md` § 1.2, § 2.1
- **Implementation:**
  ```python
  velocities = jax.vmap(metric.manifold.log_map)(full_path[:-1], full_path[1:])
  ```
- **Verdict:** OK
- **Notes:** Computes $v_i = \log_{x_i}(x_{i+1}) \in T_{x_i}M$ for each segment. The output `Trajectory.vs` has shape $(n_{steps}, D)$ while `Trajectory.xs` has shape $(n_{steps}+1, D)$. Convention is consistent: `vs[i]` is the departure velocity at `xs[i]`.

---

## Open Questions

1. **Log map fidelity:** The discrete energy quality depends on the manifold's `log_map` implementation. The default `Manifold.log_map` (`manifold.py:134–148`) uses a scaled tangent projection, which is only first-order accurate for curved manifolds. For manifolds with exact log maps (Hyperboloid, Sphere), this is overridden. A user supplying a manifold with only the default log map should be aware that the discrete geodesic may not converge to the true continuous geodesic under mesh refinement beyond first order. Is this documented?

2. **Step size and convergence guarantees:** The fixed step size $\alpha = 0.05$ (line 41) with gradient clipping at 10.0 gives an effective maximum step of 0.5. For strongly convex segments of the discrete energy, convergence is guaranteed for $\alpha < 2/L$ where $L$ is the local Lipschitz constant of the gradient. No adaptive step size (Armijo, Wolfe) is implemented. Is convergence for all supported metrics empirically validated?

3. **Randomised vs. deterministic ordering:** The randomised Gauss-Seidel ordering is non-deterministic across calls (seeded from input data). For reproducibility in scientific experiments, should there be a deterministic sweep option (e.g., cyclic ordering)?
