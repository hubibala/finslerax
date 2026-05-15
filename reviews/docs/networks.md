# Documentation Review: `nn/networks.py`

**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15

## Summary

Overall documentation quality: **needs work**.

The three public classes (`RandomFourierFeatures`, `VectorField`, `PSDMatrixField`) exported via `src/ham/nn/__init__.py` all have docstrings, but every docstring is a one-liner or short sketch with **no argument, return, or raises documentation**. None of them reference the relevant spec sections. The mathematical motivation behind the PSD construction and the Fourier embedding is described only in inline comments, not in the public-facing docstrings where users and mathematicians would look. There are zero docstring examples, and no example script in `examples/` exercises these primitives directly.

## Issue Tracker

| # | Severity | Symbol / Section | Issue | Suggested Text |
|---|----------|-----------------|-------|----------------|
| 1 | **MISSING** | `RandomFourierFeatures.__init__` | No Args documentation. Parameters `in_dim`, `mapping_size`, `scale`, `key` are undocumented. | Add an `Args:` block: `in_dim (int): Dimensionality of the input vector. mapping_size (int): Number of random frequencies (output dim will be 2 * mapping_size). scale (float): Std-dev of the Gaussian from which frequency matrix B is sampled; controls the bandwidth of the kernel approximation. key (jax.Array): JAX PRNG key.` |
| 2 | **MISSING** | `RandomFourierFeatures.__call__` | No Args/Returns documentation. | Add: `Args: x (jnp.ndarray): Input vector of shape (D,). Returns: jnp.ndarray: Fourier-embedded vector of shape (2 * mapping_size,), i.e. [cos(Bx), sin(Bx)].` |
| 3 | **UNCLEAR** | `RandomFourierFeatures` class docstring | States what the mapping does but gives no mathematical or ML-engineering motivation. The phrase "spectral bias" is jargon that a geometer won't understand; conversely, there is no formal statement for ML engineers. | Expand to: `"""Random Fourier Feature (RFF) embedding (Rahimi & Recht, 2007). Maps input x ∈ R^D to a 2M-dimensional feature space via γ(x) = [cos(Bx), sin(Bx)], where B ∈ R^{M×D} is sampled from N(0, scale²I). This approximates a shift-invariant kernel and mitigates spectral bias in coordinate-based MLPs, allowing them to learn high-frequency spatial variation in the metric or wind field."""` |
| 4 | **MISSING** | `VectorField.__init__` | No Args documentation. Parameters `dim`, `hidden_dim`, `depth`, `key`, `use_fourier`, `fourier_scale` are undocumented. | Add a full `Args:` block documenting each parameter with types and defaults. |
| 5 | **MISSING** | `VectorField.__call__` | No Args/Returns documentation. | Add: `Args: x (jnp.ndarray): Point on the manifold, shape (D,). Returns: jnp.ndarray: Vector field value W(x), shape (D,).` |
| 6 | **UNCLEAR** | `VectorField` class docstring | `"Learns a vector field W(x): R^D -> R^D"` — does not explain the role of this network in the Zermelo parameterization (it produces the wind $W^i(x)$ for Randers metrics). A geometer needs the link to `spec/MATH_SPEC.md § 5`; an ML engineer needs to know the output is *not* constrained and that norm clamping happens downstream in `RandersMetric.get_zermelo_data`. | Expand to: `"""Neural-network approximation of a smooth vector field W: R^D → R^D. In the Zermelo parameterization of Randers metrics (see spec/MATH_SPEC.md § 5), this network produces the raw wind field W^i(x); the strong-convexity constraint ‖W‖_h < 1 is enforced downstream by the RandersMetric. Optionally uses Random Fourier Features for improved high-frequency learning."""` |
| 7 | **MISSING** | `PSDMatrixField.__init__` | No Args documentation. Parameters `dim`, `hidden_dim`, `depth`, `key`, `use_fourier` are undocumented. | Add an `Args:` block. Notably, `use_fourier` defaults to `False` here (unlike `VectorField`), and the docstring should explain why ("metric fields are typically smoother than wind fields"). |
| 8 | **MISSING** | `PSDMatrixField.__call__` | No Args/Returns documentation. | Add: `Args: x (jnp.ndarray): Point on the manifold, shape (D,). Returns: jnp.ndarray: Symmetric positive-definite matrix G(x), shape (D, D).` |
| 9 | **UNCLEAR** | `PSDMatrixField` class docstring | The docstring describes the $G = AA^\top + \varepsilon I$ construction but does not explain *why* this is needed (to produce valid Riemannian metrics $h_{ij}(x)$ in the Zermelo data) or cite the spec. The hardcoded $\varepsilon = 10^{-4}$ (`src/ham/nn/networks.py:96`) is mentioned nowhere in the docstring. | Expand to include: (a) reference to `spec/MATH_SPEC.md § 5` ("Riemannian metric (Sea): $h_{ij}(x)$"), (b) explicit statement that epsilon is $10^{-4}$, and (c) note that positive-definiteness is guaranteed by construction. |
| 10 | **INACCURATE** | `PSDMatrixField` class docstring | Docstring says "Symmetric Positive Definite" but the code enforces "Symmetric Positive Definite + $\varepsilon I$." More precisely, the matrix is guaranteed to have eigenvalues $\geq \varepsilon$ (not just $> 0$). The docstring should reflect this regularization. | Suggested: `"G(x) = A(x) A(x)^T + ε I, guaranteeing eigenvalues ≥ ε = 1e-4."` |
| 11 | **MISSING** | `PSDMatrixField` | The hardcoded epsilon value `1e-4` at `src/ham/nn/networks.py:96` is not a constructor parameter and cannot be configured. This design choice is undocumented. | Add a note: `"Note: The regularization constant ε = 1e-4 is hardcoded. To use a different value, subclass and override __call__."` |
| 12 | **MISSING** | Module-level docstring | `networks.py` has no module-level docstring. `spec/ARCH_SPEC.md § 5` lists this module as `nn/networks.py — VectorField, PSDMatrixField, RandomFourierFeatures`. A top-of-file docstring should orient users. | Add: `"""Neural network building blocks for learned Finsler geometry. Provides differentiable parameterizations for vector fields (wind) and positive-definite matrix fields (Riemannian base metrics) used by the learned metric classes in ham.models.learned."""` |
| 13 | **MISSING** | All three classes | No docstring usage examples or pointers to example scripts. The `examples/` directory has no script that directly instantiates these primitives (they are used indirectly via `NeuralRiemannian` / `NeuralRanders` in `models/learned.py`). | Add a brief `Example:` block to each class showing standalone instantiation and a forward call, or add a cross-reference: `"See ham.models.learned.NeuralRanders for typical usage."` |
| 14 | **UNCLEAR** | `VectorField.__init__` | The inline comment `# smoother than relu for gradients` on the `tanh` activation choice (`src/ham/nn/networks.py:48`) is implementation rationale that belongs in the docstring, not hidden in code. Geometers need to know the activation is $C^\infty$; ML engineers need to know it's not ReLU. | Move to docstring: `"Uses tanh activation for C^∞ smoothness, which is important for higher-order autodiff through the spray and Berwald connection."` |
| 15 | **TYPO** | `PSDMatrixField` class docstring | Minor: the docstring says "matrix A of shape (D, D)" — the variable in code is `flat_A` (shape `(D*D,)`) which is *reshaped* to `(D, D)`. Not wrong per se, but could confuse a reader comparing doc to code. | Clarify: `"Network outputs a flat vector of D² elements, reshaped to a D×D factor matrix A."` |

## Coverage Matrix

| Public Symbol | Has Docstring | Args Documented | Returns Documented | Math Notation | Example |
|---|---|---|---|---|---|
| `RandomFourierFeatures` | Yes (1-line) | No | No | Partial (`cos(Bx), sin(Bx)` mentioned) | No |
| `RandomFourierFeatures.__init__` | No | No | N/A | No | No |
| `RandomFourierFeatures.__call__` | No | No | No | No | No |
| `VectorField` | Yes (1-line) | No | No | Partial ($W(x): R^D \to R^D$) | No |
| `VectorField.__init__` | No | No | N/A | No | No |
| `VectorField.__call__` | No | No | No | No | No |
| `PSDMatrixField` | Yes (4-line) | No | No | Partial ($G = A A^T + \varepsilon I$) | No |
| `PSDMatrixField.__init__` | No | No | N/A | No | No |
| `PSDMatrixField.__call__` | No | No | No | No | No |

## Spec Alignment Notes

1. **`spec/ARCH_SPEC.md § 5`** lists this module as `nn/networks.py — VectorField, PSDMatrixField, RandomFourierFeatures`. All three are present and exported via `__init__.py`. ✓

2. **`spec/ARCH_SPEC.md § 3.1`** describes `h_net` as outputting `(B, D, D)` positive definite and `w_net` as outputting `(B, D)`. The docstrings in `networks.py` do not mention batch dimensions at all. The implementations operate on single points `(D,)` and rely on `jax.vmap` upstream — this batch convention discrepancy should be documented to avoid confusion.

3. **`spec/MATH_SPEC.md § 5`** defines the Zermelo parameterization with inputs $h_{ij}(x)$ and $W^i(x)$. `PSDMatrixField` produces $h_{ij}$ and `VectorField` produces $W^i$, but neither docstring references the Zermelo formulation or cites the spec section.

4. **`spec/MATH_SPEC.md § 6.1`** mentions epsilon regularization ($F_\varepsilon$) for numerical stability. The `PSDMatrixField` uses a separate epsilon ($10^{-4}$) for positive-definiteness but does not clarify the relationship (or lack thereof) to the metric epsilon in `§ 6.1`.

5. **`spec/ARCH_SPEC.md § 3`** (`use_fourier` default) — The spec does not prescribe defaults for Fourier features. The code defaults `use_fourier=True` for `VectorField` and `use_fourier=False` for `PSDMatrixField`. This design rationale (wind fields need higher frequencies; metric fields are smoother) exists only as an inline comment (`src/ham/nn/networks.py:68`) and should be surfaced in the docstrings.
