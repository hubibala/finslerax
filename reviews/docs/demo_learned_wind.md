# Documentation Review: `examples/demo_learned_wind.py`
**Reviewer:** Doc Reviewer Agent
**Date:** 2026-05-15

## Summary
Overall documentation quality: **needs work**.

The script demonstrates an inverse problem — learning a Randers wind field from trajectory data on the sphere — but lacks a module-level docstring, omits mathematical context for Zermelo navigation, and leaves several hyperparameter choices and loss-term weights unjustified. A mathematician would struggle to connect the code to `spec/MATH_SPEC.md § 5`; an ML engineer would not know why this loss function structure was chosen over alternatives. Step numbering is inconsistent (jumps from 3 to 5), and the script accesses the private method `_get_zermelo_data` without explanation.

## Issue Tracker

| # | Severity | Location | Issue | Recommended Action |
|---|----------|----------|-------|--------------------|
| 1 | **MISSING** | `examples/demo_learned_wind.py:1` | No module-level docstring. The file's purpose, mathematical background, and expected output are undocumented. Other demos (e.g. `demo_zermelo.py`) also lack docstrings, but as a *learned* example this script requires more exposition — it is the primary tutorial for the metric-learning workflow. | Add a docstring at the top of the file, e.g.: `"""Inverse Zermelo Problem on the Sphere. Demonstrates learning a Randers wind field W(x) from sampled velocity data using NeuralRanders. The ground-truth wind is a scaled Rossby–Haurwitz wave (R=3). Three loss components are combined: energy loss (physics), metric regularization (anchor H to identity), and Jacobian regularization (smooth W). See spec/MATH_SPEC.md § 5 for the Zermelo parameterization."""` |
| 2 | **MISSING** | `examples/demo_learned_wind.py:23` | The Rossby–Haurwitz wave parameters `R=3, omega=1.0` are set without any comment explaining what they control or why these values were chosen. | Add a brief comment, e.g.: `# Rossby-Haurwitz wave with wavenumber R=3 (tricellular pattern) and angular velocity omega=1.0` |
| 3 | **MISSING** | `examples/demo_learned_wind.py:25` | The scaling factor `0.8` in `w_true = lambda x: 0.8 * true_wind_fn(x)` is undocumented. The inline comment says "True magnitude is 0.8" but does not explain *why* this value was chosen or what constraint it satisfies ($\|W\|_h < 1$ from `spec/MATH_SPEC.md § 5`). | Expand the comment: `# Scale to 0.8 to satisfy the Randers convexity constraint ||W||_h < 1 (MATH_SPEC § 5)` |
| 4 | **MISSING** | `examples/demo_learned_wind.py:28-30` | The data generation section (`N_samples = 512`) does not explain the sampling strategy. Why 512? Are these points uniformly distributed on the sphere? What is the velocity being sampled (tangent-space or ambient)? | Add a comment clarifying the sampling approach and the role of `sphere.random_sample`. |
| 5 | **UNCLEAR** | `examples/demo_learned_wind.py:37-38` | `NeuralRanders` is instantiated with `hidden_dim=32, use_fourier=True`, but neither parameter is explained. The following comment "We keep Fourier features for the topology, but tame them with regularization" is vague — a mathematician would not know what "Fourier features" means in this context, and an ML engineer would not know why they help with spherical topology. | Replace with a concrete explanation: `# Fourier random features provide frequency-aware basis functions that better capture the periodic structure of spherical fields. hidden_dim=32 is sufficient for the R=3 wave. Regularization (below) prevents overfitting high-frequency modes.` |
| 6 | **UNCLEAR** | `examples/demo_learned_wind.py:44-46` | The "Energy Loss" comment says "(The Physics)" but does not explain what `m.energy(x, v)` computes or why minimizing it recovers the wind field. Mathematically, this is the Finsler energy $E(x,v) = \frac{1}{2}F^2(x,v)$, and its minimization over the Randers parameters encodes the inverse Zermelo problem, but none of this is stated. | Add: `# Finsler energy E(x,v) = ½F²(x,v). For the true Randers metric, E=½ at unit-speed trajectories. Minimising the mean energy over samples drives the learned metric toward the ground truth (inverse Zermelo problem, MATH_SPEC § 5).` |
| 7 | **UNCLEAR** | `examples/demo_learned_wind.py:49-51` | The metric regularization loss `(H_vals - I)**2` anchors $H(x)$ to the identity. The comment says "(The Anchor)" which is cryptic. It does not explain that for the ground-truth wind the underlying Riemannian metric $h_{ij}$ is Euclidean (i.e., identity), and that this regularization encodes that prior. | Expand: `# Prior: the underlying Riemannian metric h(x) should be close to the Euclidean identity, since the ground-truth wind is defined on a round sphere with h=I.` |
| 8 | **MISSING** | `examples/demo_learned_wind.py:54-60` | The Jacobian regularization section computes the Jacobian of the wind field $\partial W / \partial x$, but does not explain why this is needed. The comment says "(The Smoother)" and "kills the ripples" — too informal for either audience. There is no mention of the mathematical rationale (e.g., Sobolev-type smoothness penalty on $W$). | Rewrite to: `# Jacobian regularization: penalise ||dW/dx||² (an H¹-Sobolev smoothness prior) to suppress high-frequency oscillations that Fourier features can introduce.` |
| 9 | **MISSING** | `examples/demo_learned_wind.py:63-65` | Loss weights `1.0` (h_reg) and `0.1` (smooth) are chosen without justification. A user trying to adapt this demo would not know how to tune them. | Add a comment documenting the rationale: `# Weights: energy (implicit 1.0) drives fitting; h_reg=1.0 strongly anchors H≈I; smooth=0.1 gently penalises W oscillations. Increase smooth if learned W shows high-frequency artifacts.` |
| 10 | **UNCLEAR** | `examples/demo_learned_wind.py:71` | Training loop runs 5001 steps. The comment says "Increased steps slightly to allow settling" — settling from what baseline? This is unexplained iteration-history commentary that will confuse readers. | Replace with: `# 5000 training steps with Adam(lr=2e-3). Monitor loss convergence; increase if the held-out cosine similarity is below 0.95.` |
| 11 | **MISSING** | `examples/demo_learned_wind.py:78-82` | The visualization section uses `generate_icosphere(radius=1.0, subdivisions=2)` without explaining why an icosphere grid is used or what `subdivisions=2` means for grid density. | Add: `# Evaluate on a uniform icosphere grid (subdivisions=2 → ~162 points) for visually balanced coverage.` |
| 12 | **UNCLEAR** | `examples/demo_learned_wind.py:84-87` | The script accesses `learner._get_zermelo_data(x)` (a private method) to extract the learned wind. No comment explains why the public API is not used or what `_get_zermelo_data` returns (H, W, λ). This is confusing for ML engineers who expect to interact with public APIs only. | Either (a) add a comment: `# Extract Zermelo data (H, W, λ) via private API; W is the learned wind vector field`, or (b) prefer the public `w_net` attribute if available. |
| 13 | **MISSING** | `examples/demo_learned_wind.py:89-99` | The evaluation metrics (cosine similarity, MSE) are computed but not explained. What do they measure? What values indicate success? There is no documented acceptance criterion. | Add: `# Cosine similarity ≈ 1.0 means learned and true winds are directionally aligned. MSE measures magnitude accuracy. Expect cos_sim > 0.95 and MSE < 0.01 for convergence.` |
| 14 | **TYPO** | `examples/demo_learned_wind.py:16` | Print statement says "Iterating: Adding Jacobian Regularization to fix vector lengths." — this reads like a developer log entry, not a user-facing description. | Replace with a descriptive message: `"Learning Randers wind field with energy + smoothness losses"` |
| 15 | **INACCURATE** | `examples/demo_learned_wind.py:76` | Step numbering jumps from `# 3. Learner` (line 36) to `# 5. Visualization` (line 76). Step 4 is missing. | Renumber to `# 4. Visualization` or add a missing `# 4. Training` header before the training loop (line 70). |
| 16 | **MISSING** | `examples/demo_learned_wind.py:40` | `optimizer = optax.adam(learning_rate=2e-3)` — the learning rate is not justified. | Add: `# lr=2e-3 balances convergence speed and stability for this problem scale.` |

## Coverage Matrix

| Public Symbol / Section | Has Docstring / Comment | Args Documented | Returns Documented | Math Notation | Example |
|------------------------|------------------------|-----------------|-------------------|---------------|---------|
| Module-level | No | N/A | N/A | No | N/A |
| `main()` | No | N/A | N/A | No | N/A |
| Step 1: Ground Truth | Partial (inline) | Partial | N/A | No | N/A |
| Step 2: Data Generation | Minimal | No | N/A | No | N/A |
| Step 3: Learner init | Partial (inline) | No | N/A | No | N/A |
| `train_step` (inner) | No | No | No | No | N/A |
| Loss: Energy | Partial ("The Physics") | No | N/A | No | N/A |
| Loss: H regularization | Partial ("The Anchor") | No | N/A | No | N/A |
| Loss: Jacobian reg | Partial ("The Smoother") | No | N/A | No | N/A |
| Visualization | Minimal | No | N/A | No | N/A |
| Evaluation metrics | No | No | N/A | No | N/A |

## Spec Alignment Notes

1. **`spec/MATH_SPEC.md § 5` (Zermelo Parameterization):** The script implements the inverse Zermelo problem but never references this section or explains the mathematical relationship between energy minimization and wind recovery. The convexity constraint $\|W\|_h < 1$ is implicitly satisfied by the 0.8 scaling but not documented.

2. **`spec/ARCH_SPEC.md § 3.1` (Randers Specialization):** The demo uses `NeuralRanders` which inherits from `Randers` and exposes `_get_zermelo_data`. The script should reference the public API path (`ham.models.learned.NeuralRanders`) and explain how it fits into the metric hierarchy.

3. **`spec/ARCH_SPEC.md § 1` (Design Philosophy — Metric-First):** The demo is a good illustration of the metric-first principle (learning the metric from data), but this connection to the library's design philosophy is never made explicit. A one-line mention would help both audiences.

4. **Notation consistency:** The spec uses $h_{ij}(x)$ for the Riemannian component and $W^i(x)$ for the wind. The code uses `H_vals`, `h_net`, and `W` — the mapping between these is never stated.
