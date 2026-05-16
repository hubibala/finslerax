# HAMTools Codebase Polishing Plan & Progress Tracker

This document tracks our progress in applying the 161 detailed reviews (Math, Code, Docs, Science) across the HAMTools codebase. We are moving from the bottom-up, ensuring the foundational geometric primitives and math utilities are immaculate before moving up the stack to solvers, neural components, and finally experiments.

## Phase 1: Core Geometry & Utilities
These files form the foundation of the differentiable Finsler geometry framework. We will implement math, code, and doc reviews for each.

- [x] `src/ham/geometry/manifold.py` (Starting Point)
- [x] `src/ham/geometry/metric.py`
- [x] `src/ham/geometry/transport.py`
- [x] `src/ham/geometry/curvature.py`
- [x] `src/ham/geometry/manifolds/` (Reorganized from `surfaces.py`)
- [x] `src/ham/geometry/zoo/` (Reorganized from `zoo.py`)
- [x] `src/ham/geometry/mesh.py`
- [x] `src/ham/utils/math.py`

## Phase 2: Solvers & Fields
Next, we tackle the numerical solvers for geodesics and simulated physics fields.

- [x] `src/ham/solvers/geodesic.py`
- [x] `src/ham/solvers/avbd.py`
- [x] `src/ham/sim/fields.py`

## Phase 3: Neural Architectures & Training
With the geometry and solvers fixed, we polish the neural components, losses, and training pipelines.

- [ ] `src/ham/nn/networks.py`
- [ ] `src/ham/models/learned.py`
- [ ] `src/ham/training/losses.py`
- [ ] `src/ham/training/pipeline.py`

## Phase 4: Biology & Models
These files apply the framework to biological data (RNA velocity) and model architectures (VAE).

- [ ] `src/ham/bio/data.py`
- [ ] `src/ham/bio/check_data.py`
- [ ] `src/ham/bio/vae.py`
- [ ] `src/ham/bio/train_geodesic.py`
- [ ] `src/ham/bio/train_joint.py`
- [ ] `src/ham/bio/train_modular.py`

## Phase 5: Visualization Tools
- [ ] `src/ham/vis/vis.py`
- [ ] `src/ham/vis/hyperbolic.py`

## Phase 6: Test Suite (16 Files)
Tests need to be updated to reflect API changes, fix bugs (e.g., PRNG reuse), and improve coverage (e.g., JAX transforms).

- [x] `tests/test_metric.py` (Completed)
- [x] `tests/test_manifold.py` (Completed)
- [x] `tests/test_transport.py` (Completed)
- [x] `tests/test_curvature.py` (New)
- [x] `tests/test_surfaces.py`
- [x] `tests/test_zoo.py`
- [x] `tests/test_mesh.py`
- [x] `tests/test_geodesic.py`
- [x] `tests/test_avbd.py` (New)
- [x] `tests/test_solver.py`
- [x] `tests/test_fields.py` (New)
- [ ] `tests/test_network.py`
- [ ] `tests/test_learned_metric.py`
- [ ] `tests/test_losses.py` (if applicable)
- [ ] `tests/test_pipeline.py`
- [ ] `tests/test_hyperbolic_vae.py`
- [ ] `tests/test_joint_training.py`
- [ ] `tests/test_geodesic_learning.py`
- [x] `tests/test_mesh_solver.py`
- [x] `tests/test_hyperboloid.py`

## Phase 7: Demos & Examples (13 Files)
Scripts used for demonstration, requiring module docstrings, bug fixes, and proper plotting.

- [ ] `examples/demo_discrete_zermelo.py`
- [ ] `examples/demo_learned_wind.py`
- [ ] `examples/demo_trajectories.py`
- [ ] `examples/demo_vortex.py`
- [ ] `examples/demo_weinreb_vis.py`
- [ ] `examples/demo_zermelo.py`
- [ ] `examples/plot_publication_figs.py`
- [ ] `examples/plot_weinreb_cell_types.py`
- [ ] `examples/plot_weinreb_destinations.py`
- [ ] `examples/preprocess_weinreb.py`
- [ ] `examples/train_vae_ablation.py`
- [ ] `examples/weinreb_smoke_test.py`
- [ ] `examples/weinreb_vae.py`

## Phase 8: Scientific Experiments
The final tier: addressing the experimental methodology, leaks, and uncorrected comparisons.

- [ ] `experiments/weinreb_experiment.py`
- [ ] `experiments/experiment_h1_geometric.py`
- [ ] `experiments/experiment_h2_directional.py`
- [ ] `experiments/experiment_h3_discriminative.py`
- [ ] `experiments/experiment_h4_simulation.py`

---
*Progress updates should be checked off using `[x]` as each file's Math, Code, and Doc reviews are fully addressed.*
