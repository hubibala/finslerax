# RESEARCH_LOG.md: The Holonomic Association Model (HAM)

**Project:** Holonomic Association Model (HAM) / HAMTools
**Date:** December 2, 2025
**Status:** Strategic Pivot to Core Infrastructure (v2.0)

## 1. Executive Summary

We are pivoting from building a specific "World Model" application to developing **`HAMTools`**, a general-purpose, differentiable library for Finsler Geometry.

Our initial experiments (v0.x) successfully demonstrated that Neural Finsler Metrics can learn asymmetric control dynamics (e.g., gravity, wind) that Riemannian models cannot. However, the lack of a rigorous, verified geometric backend makes scaling dangerous. We are pausing the "Moonshot" (AGI World Models) to build the "Rocket Engine" (a reliable, JAX-based geometry library).

---

## 2. The Research Chronology (Pivots)

This project has evolved through four distinct theoretical phases. This narrative forms the basis for the eventual paper's introduction.

### Phase 1: Semantic Geometry (The NLP Era)
* **Hypothesis:** Semantic relationships (A $\to$ B) are asymmetric. Standard embeddings (Dot Product) are symmetric ($A \cdot B = B \cdot A$).
* **Proposal:** Use a **Randers Metric** ($F = \sqrt{g} + \beta$) where $\beta$ represents "Contextual Wind."
* [cite_start]**Outcome:** Theoretically sound, but discrete word tokens provided a noisy, sparse signal for validating continuous differential geometry [cite: 116-117].

### Phase 2: The Physical Pivot (The Robotics Era)
* **Shift:** We moved from "Semantic Similarity" to "Physical Energy."
* **Insight:** "Cost" is equivalent to "Time" or "Work." The axioms of Finsler geometry map perfectly to control theory.
* **Validation:** The "Coriolis Cargo" experiment.
    * *Task:* Robot pushing cargo on a spinning surface.
    * [cite_start]*Result:* The model learned a "Westward Wind" (Coriolis force) purely from observation, achieving **92.4% Cosine Similarity** in zero-shot transfer [cite: 147-150].

### Phase 3: The Architecture Split (Navigator & Pilot)
* **Problem:** End-to-end learning entangled "Geometry" (Planning) with "Control" (Muscle movement).
* **Solution:** Decoupled the architecture.
    * **Navigator (HAM):** Lives in latent space ($S^2$). Plans the vector field.
    * **Pilot (RL):** Lives in tangent space. [cite_start]Executes the vector [cite: 126-127].

### Phase 4: The Rigor Pivot (Current State)
* **Problem:** Our solvers (`AVBD`) and metrics (`RandersFactory`) are entangled with the experimental code. We lack analytical ground truths (e.g., exact Zermelo solutions) to verify if the solver is accurate to $10^{-6}$ or just "learning to cheat."
* **Solution:** Extract the geometry engine into a standalone library (`HAMTools`).
* **Key Technical Change:** Adoption of the **Berwald Connection** over the Chern Connection to correctly model parallel transport of dynamical systems.

---

## 3. Current Technical Status (v1.1)

**3.1. Validated Components**
* **Automatic Spray Generation:** Verified `jax.grad` and `jax.hessian` Euler-Lagrange implicit solver logic.
* **Convexity Protection:** The `tanh` gating on the Randers wind vector $W$ prevents metric collapse.
* **Manifold Implementations:** Spheres, Tori, and meshes undergo AVBD boundary value solving reliably.

**3.2. Current Blockers & Instabilities (Critical Update)**
* **Geometric VAE Failure:** The initial attempt at coupling a Geometric VAE using complex geometries like the `Hyperboloid` in `src/ham/bio/` is fundamentally unstable. The joint training mechanism frequently collapses early in training, generating `nil` values.
* **Geodesic Learning Collapse:** Complex analytical geometries (e.g., Sphere, Hyperboloid) yield very poor cosine similarities during geodesic joint learning tests. The ODE solver collapses or fails to converge properly. The issue is heavily localized around the Hyperboloid formulation and its application/backpropagation interaction in deeper neural tasks.

---

## 5. Artifacts & References

* **Math Spec:** `MATH_SPEC.md` (Version 1.1, Berwald Edition)
* **Architecture:** `ARCH_SPEC.md` (Version 1.0)
* **Legacy Code:** `src/ham/` (To be refactored)
* **Validation Data:** `notebooks/geometry-check.ipynb` (To be expanded)