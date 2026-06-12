# Design Document: Differentiable Waddington Navigators via EBM-Finsler Geometry

## 1. Executive Summary

This document outlines the design and mathematical foundations for a novel trajectory inference framework within the Holonomic Association Model (HAM) library. By combining Energy-Based Models (EBMs) with asymmetric Randers (Finsler) geometry, we aim to compute continuous, directed, and biologically plausible cell differentiation trajectories.

Unlike current State-of-the-Art (SOTA) methods that rely on reversible Neural ODEs or discrete Optimal Transport, this approach formulates trajectory inference as a Boundary Value Problem (BVP) on a learned energy landscape, effectively creating a differentiable, mathematically rigorous Waddington's landscape. To ensure numerical stability and avoid the "Finsler collapse" in high dimensions, the framework operates in a dense, truncated PCA space.

---

## 2. Literature and State-of-the-Art Context

The proposed approach positions HAM to solve critical limitations in the current trajectory inference landscape:

1. **The Target Benchmark:** *Weinreb et al. (2020)* introduced a scRNA-seq dataset of hematopoiesis with clonal barcodes, providing empirical ground truth for lineage trajectories. This is the primary validation target for our framework.
2. **Optimal Transport (Mass vs. Path):** *Schiebinger et al. (2019)* popularized Waddington-OT, which computes transition probabilities between cell populations across timepoints. However, OT provides probabilistic couplings, not continuous, individual cell-level paths.
3. **Neural ODEs (IVPs vs. BVPs):** *Tong et al. (2020)* introduced TrajectoryNet, using continuous normalizing flows and Neural ODEs. These solve Initial Value Problems (IVPs), making it difficult to explicitly query the path between *Cell A* and *Cell B*. Furthermore, standard ODEs are reversible, failing to capture the strict asymmetry of biological differentiation.
4. **The Dimensionality Limit of Stochastic Metrics:** As demonstrated by *Pouplin et al. (2023)*, when learning a stochastic pullback metric (e.g., via Gaussian Processes or VAEs) in high dimensions, the variance shrinks and the expected Finsler norm collapses into a standard Riemannian norm. To preserve the asymmetric advantages of Finsler geometry and avoid the severe curvature distortions of VAEs, we map the data to a dense PCA space where the base metric is flat Euclidean, allowing the directed energy gradient to dominate the path routing.

---

## 3. Mathematical Formulation

### 3.1. State Space Reduction (PCA)

Operating in the raw 5000+ dimensional scRNA-seq space is sparse and computationally intractable for geodesic solvers. Operating in a VAE latent space introduces massive curvature distortion ($g = J^T J$) that crashes the Euler-Lagrange integrators.

Instead, we project the data into a $d$-dimensional PCA space ($d \approx 30$ to $50$):


$$x \in \mathbb{R}^d = U^T x_{raw}$$


Because PCA is a linear projection, the Jacobian is constant, and the induced base metric is uniformly flat (the Euclidean identity matrix).

### 3.2. Phase 1: The Energy-Based Model (EBM)

We learn a scalar energy field $E_\theta(x)$ over the PCA space representing the biological potential (Waddington's altitude), where the data distribution is defined as:


$$p_\theta(x) = \frac{e^{-E_\theta(x)}}{Z(\theta)}$$


This model is trained decoupled from the geometric solver using standard Score Matching or Flow Matching techniques. Mature, high-density cell states occupy low-energy valleys, while stem cells occupy high-energy peaks.

### 3.3. Phase 2: The EBM-Induced Randers Metric

We define an asymmetric Finsler metric of the Randers type to govern travel through the PCA space.


$$F(x, v) = \sqrt{h_x(v, v)} + \langle W(x), v \rangle$$

* **The Sea (Riemannian Base):** Because PCA is a linear transformation, we set the base metric to the identity: $h_x(v, w) = v^T I w$.
* **The Wind (EBM Gradient):** We define the directed biological flow as the negative gradient of the energy field:

$$W_{raw}(x) = -\lambda \nabla_x E_\theta(x)$$


* **The Squasher:** To maintain strong convexity and valid Finsler geometry, we enforce Zermelo's constraint ($\|W\|_h < 1$) using HAM's built-in causality-preserving wind squasher:

$$W(x) = \text{Squash}(W_{raw}(x))$$



### 3.4. Phase 3: Trajectory Routing (AVBD Solver)

Given a source stem cell $x_A$ and a target mature cell $x_B$, we formulate a Boundary Value Problem (BVP). We seek the curve $\gamma(t)$ that minimizes the Finsler energy functional:


$$\mathcal{E}[\gamma] = \int_0^1 \frac{1}{2} F(\gamma, \dot{\gamma})^2 dt$$


Because $F(x,v) \neq F(x, -v)$, the geodesic will naturally "flow" down the EBM gradient toward the mature state. Uphill travel against the gradient is heavily penalized. We solve this using HAM's `AVBDSolver` (Augmented Vertex Block Descent), which iteratively relaxes the path to satisfy the Euler-Lagrange equations.

---

## 4. Implementation Gap Analysis

To execute this experimental suite, several additions must be made to the current `hubibala/ham` repository structure.

### 4.1. Missing Models and Geometry Components

* **`src/ham/nn/ebm.py` (NEW):** We need a module for training standard Energy-Based Models or Score Networks in JAX/Equinox. Currently, `ham.nn.networks` only handles vector fields and PSD matrix fields.
* **`src/ham/models/learned.py` (UPDATE):** * Create an `EnergyBasedRanders(FinslerMetric)` class.
* This class must take a pretrained scalar EBM, auto-differentiate it via `jax.grad` to extract the wind field $W(x)$, apply the norm squasher, and use a flat Euclidean base metric.



### 4.2. Missing Data Preprocessing Pipeline

* **`research/weinreb/preprocess_weinreb.py` (UPDATE):**
* The current pipeline seems geared toward VAE latent extraction.
* We must implement a robust PCA pipeline that saves the PCA projection matrix ($U$) and the mean vector, ensuring we can project generated trajectories back into raw gene space for biological validation.



### 4.3. Training Pipeline Modifications

* **`src/ham/training/pipeline.py` (UPDATE):** * The current `HAMPipeline` expects to train the metric end-to-end.
* We need explicit support for **Decoupled Training**: Phase 1 trains the `EBM` module purely on density matching (Score Matching loss). Phase 2 *freezes* the EBM, instantiates the `EnergyBasedRanders` metric, and optionally fine-tunes a scaling parameter $\lambda$ using a geodesic alignment loss against the Weinreb barcodes.



### 4.4. Experimental Suite (`research/weinreb/`)

* **`experiment_h5_ebm_finsler.py` (NEW):** A new top-level experiment script that coordinates the PCA -> EBM -> AVBD pipeline.
* **Evaluation Metrics:** We currently lack robust biological evaluation metrics to compare the AVBD output to the SOTA. We need to implement:
* *Lineage Alignment Score:* How well does the computed geodesic overlap with the empirical barcode clones?
* *Directionality Score:* Verify that the AVBD solver successfully breaks symmetry (Path $A \to B$ is biologically valid, Path $B \to A$ is heavily penalized).