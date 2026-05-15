# Math Review: test_mesh.py

**Reviewer:** Math Reviewer Agent  
**Date:** 2025-05-15  
**Spec Version:** 1.1.0 (Berwald Revision)  
**Source:** `tests/test_mesh.py`  
**Cross-referenced:** `src/ham/geometry/mesh.py`

## Summary

**Verdict: Minor Issues.** All expected values and mathematical assertions in the test file are analytically correct. The tests verify orthogonal projection, tangent-space projection via Gram-Schmidt, and surface sampling for piecewise-linear triangular meshes. Two WARNING-level findings concern incomplete geometric coverage of edge/vertex projection cases and the lack of an in-triangle containment check for random samples. One NOTE concerns the conservatism of the tolerances.

---

## Formula-by-Formula Audit

### 1. Tetrahedron Closest-Face Projection (`test_standard_3d_tetrahedron`, lines 21–24)

- **Spec Reference:** Not in `spec/MATH_SPEC.md` (discrete geometry, no continuous Finsler analog).
- **Literature Reference:** Eberly, *Distance Between Point and Triangle in 3D*, Geometric Tools (2021).
- **Implementation:**
  Query point $p = (0.2,\, 0.2,\, -0.5)$ projected onto a tetrahedron with faces
  $\{[0,1,2],\,[0,1,3],\,[0,2,3],\,[1,2,3]\}$.
  Expected result: $(0.2,\, 0.2,\, 0.0)$.

  **Verification (all four faces):**

  | Face | Plane | Barycentric interior? | Closest point | $d^2$ |
  |------|-------|-----------------------|---------------|-------|
  | $[0,1,2]$ | $z = 0$ | Yes ($s=0.2,\, t=0.2,\, s+t=0.4 \le 1$) | $(0.2,\, 0.2,\, 0)$ | $0.25$ |
  | $[0,1,3]$ | $y = 0$ | No ($z$-coord $= -0.5 < 0$) | Edge $(0.2,\, 0,\, 0)$ | $0.29$ |
  | $[0,2,3]$ | $x = 0$ | No ($z$-coord $= -0.5 < 0$) | Edge $(0,\, 0.2,\, 0)$ | $0.29$ |
  | $[1,2,3]$ | $x+y+z=1$ | No (projected point outside) | Edge/vertex | $> 0.25$ |

  Face $[0,1,2]$ achieves the minimum distance $d = 0.5$. Expected value is correct.

- **Verdict:** CORRECT

---

### 2. Tangent-Space Projection on XY Face (`test_standard_3d_tetrahedron`, lines 27–30)

- **Spec Reference:** Not in `spec/MATH_SPEC.md`. Standard orthogonal projection onto the tangent plane of a piecewise-linear surface.
- **Literature Reference:** Standard Gram-Schmidt in $\mathbb{R}^N$; see e.g., Botsch et al., *Polygon Mesh Processing*, §1.
- **Implementation:**
  At $x = (0.2,\, 0.2,\, 0)$ on face $[0,1,2]$, with input $v = (1,\, 1,\, 1)$, the expected tangent projection is $(1,\, 1,\, 0)$.

  The face edge vectors are $\vec{u} = (1,0,0)$, $\vec{w} = (0,1,0)$. Gram-Schmidt yields:
  $$e_1 = (1,0,0), \quad e_2 = (0,1,0)$$
  $$v_{\tan} = (v \cdot e_1)\,e_1 + (v \cdot e_2)\,e_2 = 1 \cdot (1,0,0) + 1 \cdot (0,1,0) = (1,1,0)$$

- **Verdict:** CORRECT

---

### 3. High-Dimensional Projection ($\mathbb{R}^4$) (`test_high_dim_embedding`, lines 48–50)

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Implementation:**
  Single triangle with vertices $A=(0,0,0,0)$, $B=(1,0,0,0)$, $C=(0,1,0,0)$ in $\mathbb{R}^4$.
  Query $p = (0.2,\, 0.2,\, 0,\, 10)$, expected projection $(0.2,\, 0.2,\, 0,\, 0)$.

  Gram matrix: $\vec{ab} = (1,0,0,0)$, $\vec{ac} = (0,1,0,0)$, $\vec{ap} = (0.2,\, 0.2,\, 0,\, 10)$.
  $$d_1 = 0.2,\; d_2 = 0.2,\; d_3 = 1,\; d_4 = 0,\; d_5 = 1$$
  $$\det = 1 \cdot 1 - 0 = 1, \quad s = 0.2, \quad t = 0.2$$
  Interior check: $s + t = 0.4 \le 1$ ✓. Result: $A + 0.2\,\vec{ab} + 0.2\,\vec{ac} = (0.2,\, 0.2,\, 0,\, 0)$.

- **Verdict:** CORRECT

---

### 4. High-Dimensional Tangent Projection ($\mathbb{R}^4$) (`test_high_dim_embedding`, lines 54–56)

- **Implementation:**
  At $(0.2,\, 0.2,\, 0,\, 0)$ with $v = (1,1,1,1) \in \mathbb{R}^4$, expected tangent projection $(1,1,0,0)$.

  Edge vectors in $\mathbb{R}^4$: $\vec{u} = (1,0,0,0)$, $\vec{w} = (0,1,0,0)$. These are already orthonormal, so Gram-Schmidt gives $e_1 = (1,0,0,0)$, $e_2 = (0,1,0,0)$.
  $$v_{\tan} = 1 \cdot (1,0,0,0) + 1 \cdot (0,1,0,0) = (1,1,0,0)$$

- **Verdict:** CORRECT

---

### 5. Random Sampling on Surface (`test_high_dim_embedding`, lines 59–62)

- **Spec Reference:** Not in `spec/MATH_SPEC.md`.
- **Literature Reference:** Osada et al., *Shape Distributions*, ACM TOG 21(4), 2002, §4.2 (uniform triangle sampling via fold-over of unit square).
- **Implementation (in `mesh.py`, lines 95–108):**
  The code draws $(r_1, r_2) \sim \text{Uniform}([0,1]^2)$ and applies the standard fold-over:
  $$
  (r_1, r_2) \mapsto \begin{cases} (r_1, r_2) & \text{if } r_1 + r_2 \le 1 \\ (1-r_1, 1-r_2) & \text{if } r_1 + r_2 > 1 \end{cases}
  $$
  then forms $p = (1 - r_1 - r_2)\,A + r_1\,B + r_2\,C$. This produces a uniform distribution over the triangle. The area-weighted face selection (line 100) uses $\text{Area} = \tfrac{1}{2}\sqrt{\|\vec{u}\|^2\|\vec{v}\|^2 - (\vec{u}\cdot\vec{v})^2}$, which is the correct generalization of the cross-product magnitude via Lagrange's identity to $\mathbb{R}^N$.

  **Test assertion:** checks `samples.shape == (10, 4)` and `samples[:, 2:] ≈ 0` with `atol=1e-5`.

- **Verdict:** WARNING
- **Notes:** The test verifies that samples have zero off-plane coordinates but does **not** verify that samples satisfy the in-triangle constraint $x_1 \ge 0,\; x_2 \ge 0,\; x_1 + x_2 \le 1$. A sampling bug that places points outside the triangle (e.g., if the fold-over reflection were omitted or inverted) would pass this test undetected.
- **Recommended Action:** Add an assertion verifying that the barycentric coordinates of all samples are non-negative and sum to at most 1. For the specific test triangle ($A = 0$, $B = e_1$, $C = e_2$), this reduces to checking `samples[:, 0] >= 0`, `samples[:, 1] >= 0`, and `samples[:, 0] + samples[:, 1] <= 1`.

  `tests/test_mesh.py:62`

---

### 6. Missing Edge/Vertex Projection Test Cases

- **Spec Reference:** The projection algorithm in `src/ham/geometry/mesh.py:37–48` has two branches: interior barycentric projection and edge-segment fallback.
- **Verdict:** WARNING
- **Notes:** Every projection test in the file uses a query point whose closest point lies in the **interior** of a face (barycentric $s \ge 0$, $t \ge 0$, $s + t \le 1$). The edge-projection branch (`project_segment`, `mesh.py:37–43`) and the vertex-proximity case are never exercised. A sign error or clipping mistake in the edge projection code would be invisible to this test suite.
- **Recommended Action:** Add at least one test where the nearest point lies on a triangle **edge** (e.g., query $(0.5,\, -0.1,\, 0)$ → expected $(0.5,\, 0,\, 0)$ on edge $[0,1]$ of face $[0,1,2]$) and one where it lies on a **vertex** (e.g., query $(-0.1,\, -0.1,\, -0.1)$ → expected $(0,\, 0,\, 0)$).

  `tests/test_mesh.py:16–30`

---

### 7. Tolerance Levels

- **Implementation:** All `assert_allclose` calls use `atol=1e-5` with 64-bit arithmetic (`jax_enable_x64=True`, line 5).
- **Verdict:** NOTE
- **Notes:** For the axis-aligned geometries in these tests, the projection and Gram-Schmidt results are representable exactly (or to machine epsilon $\approx 10^{-16}$) in double precision. The tolerance $10^{-5}$ is safe but conservative by roughly 10 orders of magnitude. This is not incorrect—loose tolerances never cause false negatives—but it means the tests would not detect a subtle precision regression (e.g., accidental use of `float32`).

  `tests/test_mesh.py:23,24,29,30,50,56,62`

---

## Open Questions

1. **Tilted-triangle coverage.** All test triangles are axis-aligned, making Gram-Schmidt trivial. A triangle with an oblique normal (e.g., vertices $(0,0,0)$, $(1,0,0)$, $(0.5, 0.5, 0.7)$) would exercise the full Gram-Schmidt orthogonalization path and the non-trivial cross terms in the Gram matrix determinant. Should such a test be added?

2. **Retract and face-weight tests.** The methods `retract()` and `get_face_weights()` in `mesh.py` are untested. `get_face_weights` uses a softmax with a hardcoded temperature of 100.0; its mathematical behaviour at face boundaries (where two faces are equidistant) is not tested and may exhibit discontinuities or non-unique maxima.
