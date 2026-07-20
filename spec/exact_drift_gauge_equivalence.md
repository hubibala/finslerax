# The Exact-Drift Gauge: Projective Invariance, Potential Shaping, and Hodge Identifiability

> A self-contained note on a single proposition: **the exact (gradient) part of an
> additive drift is a gauge freedom of optimal behaviour.** It appears as *projective
> invariance* of Finsler geodesics, as *potential-based reward shaping* in optimal
> control, and generally as the *exact component* of a Hodge decomposition, whose
> identifiability is governed by an observation-channel duality. The first two are the
> **same theorem** under a dictionary; the third is the same principle with a twist.

Date: 2026-07-01. Prerequisites: Riemannian/Finsler geometry, dynamic programming,
Hodge/Helmholtz decomposition. No domain-specific content.

---

## 0. The proposition

Let a base "kinetic" cost be augmented by an additive drift term. Then:

> **Proposition (exact-drift gauge).** Adding the differential of a potential to the
> cost leaves the set of optimal trajectories invariant. Consequently, recovering the
> cost from optimal-behaviour observations determines it only *modulo* an exact
> 1-form — the potential is a gauge freedom. The identifiable content is the drift's
> non-exact (rotational) part. The gauge is broken only by a *complementary* observation
> channel (arc-length parametrization / cost magnitude), which restores rigidity.

This statement has three faces:

| Face | The invisible transform | The invariance theorem |
|------|-------------------------|------------------------|
| **F1. Finsler geometry** | add a closed 1-form `β = dU` to a Randers metric | geodesics unchanged *as point sets* (projective invariance) |
| **F2. Optimal control** | potential shaping `c → c + Φ(x') − Φ(x)` | optimal policy unchanged (uniquely so) |
| **F3. Hodge / transport** | shift the gradient part of a drift field | argmin-behaviour sees only the rotational part |

F1 and F2 are a **tight isomorphism** (§3): same telescoping proof, term-by-term
dictionary, matching scaling gauge. F3 is the **same Hodge/gauge skeleton in an
inference setting** (§4), with the subtlety that *which* Hodge component is the gauge
**flips with the observation channel** — a duality that turns the diagnosis into a
constructive recovery recipe.

---

## 1. Setup and notation

Let `M` be a smooth manifold with a base Riemannian metric `g` and norm
`α(x,v) = √(g_x(v,v))`. A **1-form** `β = β_i(x)\,dx^i` acts on velocities; its line
integral along a curve `γ` is `∫_γ β = ∫ β_{γ(t)}(γ'(t))\,dt`. Recall:

- `β` is **closed** if `dβ = 0`; **exact** if `β = dU` for a scalar potential `U`.
- (Poincaré) On a simply-connected domain, closed ⟺ exact. On a manifold with
  `H^1(M) ≠ 0` (e.g. a torus), there exist closed-but-not-exact **harmonic** forms.
- **Helmholtz–Hodge:** any 1-form (or vector field) decomposes orthogonally as
  `β = dU  ⊕  δω  ⊕  h`  — exact ⊕ co-exact (rotational) ⊕ harmonic.
- **Fundamental telescoping fact:** for exact `β = dU`,
  `∫_γ dU = U(γ_{\text{end}}) − U(γ_{\text{start}})` depends only on the endpoints.

Everything below is a corollary of that last line.

---

## 2. Face F1 — Randers projective invariance

A **Randers metric** is `F(x,v) = α(x,v) + β_x(v)` with `‖β‖_g < 1` (so `F > 0`,
convex). Its length functional is `L_F[γ] = ∫ F(γ,γ')\,dt = L_α[γ] + ∫_γ β`.

**Theorem (projective invariance).** If `β` is closed, the `F`-geodesics coincide *as
point sets* with the geodesics of the underlying `α`. More generally, for a **fixed**
base `α` the geodesics of `α + β` depend on `β` **only through its non-exact part**:
for any `β` and any potential `U`,
```
L_{α+β+dU}[γ] = L_{α+β}[γ] + (U(B) − U(A))     for every path γ: A → B,
```
so the two functionals share minimisers — the added `dU` is a constant offset over the
feasible set and cannot move the `argmin`.

**Converse.** `β` closed is also *necessary*: if `dβ ≠ 0`, the circulation
`∮ β = ∫∫ dβ` around a loop is nonzero, so `∫_γ β` is genuinely path-dependent and the
geodesics bend. Hence:

> geodesics of `α + β` = geodesics of `α`  (as point sets)  ⟺  `dβ = 0`.

**Inverse problem.** Recovering a Randers metric from its geodesics-as-point-sets
determines `β` only up to a closed form, i.e. determines only `dβ` (the "curl", the
antisymmetrised covariant derivative `s_{ij} = ½(β_{i|j} − β_{j|i})`). This is a
restricted case of the **inverse problem of the calculus of variations** — the
*projective metrizability* problem, a.k.a. the Finslerian **Hilbert fourth problem** —
whose freedom is characterised by the holonomy of the geodesic spray (Muzsnay and
collaborators). The relevant rigidity result is:

> **Parametrization-rigidity (Bucataru–Muzsnay).** As *unparametrized* point sets the
> geodesics leave the metric with large freedom; as *arc-length-parametrized* curves the
> metric is essentially **rigid** (determined).

So the gauge is exactly the loss of parametrization: cost/timing data restores identifiability.

---

## 3. Face F2 — potential-based reward shaping (and why F1 = F2)

Consider a deterministic optimal-control / dynamic-programming problem with additive
stage cost `c(x, u, x')` and value `V`. **Potential shaping** replaces
`c → c' = c + γ Φ(x') − Φ(x)` for a scalar potential `Φ` and discount `γ`.

**Theorem (Ng–Harada–Russell).** The optimal policy is invariant under potential
shaping, and — for the invariance to hold for *all* transition dynamics — potential
shaping is the **only** additive transform with this property. Proof, `γ = 1` case: the
shaping term telescopes along any trajectory to `Φ(x_T) − Φ(x_0)`, a function of the
endpoints alone; equivalently the value shifts uniformly in the controlled variable,
`V'(x) = V(x) − Φ(x)` and `Q'(x,u) = Q(x,u) − Φ(x)`, so `argmin_u` is unchanged.

**Inverse problem (identifiability).** Recovering the cost from optimal behaviour
(inverse optimal control / IRL) determines it only up to potential shaping and positive
scaling; this is the modern *partial-identifiability* characterisation of reward
learning (Skalse et al.).

### 3.1 F1 and F2 are the same theorem

The two proofs are the *same computation* — a term that telescopes to the endpoints
cannot change the optimiser. The correspondence is exact:

| Finsler / geodesic (F1) | Optimal control (F2) |
|---|---|
| potential `U(x)` | shaping potential `Φ(x)` |
| add exact 1-form `β = dU` | add potential shaping `γΦ(x') − Φ(x)` |
| `∫_γ dU = U(B) − U(A)` (boundary term) | `Σ_t (Φ(x_{t+1}) − Φ(x_t)) = Φ(x_T) − Φ(x_0)` (telescope) |
| geodesic set invariant | optimal policy invariant |
| **uniqueness:** only closed `β` preserves geodesics | **uniqueness:** only potential shaping preserves policy |
| recover metric ↦ `β mod` closed forms | recover cost ↦ reward `mod` shaping |
| identifiable object `= dβ` (rotational part) | identifiable object `=` cost mod shaping |
| scaling gauge `F → cF`, `c>0` | scaling gauge `c → κ c`, `κ>0` |

Two independent remarks that keep this *tight* rather than loose:

- **Scope (fixed base).** The *full* geodesic gauge is larger than "closed `β`":
  projectively-equivalent base metrics `α` add more freedom (the general Hilbert-fourth
  content). Holding `α` fixed and learning only the drift collapses the gauge to
  *exactly* closed-form addition, matching the reward-shaping gauge one-to-one.
- **Discounting.** For `γ < 1` the shaping term is a *discounted coboundary* rather than
  a pure exact differential; the clean exact-1-form correspondence is the undiscounted
  `γ = 1` case, which is the natural regime of a length/geodesic functional.

---

## 4. Face F3 — the Hodge decomposition and the observation-channel duality

Now drop the "optimal-path" framing and consider recovering a **drift field** `b` of a
dynamical system directly. Hodge-split `b = −∇U + b_{\perp}` into gradient and
divergence-free (rotational) parts. The key phenomenon:

> **Which Hodge component is the gauge depends on what is observed.**

- **Channel A — argmin behaviour / optimal-path *shape*** (the F1/F2 setting). The
  **gradient** part `−∇U` is inert (§2–§3); the **rotational** part `b_{\perp}` is the
  identifiable signal. Intuition: along a route where `∇U` is orthogonal to motion, only
  the rotational part can break directional symmetry.
- **Channel B — temporal marginal *snapshots*** (dynamic optimal transport / the
  Schrödinger-bridge setting). By the **Benamou–Brenier** action, replacing a velocity
  field by its curl-free component never increases the transport cost, and (Brenier) the
  optimal map is the gradient of a convex potential. Hence marginals identify the
  **gradient** part and the **rotational flux is invisible** — the *opposite* gauge.

So Channel A and Channel B are **dual**: each sees exactly the Hodge component the other
cannot. This is the honest reason F3 is not literally F1/F2 but the *same principle*:
F1/F2 are Channel A; the transport setting is Channel B; they share the decomposition and
the gauge structure but sit on opposite sides of it.

### 4.1 The constructive payoff — break the gauge with the complementary channel

Because the two channels are complementary, the recipe writes itself:

| Setting | What the primary channel misses | Complementary channel that recovers it |
|---|---|---|
| geodesic / control (Channel A) | the gradient potential `U` | **cost magnitude / arc-length timing** (parametrization-rigidity, §2) |
| reward learning (Channel A) | the shaping potential `Φ` | **return magnitudes**, not just preferences |
| transport / marginals (Channel B) | the rotational flux `b_{\perp}` | **velocity observations / more marginals** |

**A clean estimator for Channel A.** When `β = dU` is exact, path reversal isolates the
gauge: with `γ^{-1}` the reverse curve, `L_α` is unchanged but `∫_{γ^{-1}} β = −∫_γ β`, so
```
cost(A → B) − cost(B → A) = 2∫_γ β = 2(U(B) − U(A)).
```
Over a connected graph of endpoint pairs this is a linear system in the node potentials —
a convex least-squares, unique up to an additive constant. Thus the *directional cost
asymmetry* — precisely the quantity that is invisible in path *shape* — recovers the
potential exactly. The obstruction and its resolution are the same fact viewed twice.

---

## 5. Subtleties that keep the statement honest

1. **Closed vs exact (topology).** On a non-simply-connected `M` (`H^1 ≠ 0`) a **harmonic**
   `β` is closed but not exact. It is invisible to *continuous* deformation within a
   homotopy class, yet its loop periods `∮ β` shift the cost *between* classes, so it can
   flip which topological route is globally optimal. The strictly-invisible gauge is the
   **exact** part; the harmonic part is a boundary case, observable only through discrete
   class selection. On simply-connected domains this distinction disappears.
2. **Convexity/regularity caps.** The Randers `‖β‖_g < 1` bound is the well-posedness
   boundary; outside it the metric ceases to be a genuine (positive, convex) norm. The
   gauge argument is exact *inside* the well-posed regime.
3. **Higher-order coupling.** In representations where the drift enters nonlinearly (e.g.
   the Zermelo `a_{ij} = (λ g_{ij} + W_i W_j)/λ^2` form), an exact drift is invisible only
   to leading order; the residual path signature is second order in `‖β‖` and vanishes in
   the small-drift limit. The clean, all-orders statement holds in the additive Randers
   form `α + β`.
4. **Uniqueness needs "for all dynamics."** Both uniqueness claims (only closed `β`; only
   potential shaping) require the invariance to hold universally; for a *fixed* instance
   there can be accidental extra invariances.

---

## 6. Summary

There is one theorem wearing three costumes. Adding an exact 1-form to a cost — a
gradient to a drift, a potential difference to a reward — is a **gauge transformation of
optimal behaviour**: it changes the value of every trajectory by a boundary term and the
choice of trajectory not at all. Therefore any method that learns a cost/metric/drift
from *optimal-path shape* is blind to the exact component and identifies only the
rotational component; this is Finsler projective non-rigidity and reward-shaping
non-identifiability, verbatim. The dual channel — arc-length parametrization, cost
magnitude, or (in the transport setting) marginals — carries exactly the missing
component and restores rigidity. Recognising the exact part as a *gauge* rather than a
*failure* converts a negative result ("the drift is unrecoverable") into a positive one
("recover the rotational part from shape, the gradient part from timing").

---

## 7. References

**Finsler geometry / inverse problem of the calculus of variations**
- *On projective equivalence and pointwise projective relation of Randers metrics*,
  [arXiv:1112.6143](https://arxiv.org/pdf/1112.6143) — `β` closed ⟺ Randers geodesics
  equal the underlying Riemannian geodesics as point sets.
- Bucataru & Muzsnay, *Projective and Finsler metrizability: parameterization-rigidity of
  the geodesics*, [arXiv:1108.4628](https://arxiv.org/pdf/1108.4628) — unparametrized ⇒
  free, arc-length ⇒ rigid.
- Muzsnay et al., *Metrizability of Holonomy-Invariant Projective Deformation of Sprays*,
  Canad. Math. Bull.; *On the Finsler-metrizabilities of spray manifolds*, Period. Math.
  Hungar. — freedom characterised by the spray holonomy.
- *About the integrability of the Rapcsák equation*,
  [arXiv:1505.04884](https://arxiv.org/pdf/1505.04884) — the linear system whose rank
  measures projective metrizability freedom.
- *Deformations and Hilbert's Fourth Problem*,
  [arXiv:1209.0845](https://arxiv.org/pdf/1209.0845).

**Optimal control / reward identifiability**
- Ng, Harada & Russell, *Policy Invariance under Reward Transformations* (1999) — potential
  shaping preserves the optimal policy and is uniquely characterised.
- Skalse, Farrugia-Roberts, Russell, Abate & Gleave, *Invariance in Policy Optimisation
  and Partial Identifiability in Reward Learning*, ICML 2023,
  [arXiv:2203.07475](https://arxiv.org/pdf/2203.07475).

**Hodge decomposition / optimal transport**
- Benamou & Brenier, dynamic-transport (fluid) formulation; Brenier's theorem — the
  optimal map is the gradient of a convex potential (curl-free).
- Neklyudov et al., *Action Matching: Learning Stochastic Dynamics from Samples* — the
  Helmholtz–Hodge split `v = ∇ψ + s` and curl-free optimality of transport.
