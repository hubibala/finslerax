"""Assemble (and optionally execute) the robot-arm geodesics walkthrough notebook.

Usage:
    python -m experiments.arm.build_notebook          # write unexecuted
    python -m experiments.arm.build_notebook --run    # write + execute

Produces ``experiments/arm/arm_geodesics.ipynb``. The notebook reuses the
experiment package (no duplicated physics) and renders interactive Plotly
scenes: 2-D animation players for the arm motions, the arrival-time field on
the barrier metric, and a 3-D configuration-space scene for the 3-DoF arm.
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "arm_geodesics.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def co(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md(r"""
# Energy-Optimal Arm Motion as Finsler Geometry — and What Demonstrations Can Teach

This notebook is a walkthrough of the `experiments/arm` study built on the HAM
framework. It develops
one idea carefully: **a gravity-loaded robot arm's cost of motion is an *asymmetric*
metric on its configuration space**, and HAM's differentiable Randers machinery plans
with it, constrains it, and — the deepest part — tells us *exactly which parts of it
demonstrations can and cannot reveal*.

The plan:

1. **Geometry** — why an arm's energy cost is a layered Randers metric: a kinetic
   base, a gravity "wind", and obstacles folded in as a conformal warp.
2. **The robot** — a planar arm whose mass matrix and gravity come from autodiff of
   its own kinematics; the workspace ↔ configuration-space duality, animated.
3. **Forward planning** — obstacle avoidance with *demonstrable intent*: the plan is
   validated against the globally optimal route from an arrival-time field.
4. **The ledger and the map** — gravity makes directed costs differ by ~20% while
   provably leaving the optimal path unchanged. Not a bug: a gauge symmetry.
5. **Exact constraints** — carrying a cup level with an Augmented Lagrangian, where
   a tuned penalty plateaus.
6. **The exact-drift gauge** — the capstone: recovering the drift from demonstrated
   *paths* is provably impossible, recovering it from *round-trip costs* is a convex
   least-squares. One theorem, worn three ways (Finsler projective invariance =
   potential-based reward shaping = a Hodge decomposition).
7. **Higher dimensions** — the technical cut-through: grids and arrival fields die
   with dimension, but the estimators of §6 are dimension-blind. We recover the
   gravity potential of a 5-DoF arm from timed round trips alone — the regime a
   *learned latent metric* lives in.
8. **Validation and honest limits.**

Nothing here is a black box: every number comes from the same small package
(`experiments/arm`), and the claims are checked against independent solutions.
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Setup

We import the experiment package and HAM's shared plotting style. Plotly renders the
interactive scenes; the ▶ Play buttons animate the arm motions.
""")

co(r"""
import os
import pathlib
import sys

# Locate the repository root (the folder containing `experiments/` and `src/`).
_p = pathlib.Path.cwd()
while not (_p / "experiments").exists() and _p != _p.parent:
    _p = _p.parent
sys.path.insert(0, str(_p))

os.environ.setdefault("JAX_ENABLE_X64", "0")

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

from ham.solvers import AVBDSolver, ExponentialMap
from ham.vis.style import PALETTE, use_ham_style

from experiments.arm import (
    AnalyticDistance,
    AVBDPlanner,
    CircleScene,
    EikonalPlanner,
    PlanarArm,
    PotentialWind,
    VortexWind,
    angle_manifold,
    build_arm_metric,
    fit_field,
    polyline_deviation,
    recover_form_lsq,
    recover_potential_lsq,
    segment_asymmetry,
    segment_costs,
    shape_identifiability,
)
from experiments.arm.constraints import max_constraint_violation, upright_constraint
from experiments.arm.evaluate import form_cosine, min_clearance
from experiments.arm.fields import star_grad
from experiments.arm.learn import cost_matching_loss, el_residual_loss

pio.renderers.default = "plotly_mimetype"
use_ham_style()
print("ready")
""")

co(r'''
# Shared plotting helpers: a 2-D layout and an animation player for arm motions.
def flat_layout(fig, title, height=480, width=None, equal=True):
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=15, color=PALETTE["ink"])),
        height=height, paper_bgcolor="white", plot_bgcolor="white",
        margin=dict(l=10, r=10, t=48, b=10),
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.65)",
                    bordercolor="#d8dee6", borderwidth=1),
    )
    if width:
        fig.update_layout(width=width)
    if equal:
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    return fig


def arm_trace(robot, q, color, width=5, name=None, opacity=1.0):
    pts = np.array(robot.link_points(jnp.asarray(q)))
    return go.Scatter(x=pts[:, 0], y=pts[:, 1], mode="lines+markers",
                      line=dict(color=color, width=width),
                      marker=dict(size=5, color=color), opacity=opacity,
                      name=name, showlegend=name is not None, hoverinfo="skip")


def player(fig, frames, prefix="step "):
    """Attach Play/Pause + a scrub slider."""
    play = dict(label="▶ Play", method="animate",
                args=[None, {"frame": {"duration": 140, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 0}}])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate"}])
    steps = [dict(method="animate", label=f.name,
                  args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}]) for f in frames]
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.02,
                          xanchor="left", yanchor="bottom", buttons=[play, pause])],
        sliders=[dict(active=0, x=0.18, len=0.78, y=-0.04, yanchor="top",
                      currentvalue=dict(prefix=prefix, font=dict(size=12)),
                      steps=steps)],
    )
    return fig
''')

# ---------------------------------------------------------------------------
md(r"""
## 2. The geometry: an arm's effort is a layered Randers metric

A path $q(t)$ through the arm's joint angles has a natural "effort" cost, built in
three layers, each with a mechanical justification:

$$F(q,\dot q) \;=\; \rho(q)\Big(\underbrace{\sqrt{\dot q^\top M(q)\,\dot q}}_{\text{kinetic base}}
\;+\; \underbrace{\beta(q)\cdot \dot q}_{\text{gravity drift}}\Big)$$

| Layer | Form | Why |
|---|---|---|
| Kinetic base | mass matrix $M(q)=\sum_i m_i J_i^\top J_i$ | moving heavy, extended links costs more |
| Gravity drift | wind $W=-g_s\,M^{-1}\nabla U$, so $\beta = g_s\,dU/\lambda$ | descending is cheaper than lifting — the cost is *directional* |
| Obstacles | conformal warp $\rho = 1+\alpha/(\mathrm{dist}-\delta)_+$ | collision states are infinitely expensive, smoothly |

Two structural facts drive everything below.

**Asymmetry.** $F(q,v)\neq F(q,-v)$: no symmetric (Riemannian) metric can say
"lifting costs more than lowering". This is exactly what a **Randers metric**
encodes, and it is HAM's native object.

**Exactness.** The gravity force is raised with the *mobility* $M^{-1}$ — not the
identity — which makes the drift 1-form $\beta = g_s\,dU/\lambda$ an **exact** form
(the differential of the potential). That single detail decides §4 and §6: an exact
drift is a *gauge transformation* of optimal behaviour.
""")

co(r"""
arm = PlanarArm(2, lengths=[1.0, 1.0])
ang = angle_manifold(arm)
GS = 0.15
metric = build_arm_metric(arm, manifold=ang, gravity_strength=GS)

q = jnp.array([0.4, -0.6])
up = arm.gravity(q) / jnp.linalg.norm(arm.gravity(q))   # uphill direction
H, W, lam = metric.zermelo_data(q)
print(f"mass matrix M(q) =\n{np.array(arm.inertia(q)).round(3)}")
print(f"cost of moving uphill  : {float(metric.metric_fn(q, up)):.3f}")
print(f"cost of moving downhill: {float(metric.metric_fn(q, -up)):.3f}")
print(f"mild-wind margin lam = 1 - |W|^2_H = {float(lam):.3f}  (well-posed: 0 < lam <= 1)")
# The exactness anchor: -(H W) equals gs * gradU to numerical precision.
print(f"drift form check |-HW - gs∇U| = "
      f"{float(jnp.max(jnp.abs(-(H @ W) - GS * arm.gravity(q)))):.2e}")
""")

md(r"""
### The two views of one machine

Every motion lives in two pictures at once: the **workspace** (where the physical
links swing) and the **configuration space** (where the motion is a single curve
$q(t)$, and where all the geometry happens). The animation sweeps the elbow joint
and shows both views moving together; the background is the gravitational potential
$U(q)$, whose gradient is the drift.
""")

co(r"""
# Left: the physical arm. Right: the same motion as a point moving over U(q).
g = np.linspace(-2.2, 2.2, 120)
G1, G2 = np.meshgrid(g, g, indexing="ij")
U = np.array(jax.vmap(arm.potential)(jnp.stack([G1.ravel(), G2.ravel()], -1))).reshape(G1.shape)

sweep = jnp.stack([0.4 * jnp.sin(jnp.linspace(0, 2 * np.pi, 30)) + 0.3,
                   jnp.linspace(-2.0, 2.0, 30)], axis=1)

fig = make_subplots(rows=1, cols=2, column_widths=[0.5, 0.5],
                    subplot_titles=("workspace", "configuration space over U(q)"))
fig.add_trace(arm_trace(arm, sweep[0], PALETTE["primary"], name="arm"), row=1, col=1)
fig.add_trace(go.Heatmap(x=g, y=g, z=U.T, colorscale="RdBu_r", showscale=False,
                         hoverinfo="skip"), row=1, col=2)
fig.add_trace(go.Scatter(x=[float(sweep[0, 0])], y=[float(sweep[0, 1])], mode="markers",
                         marker=dict(size=12, color=PALETTE["ink"]), name="q"),
              row=1, col=2)
frames = []
for k in range(sweep.shape[0]):
    frames.append(go.Frame(name=str(k), data=[
        arm_trace(arm, sweep[k], PALETTE["primary"]),
        go.Heatmap(x=g, y=g, z=U.T, colorscale="RdBu_r", showscale=False),
        go.Scatter(x=[float(sweep[k, 0])], y=[float(sweep[k, 1])], mode="markers",
                   marker=dict(size=12, color=PALETTE["ink"]))]))
fig.frames = frames
fig.update_xaxes(range=[-2.3, 2.3], row=1, col=1)
fig.update_yaxes(range=[-2.3, 2.3], scaleanchor="x", scaleratio=1, row=1, col=1)
flat_layout(fig, "One machine, two views (press Play)", height=470, equal=False)
player(fig, frames)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. Forward planning: avoidance you can verify

A demo of obstacle avoidance is only convincing if the obstacle actually *constrains
the optimum*. So the task is engineered for it: the arm must swing its end-effector
from pointing down to pointing up, and the direct motion drives the nearly-extended
arm straight through a disk — 16% of the straight path is in collision, deepest at
mid-swing.

The barrier folds the obstacle into the metric ($H\to\rho^2 H$, $W\to W/\rho$), and a
continuation schedule anneals its strength (a cold solve at full stiffness diverges —
that failure mode is itself gated, as validation rung L4). The final barrier is
*localized*: $\rho\approx 1$ in free space, so it is a wall, not molasses.

The claim "the detour is optimal" is validated *globally*: we solve the eikonal
arrival-time field on the **same barrier metric** — an Eulerian PDE solve with no
initialization and no basins — and back-trace its characteristic. The local
(Lagrangian) AVBD geodesic must ride the same route at the same cost.
""")

co(r"""
CIRCLE = (1.3, 0.9, 0.35)
qa, qb = jnp.array([-1.2, -0.6]), jnp.array([1.8, -0.6])
dist = AnalyticDistance(arm, CircleScene([list(CIRCLE)]))
base = build_arm_metric(arm, manifold=ang)


def make_barrier(alpha):
    return build_arm_metric(arm, dist, manifold=ang, alpha=alpha, delta=0.05)


straight = jnp.linspace(qa, qb, 25)
geo = AVBDPlanner(step_size=0.05, iterations=600).plan(
    make_barrier, qa, qb, n_steps=24, alphas=(0.01, 0.03, 0.06, 0.15)).xs

d_str = jax.vmap(dist)(straight)
d_geo = jax.vmap(dist)(geo)
print(f"straight sweep : min clearance {float(d_str.min()):+.3f}  "
      f"({100 * float(jnp.mean((d_str < 0).astype(float))):.0f}% of the path in collision)")
print(f"avoided geodesic: min clearance {float(d_geo.min()):+.3f} "
      f"at node {int(jnp.argmin(d_geo))}/24 — tightest point mid-swing")
""")

co(r"""
# The global check: arrival field + characteristic on the barrier metric.
m_obs = make_barrier(0.15)
EXT = (-2.8, 2.8, -2.8, 2.8)
eik = EikonalPlanner(max_iters=800)
T_obs = eik.arrival_field(m_obs, qa, EXT, (161, 161))
t_goal = float(eik.sample(T_obs, EXT, qb))
char = eik.trace_path(T_obs, EXT, qb, tol_frac=0.01)
char = jnp.concatenate([char, qa[None]], axis=0)
len_avbd = float(m_obs.arc_length(geo))
dev = float(jnp.maximum(polyline_deviation(geo, char), polyline_deviation(char, geo)))
print(f"AVBD cost {len_avbd:.3f}  vs  eikonal arrival {t_goal:.3f} "
      f"(rel {abs(len_avbd - t_goal) / t_goal:.1%}, grid-resolution agreement)")
print(f"route deviation AVBD vs characteristic: {dev:.3f} rad (same corridor)")
""")

co(r"""
# C-space: the arrival field, the obstacle shadow, and all three routes.
axg = np.linspace(EXT[0], EXT[1], T_obs.shape[0])
Tc = np.clip(np.array(T_obs), 0, 2.2 * t_goal)
gg = np.linspace(EXT[0], EXT[1], 200)
GG1, GG2 = np.meshgrid(gg, gg, indexing="ij")
clr = np.array(jax.vmap(dist)(jnp.stack([GG1.ravel(), GG2.ravel()], -1))).reshape(GG1.shape)

fig = go.Figure()
fig.add_trace(go.Heatmap(x=axg, y=axg, z=Tc.T, colorscale="Turbo",
                         colorbar=dict(title="arrival time", thickness=14, len=0.7)))
fig.add_trace(go.Contour(x=gg, y=gg, z=clr.T, showscale=False,
                         contours=dict(start=0, end=0, coloring="none"),
                         line=dict(color="white", width=3), hoverinfo="skip",
                         name="obstacle boundary"))
for path, color, dash, name in [
        (straight, PALETTE["rose"], "dash", "straight (collides)"),
        (char, "white", None, "eikonal characteristic (global)"),
        (geo, PALETTE["ink"], None, "AVBD geodesic")]:
    p = np.array(path)
    fig.add_trace(go.Scatter(x=p[:, 0], y=p[:, 1], mode="lines",
                             line=dict(color=color, width=4, dash=dash), name=name))
lo = min(float(np.array(geo)[:, 1].min()), float(np.array(char)[:, 1].min())) - 0.3
fig.update_yaxes(range=[lo, 2.4])
flat_layout(fig, "The local geodesic rides the globally optimal route", height=560,
            equal=False)
fig.show()
""")

md(r"""
And the physical picture. Press **Play**: the blue arm executes the geodesic — watch
it tuck the elbow to slip past the disk and re-extend — while the faint red arm
executes the straight motion and drives through the obstacle.
""")

co(r"""
def two_arm_frames(robot, path_a, path_b, color_a, color_b):
    frames = []
    n = path_a.shape[0]
    ee = np.array(jax.vmap(robot.fk)(path_a))
    for k in range(n):
        frames.append(go.Frame(name=str(k), data=[
            arm_trace(robot, path_b[k], color_b, width=3, opacity=0.35),
            arm_trace(robot, path_a[k], color_a, width=6),
            go.Scatter(x=ee[: k + 1, 0], y=ee[: k + 1, 1], mode="lines",
                       line=dict(color=color_a, width=2), opacity=0.7,
                       hoverinfo="skip")]))
    return frames


frames = two_arm_frames(arm, geo, straight, PALETTE["primary"], PALETTE["rose"])
fig = go.Figure(data=list(frames[0].data), frames=frames)
fig.add_shape(type="circle", x0=CIRCLE[0] - CIRCLE[2], y0=CIRCLE[1] - CIRCLE[2],
              x1=CIRCLE[0] + CIRCLE[2], y1=CIRCLE[1] + CIRCLE[2],
              fillcolor=PALETTE["rose"], opacity=0.75, line_width=0)
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                         marker=dict(size=9, color=PALETTE["ink"], symbol="square"),
                         name="base"))
fig.update_xaxes(range=[-2.2, 2.2]); fig.update_yaxes(range=[-2.2, 2.2])
flat_layout(fig, "Executing the plan (blue: geodesic — red ghost: straight)", height=560)
player(fig, frames)
fig.show()
""")

co(r"""
# Clearance along the motion: the barrier is *active* — the optimum hugs the margin
# exactly where the straight path is deepest in collision.
s = np.linspace(0, 1, 25)
fig = go.Figure()
fig.add_trace(go.Scatter(x=s, y=np.array(d_str), mode="lines", name="straight",
                         line=dict(color=PALETTE["rose"], width=3, dash="dash")))
fig.add_trace(go.Scatter(x=s, y=np.array(d_geo), mode="lines", name="avoided geodesic",
                         line=dict(color=PALETTE["primary"], width=3)))
fig.add_hline(y=0.0, line_color=PALETTE["ink"], line_width=1)
fig.add_hline(y=0.05, line_dash="dot", line_color=PALETTE["muted"],
              annotation_text="barrier margin δ")
fig.add_hrect(y0=-0.45, y1=0.0, fillcolor=PALETTE["rose"], opacity=0.10, line_width=0)
flat_layout(fig, "Whole-arm clearance along the motion", height=380, equal=False)
fig.update_xaxes(title="path parameter"); fig.update_yaxes(title="clearance")
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. The ledger and the map: gravity is big, and invisible

Now switch the obstacle off and gravity on, and plan the same swing both ways. Two
measurements, one delightful tension:

- **The ledger.** The directed geodesic costs differ by ~20%: A→B fights the lift,
  B→A rides the drift. Whenever the *direction* of a motion is free — task
  sequencing, pick order, return legs — the Randers model says which way is cheap.
- **The map.** The gravity-aware and gravity-blind *plans coincide as point sets*.
  Planning "with gravity in mind" does not bend the route at all (to first order).

The tension is a theorem, not a defect. The drift form $\beta = g_s\,dU/\lambda$ is
**exact**, and adding an exact form to a cost changes every path's cost by a boundary
term $g_s\big(U(B)-U(A)\big)$ — the same for all paths between the same endpoints —
so it cannot change *which* path wins. That is Finsler **projective invariance**,
and it is literally the same telescoping argument as potential-based reward shaping
in RL (Ng et al., 1999). Gravity lives in the ledger, not on the map.
""")

co(r"""
A, B = jnp.array([-1.1, 1.4]), jnp.array([1.1, 1.4])
m_sym = build_arm_metric(arm, manifold=ang, gravity_strength=0.0)
m_asym = build_arm_metric(arm, manifold=ang, gravity_strength=0.12)
planner = AVBDPlanner(step_size=0.04, iterations=500)

plan_blind = planner.plan(lambda a: m_sym, A, B, n_steps=32).xs
plan_aware = planner.plan(lambda a: m_asym, A, B, n_steps=32).xs

cost_ab = float(m_asym.arc_length(plan_aware))
cost_ba = float(m_asym.arc_length(plan_aware[::-1]))
shape_dev = float(jnp.maximum(polyline_deviation(plan_aware, plan_blind),
                              polyline_deviation(plan_blind, plan_aware)))
print(f"ledger: cost A->B = {cost_ab:.3f}   B->A = {cost_ba:.3f}   "
      f"direction choice saves {100 * (1 - cost_ba / cost_ab):.1f}%")
print(f"map:    aware-vs-blind plan deviation = {shape_dev:.4f} rad (they coincide)")
# The boundary-term identity, checked on the actual path:
lhs = cost_ab - cost_ba
rhs = 2 * 0.12 * float(arm.potential(B) - arm.potential(A))
print(f"asymmetry {lhs:.3f}  vs  2·gs·(U(B)-U(A)) = {rhs:.3f}   (the telescoping term)")
""")

md(r"""
The animation runs the *same route* in both directions and grows the two cost
ledgers as the arm moves. Same point set — different bill.
""")

co(r"""
segs = segment_costs(m_asym, plan_aware)
segs_rev = segment_costs(m_asym, plan_aware[::-1])
cum_ab = np.concatenate([[0], np.cumsum(np.array(segs))])
cum_ba = np.concatenate([[0], np.cumsum(np.array(segs_rev))])
pa = np.array(plan_aware)
n = pa.shape[0]

fig = make_subplots(rows=1, cols=2, column_widths=[0.45, 0.55],
                    subplot_titles=("the motion", "the two ledgers"))
fig.add_trace(arm_trace(arm, pa[0], PALETTE["primary"], name="A→B (lifting)"), row=1, col=1)
fig.add_trace(arm_trace(arm, pa[-1], PALETTE["accent"], name="B→A (descending)"), row=1, col=1)
fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", name=f"A→B  (total {cum_ab[-1]:.2f})",
                         line=dict(color=PALETTE["primary"], width=3)), row=1, col=2)
fig.add_trace(go.Scatter(x=[0], y=[0], mode="lines", name=f"B→A  (total {cum_ba[-1]:.2f})",
                         line=dict(color=PALETTE["accent"], width=3)), row=1, col=2)
frames = []
for k in range(n):
    x = np.linspace(0, 1, n)[: k + 1]
    frames.append(go.Frame(name=str(k), data=[
        arm_trace(arm, pa[k], PALETTE["primary"]),
        arm_trace(arm, pa[n - 1 - k], PALETTE["accent"]),
        go.Scatter(x=x, y=cum_ab[: k + 1], mode="lines",
                   line=dict(color=PALETTE["primary"], width=3)),
        go.Scatter(x=x, y=cum_ba[: k + 1], mode="lines",
                   line=dict(color=PALETTE["accent"], width=3))]))
fig.frames = frames
fig.update_xaxes(range=[-2.3, 2.3], row=1, col=1)
fig.update_yaxes(range=[-2.3, 2.3], scaleanchor="x", scaleratio=1, row=1, col=1)
fig.update_xaxes(title="path parameter", range=[0, 1], row=1, col=2)
fig.update_yaxes(title="accumulated cost", range=[0, float(cum_ab[-1]) * 1.08], row=1, col=2)
flat_layout(fig, "Same route, different bill (press Play)", height=470, equal=False)
player(fig, frames)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Exact constraints: carrying a cup

A redundant 3-link arm can reach the goal while letting its end-effector tilt — the
minimum-energy motion exploits exactly that slack. To carry a cup, the orientation
$\sum_i \theta_i$ must be held at zero *along the whole motion*: an equality
constraint $c(q)=0$ on every interior waypoint.

AVBD enforces this natively with its **Augmented Lagrangian** (dual variables, not
just penalties). The comparison below is against a *well-tuned* pure penalty with a
continuation ramp — the strongest fair baseline — which still plateaus at the classic
$\sim 1/\mu$ residual. The ALM drives the violation three orders of magnitude lower
at essentially the unconstrained energy.
""")

co(r"""
arm3 = PlanarArm(3, lengths=[1.0, 0.8, 0.6])
ang3 = angle_manifold(arm3)
m3 = build_arm_metric(arm3, manifold=ang3, gravity_strength=0.02)
qs3, qg3 = jnp.array([1.2, -0.6, -0.6]), jnp.array([-0.9, 1.1, -0.2])
c = upright_constraint(arm3, ang3, target_angle=0.0)

uncon = AVBDSolver(step_size=0.05, iterations=500).solve(
    m3, qs3, qg3, n_steps=24, train_mode=False)
alm = AVBDSolver(step_size=0.02, beta=1.2, iterations=800, tol=1e-4).solve(
    m3, qs3, qg3, n_steps=24, constraints=[c], train_mode=False)
v_un = float(max_constraint_violation(uncon.xs, [c]))
v_alm = float(max_constraint_violation(alm.xs, [c]))
print(f"max tilt: unconstrained = {v_un:.4f} rad   ALM = {v_alm:.2e} rad")
print(f"energy:   unconstrained = {float(uncon.energy):.3f}   ALM = {float(alm.energy):.3f}")
""")

co(r"""
def tilted_cup(robot, q, color):
    ee = np.array(robot.fk(jnp.asarray(q)))
    phi = float(robot.ee_orientation(jnp.asarray(q)))
    dx, dy = 0.16 * np.cos(phi), 0.16 * np.sin(phi)
    return go.Scatter(x=[ee[0] - dx, ee[0] + dx], y=[ee[1] - dy, ee[1] + dy],
                      mode="lines", line=dict(color=color, width=6),
                      hoverinfo="skip", showlegend=False)


frames = []
for k in range(uncon.xs.shape[0]):
    frames.append(go.Frame(name=str(k), data=[
        arm_trace(arm3, uncon.xs[k], PALETTE["rose"], width=3, opacity=0.5),
        tilted_cup(arm3, uncon.xs[k], PALETTE["rose"]),
        arm_trace(arm3, alm.xs[k], PALETTE["primary"], width=5),
        tilted_cup(arm3, alm.xs[k], PALETTE["ink"])]))
fig = go.Figure(data=list(frames[0].data), frames=frames)
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                         marker=dict(size=9, color=PALETTE["ink"], symbol="square"),
                         name="base"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color=PALETTE["rose"]), name="unconstrained (cup tilts)"))
fig.add_trace(go.Scatter(x=[None], y=[None], mode="lines",
                         line=dict(color=PALETTE["primary"]), name="ALM (cup level)"))
fig.update_xaxes(range=[-2.6, 2.6]); fig.update_yaxes(range=[-1.2, 2.6])
flat_layout(fig, "Carrying the cup: ALM holds it level to 1e-4 (press Play)", height=520)
player(fig, frames)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. The exact-drift gauge: what demonstrations can teach

Here is the question that turns this from a planning demo into a piece of science:
**given demonstrated motions, can you recover the drift field** (the "gravity" of the
world that produced them)?

For the shape of the demonstrated paths, the answer is a theorem, and it is *no*:

> Adding an exact 1-form $\beta = dU$ to a cost changes the value of every path by
> the boundary term $U(B)-U(A)$ and the choice of path not at all. Hence the exact
> part of a drift is **invisible in path shape** — no amount of data, no better
> optimizer, no bigger network can recover it from where the demos go.

One theorem, three costumes ([`spec/exact_drift_gauge_equivalence.md`](../../spec/exact_drift_gauge_equivalence.md)):

| Face | The invisible transform | The invariance |
|---|---|---|
| Finsler geometry | add closed $\beta$ to a Randers metric | geodesics unchanged as point sets |
| Optimal control / IRL | potential shaping $c \to c + \Phi(x')-\Phi(x)$ | optimal policy unchanged (Ng et al. 1999) |
| Hodge / transport | shift the gradient part of a drift | argmin behaviour sees only the rotational part |

And the constructive flip side: the *complementary observation channel* — **cost and
timing** — carries exactly the missing component (Bucataru-Muzsnay
parametrization-rigidity). The cleanest such observation is the round trip:

$$\mathrm{cost}(\gamma) - \mathrm{cost}(\gamma^{-1}) \;=\; 2\!\int_\gamma\!\beta
\;=\; 2\,g_s\big(U(B)-U(A)\big),$$

which turns drift recovery into a **convex least-squares** — no seeds, no solver in
the loop, no local minima.

We measure all of this on two ground-truth regimes: **G** (pure gravity — exact
drift) and **G+R** (gravity plus a solenoidal vortex — a rotational part that *does*
bend paths, at first order).
""")

co(r"""
vortex = VortexWind(arm, center=(0.2, 0.3), strength=0.12, width=0.7)
mG = build_arm_metric(arm, manifold=ang, gravity_strength=GS)
mGR = build_arm_metric(arm, manifold=ang, gravity_strength=GS, wind_field=vortex)

gen = AVBDSolver(step_size=0.05, iterations=400)
pairs = jax.random.uniform(jax.random.PRNGKey(0), (12, 2, 2), minval=-1.2, maxval=1.2)
demosG = [gen.solve(mG, p[0], p[1], n_steps=16, train_mode=False).xs for p in pairs]
demosGR = [gen.solve(mGR, p[0], p[1], n_steps=16, train_mode=False).xs for p in pairs]
regionG = jnp.concatenate(demosG, axis=0)
print(f"{len(demosG)} demonstrations per regime")
""")

md(r"""
### 6a. The theorem, made visible: a scaling law

Trace the geodesic spray from the *same start and initial direction* under the base
metric and under each drift, and measure how far the point sets bend apart. The
exact (gravity) drift bends paths only at **second order** in its strength — a small
representation leak of the Zermelo form, zero in the additive Randers form — while
the rotational vortex bends them at **first order**, an order of magnitude harder at
matched strength.
""")

co(r"""
A0, v0 = jnp.array([-0.9, 0.2]), jnp.array([1.6, 0.8])
em, em_ref = ExponentialMap(max_steps=64), ExponentialMap(max_steps=96)
ref, _ = em_ref.trace(base, A0, v0)
strengths = np.array([0.05, 0.1, 0.2, 0.3])
dev_g, dev_r = [], []
for s_ in strengths:
    pg, _ = em.trace(build_arm_metric(arm, manifold=ang, gravity_strength=float(s_)), A0, v0)
    dev_g.append(float(polyline_deviation(pg, ref)))
    vw = VortexWind(arm, center=(0.2, 0.3), strength=float(s_) / 4.0, width=0.7)
    pv, _ = em.trace(build_arm_metric(arm, manifold=ang, wind_field=vw), A0, v0)
    dev_r.append(float(polyline_deviation(pv, ref)))

fig = go.Figure()
fig.add_trace(go.Scatter(x=strengths, y=dev_g, mode="lines+markers",
                         name="gravity (exact drift)", line=dict(color=PALETTE["primary"])))
fig.add_trace(go.Scatter(x=strengths / 4, y=dev_r, mode="lines+markers",
                         name="vortex (rotational)", line=dict(color=PALETTE["rose"])))
for expo, x0, y0, color in [(2, strengths, dev_g, PALETTE["primary"]),
                            (1, strengths / 4, dev_r, PALETTE["rose"])]:
    xs_ = np.array([x0[0], x0[-1]])
    fig.add_trace(go.Scatter(x=xs_, y=y0[0] * (xs_ / x0[0]) ** expo, mode="lines",
                             line=dict(color=color, dash="dot"), name=f"slope {expo}"))
fig.update_xaxes(type="log", title="drift strength")
fig.update_yaxes(type="log", title="point-set deviation (rad)")
flat_layout(fig, "Path bending: 1st order for rotational, 2nd order for exact drift",
            height=420, equal=False)
fig.show()
""")

md(r"""
### 6b. Predicting identifiability *before* fitting anything

A discrete projective-metrizability test: do the demos already trace base-metric
geodesics as point sets? On pure gravity they do (score ≈ 0 — path shape carries
nothing); with the vortex they measurably do not. This is the *a-priori* version of
Muzsnay's spray-holonomy criterion — you know which recovery will fail before
running it.
""")

co(r"""
sG = shape_identifiability(base, demosG[:8], gen)
sGR = shape_identifiability(base, demosGR[:8], gen)
print(f"shape-identifiability score:  G = {sG:.4f}   G+R = {sGR:.4f}   "
      f"(ratio {sGR / sG:.1f}x)")
""")

md(r"""
### 6c. The channel experiment

Same scalar-potential hypothesis class, same demonstrations, two losses:

- **L-B (path shape)** — the demos should satisfy the geodesic equation of the
  fitted metric, with the residual projected onto the path normal. (The projection
  matters: solver demos are sampled at constant $F$-speed, so raw vertex spacing is
  *timing in disguise* — leaving it in quietly mixes the channels.)
- **L-T (timing)** — match the observed per-segment traversal costs.

The shape loss has only the second-order leak to feed on: it overfits a few demos
and *degrades* as demos are added. The timing loss identifies the potential
essentially perfectly. We report R² of the recovered potential against the true
$g_s U$ (additive constant removed).
""")

co(r"""
def make_wind_metric(field):
    return build_arm_metric(arm, manifold=ang, wind_field=field)


def potential_r2(U_fn, region):
    u_hat = jax.vmap(U_fn)(region)
    u_true = GS * jax.vmap(arm.potential)(region)
    u_hat = u_hat - jnp.mean(u_hat)
    u_true = u_true - jnp.mean(u_true)
    return 1.0 - float(jnp.sum((u_hat - u_true) ** 2) / jnp.sum(u_true**2))


costsG = [segment_costs(mG, d) for d in demosG]
f0 = PotentialWind(arm, width=32, depth=2, key=jax.random.PRNGKey(1))
ft, _ = fit_field(f0, demosG, make_wind_metric, loss_fn=cost_matching_loss,
                  steps=600, lr=5e-3, demo_costs=costsG)
r2_shape = []
for seed in (1, 2, 3):
    fb, _ = fit_field(PotentialWind(arm, width=32, depth=2, key=jax.random.PRNGKey(seed)),
                      demosG, make_wind_metric, loss_fn=el_residual_loss,
                      steps=600, lr=5e-3, normal_only=True)
    r2_shape.append(potential_r2(fb.potential, regionG))
print(f"L-T (timing)  potential R² = {potential_r2(ft.potential, regionG):+.3f}")
print(f"L-B (shape)   potential R² = {min(r2_shape):+.2f} .. {max(r2_shape):+.2f}  (3 seeds)")
""")

md(r"""
### 6d. The convex estimator: the potential from round-trip costs

Every demo segment, run forward and backward, yields one linear equation
$\phi(x_{k+1})^\top w - \phi(x_k)^\top w \approx \Delta U_k$ in RBF coefficients $w$.
A ridge least-squares over the demo graph recovers the potential globally — the
quantity that was *provably* invisible in path shape falls out of a linear solve.
""")

co(r"""
starts = jnp.concatenate([d[:-1] for d in demosG], axis=0)
ends = jnp.concatenate([d[1:] for d in demosG], axis=0)
deltas = jnp.concatenate([segment_asymmetry(mG, d) for d in demosG], axis=0)
gc = jnp.linspace(-1.4, 1.4, 6)
C1, C2 = jnp.meshgrid(gc, gc, indexing="ij")
centers = jnp.stack([C1.ravel(), C2.ravel()], -1)
U_hat, _ = recover_potential_lsq(starts, ends, deltas, centers, width=0.7)
print(f"cost-asymmetry LSQ: potential R² = {potential_r2(U_hat, regionG):.4f} "
      "(convex — no seeds, no inner solver)")

gp = np.linspace(-1.3, 1.3, 110)
GP1, GP2 = np.meshgrid(gp, gp, indexing="ij")
pts = jnp.stack([GP1.ravel(), GP2.ravel()], -1)
Ut = np.array(GS * jax.vmap(arm.potential)(pts)).reshape(GP1.shape)
Uh = np.array(jax.vmap(U_hat)(pts)).reshape(GP1.shape)
fig = go.Figure()
fig.add_trace(go.Heatmap(x=gp, y=gp, z=(Ut - Ut.mean()).T, colorscale="RdBu_r",
                         colorbar=dict(title="true gₛU", thickness=14, len=0.7)))
fig.add_trace(go.Contour(x=gp, y=gp, z=(Uh - Uh.mean()).T, showscale=False,
                         contours=dict(coloring="none"),
                         line=dict(color=PALETTE["ink"], width=1.5), name="recovered"))
for d in demosG:
    dp = np.array(d)
    fig.add_trace(go.Scatter(x=dp[:, 0], y=dp[:, 1], mode="lines", opacity=0.45,
                             line=dict(color=PALETTE["muted"], width=1),
                             showlegend=False, hoverinfo="skip"))
flat_layout(fig, "Fill: true potential — lines: recovered from round-trip asymmetries",
            height=520)
fig.show()
""")

md(r"""
### 6e. It generalizes: the *whole* drift form on G+R

The round-trip asymmetry observes $\int_\gamma \beta$ — the whole 1-form, not just
its exact part. On the gravity+vortex regime, the same linear solve (with gradient
*and* rotational RBF atoms) recovers the full drift at cosine ≈ 0.99 where the demos
went. Its Hodge *split* into the two parts is softer — separating $dU$ from
$\star d\psi$ given line integrals along finitely many curves is only determined up
to near-harmonic leakage — and we report that honestly rather than hiding it.
""")

co(r"""
startsR = jnp.concatenate([d[:-1] for d in demosGR], axis=0)
endsR = jnp.concatenate([d[1:] for d in demosGR], axis=0)
deltasR = jnp.concatenate([segment_asymmetry(mGR, d) for d in demosGR], axis=0)
regionGR = jnp.concatenate(demosGR, axis=0)
U_hatR, psi_hatR, b_hatR = recover_form_lsq(startsR, endsR, deltasR, centers, width=0.7)

b_true = (GS * jax.vmap(arm.gravity)(regionGR)
          + jax.vmap(lambda q: star_grad(vortex.stream, q))(regionGR))
cos_full = form_cosine(jax.vmap(b_hatR)(regionGR), b_true)
print(f"full drift form recovered at cosine {cos_full:.3f}  "
      f"(soft split: U R²={potential_r2(U_hatR, regionGR):.2f})")

qs_ = regionGR[::5]
bt = np.array(GS * jax.vmap(arm.gravity)(qs_)
              + jax.vmap(lambda q: star_grad(vortex.stream, q))(qs_))
bl = np.array(jax.vmap(b_hatR)(qs_))
P = np.array(qs_)


def quiver(P, V, color, name, sc=0.55):
    xs_, ys_ = [], []
    for p, v in zip(P, V):
        xs_ += [p[0], p[0] + sc * v[0], None]
        ys_ += [p[1], p[1] + sc * v[1], None]
    return go.Scatter(x=xs_, y=ys_, mode="lines", line=dict(color=color, width=2),
                      name=name, hoverinfo="skip")


fig = go.Figure()
fig.add_trace(quiver(P, bt, PALETTE["ink"], "true drift form β"))
fig.add_trace(quiver(P, bl, PALETTE["rose"], "recovered from asymmetry"))
flat_layout(fig, f"G+R: the whole drift form from round-trip costs (cos {cos_full:.3f})",
            height=520)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Higher dimensions: the latent-space regime

Everything visual so far leaned on the 2-DoF luxury: arrival-time fields on a grid,
C-space heatmaps, obstacle shadows. **All of that dies with dimension** — a grid at
5 DoF would need $10^{10}$ cells. This is not a corner case; it is the *normal*
regime for HAM, where the metric lives on a learned latent space with no grid, no
ground-truth field, and nothing to look at.

The point of this section: the machinery that carries the *science* — the AVBD
geodesic solver, the timing losses, and the convex asymmetry estimators — is
**dimension-blind**. What thins out is only the grid-based *instrumentation*
(eikonal cross-checks, and the a-priori diagnostic, whose solver floor grows with
geodesic basin multiplicity).

First, 3 DoF — the last dimension you can see. The configuration-space obstacle is
now a genuinely warped 3-D body, and the geodesic threads around it.

A discretization lesson earned the hard way: at 24 segments this task **tunnels** —
the barrier repels vertices out of the expensive zone until one long chord spans
the obstacle body, and because AVBD's discrete energy samples each segment at its
*midpoint*, that chord feels no barrier at all. Vertex-only clearance even reports
it as safe. Two habits fix it, permanently: clearance is always evaluated on a
*densified* path (`min_clearance` interpolates within segments), and the segment
length must stay well below the obstacle body's thickness — 48 segments here.
""")

co(r"""
arm3d = PlanarArm(3, lengths=[0.9, 0.8, 0.7])
ang3d = angle_manifold(arm3d)
dist3 = AnalyticDistance(arm3d, CircleScene([[1.6, 0.9, 0.3]]))
qa3, qb3 = jnp.array([-0.9, -0.3, -0.3]), jnp.array([1.3, -0.4, 0.2])
straight3 = jnp.linspace(qa3, qb3, 49)
geo3 = AVBDPlanner(step_size=0.02, iterations=1200, resample_between=True).plan(
    lambda a: build_arm_metric(arm3d, dist3, manifold=ang3d, alpha=a, delta=0.04),
    qa3, qb3, n_steps=48, alphas=(0.02, 0.05, 0.12, 0.3)).xs
seg3 = jnp.linalg.norm(geo3[1:] - geo3[:-1], axis=-1)
print(f"3-DoF: straight densified clearance "
      f"{float(min_clearance(dist3, ang3d, straight3)):+.3f}  ->  planned "
      f"{float(min_clearance(dist3, ang3d, geo3)):+.3f}")
print(f"max segment {float(seg3.max()):.2f} rad — well below the obstacle body "
      "thickness (no room to tunnel)")

# The C-space obstacle as a 3-D isosurface + both routes.
ng = 36
gg3 = np.linspace(-1.6, 1.9, ng)
GA, GB, GC = np.meshgrid(gg3, gg3, gg3, indexing="ij")
D3 = np.array(jax.vmap(dist3)(jnp.stack([GA.ravel(), GB.ravel(), GC.ravel()], -1)))

fig = go.Figure()
fig.add_trace(go.Isosurface(
    x=GA.ravel(), y=GB.ravel(), z=GC.ravel(), value=D3,
    isomin=0.0, isomax=0.0, surface_count=1, opacity=0.35,
    colorscale=[[0, PALETTE["rose"]], [1, PALETTE["rose"]]], showscale=False,
    caps=dict(x_show=False, y_show=False, z_show=False), name="C-obstacle"))
for path, color, dash, name in [(straight3, PALETTE["rose"], "dash", "straight (collides)"),
                                (geo3, PALETTE["primary"], None, "AVBD geodesic (clear)")]:
    p = np.array(path)
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                               line=dict(color=color, width=7, dash=dash), name=name))
fig.update_layout(
    title=dict(text="3-DoF configuration space: the obstacle is a warped 3-D body",
               x=0.5, font=dict(size=15, color=PALETTE["ink"])),
    scene=dict(xaxis_title="θ₁", yaxis_title="θ₂", zaxis_title="θ₃",
               camera=dict(eye=dict(x=1.7, y=1.5, z=1.1))),
    height=600, margin=dict(l=0, r=0, t=44, b=0), paper_bgcolor="white",
    legend=dict(x=0.02, y=0.98))
fig.show()
""")

md(r"""
And the same motion in the workspace — press **Play** to watch the 3-link arm slip
past the disk.
""")

co(r"""
frames = two_arm_frames(arm3d, geo3, straight3, PALETTE["primary"], PALETTE["rose"])
fig = go.Figure(data=list(frames[0].data), frames=frames)
fig.add_shape(type="circle", x0=1.6 - 0.3, y0=0.9 - 0.3, x1=1.6 + 0.3, y1=0.9 + 0.3,
              fillcolor=PALETTE["rose"], opacity=0.75, line_width=0)
fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers",
                         marker=dict(size=9, color=PALETTE["ink"], symbol="square"),
                         name="base"))
fig.update_xaxes(range=[-2.6, 2.6]); fig.update_yaxes(range=[-2.6, 2.6])
flat_layout(fig, "3-DoF avoidance in the workspace", height=540)
player(fig, frames)
fig.show()
""")

md(r"""
### 5 DoF: recovery with nothing to look at

Now the cut-through. A 5-DoF arm under gravity; demonstrations are timed round trips
between random configurations. There is no grid, no picture of the potential, and no
eikonal field to cross-check against — exactly the situation of a *learned latent
metric* in a world model. Two dimension-blind choices do all the work:

- **data-adaptive features**: the RBF centers are subsampled demo vertices (a grid of
  centers would be exponentially wasteful and mostly unsupported by data);
- **the convex estimator of §6d**, unchanged — it never knew what dimension it was in.

The recovered potential is compared point-by-point against the true $g_s U(q)$ on the
demo support. (Note the solver detail that makes 5 DoF work at all: the 5-link mass
matrix is stiff, and AVBD's fixed step must drop to 0.02 — at 0.05 it diverges.)
""")

co(r"""
arm5 = PlanarArm(5, lengths=[0.6] * 5)
ang5 = angle_manifold(arm5)
m5 = build_arm_metric(arm5, manifold=ang5, gravity_strength=GS)
gen5 = AVBDSolver(step_size=0.02, iterations=800)
pairs5 = jax.random.uniform(jax.random.PRNGKey(0), (10, 2, 5), minval=-1.0, maxval=1.0)
demos5 = [gen5.solve(m5, p[0], p[1], n_steps=16, train_mode=False).xs for p in pairs5]

starts5 = jnp.concatenate([d[:-1] for d in demos5], axis=0)
ends5 = jnp.concatenate([d[1:] for d in demos5], axis=0)
deltas5 = jnp.concatenate([segment_asymmetry(m5, d) for d in demos5], axis=0)
region5 = jnp.concatenate(demos5, axis=0)
centers5 = region5[::4]                      # data-adaptive: demo vertices as features

U5, _ = recover_potential_lsq(starts5, ends5, deltas5, centers5, width=1.2, ridge=1e-3)
u_hat = jax.vmap(U5)(region5); u_hat = u_hat - jnp.mean(u_hat)
u_true = GS * jax.vmap(arm5.potential)(region5); u_true = u_true - jnp.mean(u_true)
r2_5 = 1.0 - float(jnp.sum((u_hat - u_true) ** 2) / jnp.sum(u_true**2))
print(f"5-DoF potential recovery from round-trip costs: R² = {r2_5:.3f}")
print(f"  ({len(demos5)} demos, {centers5.shape[0]} data-adaptive RBF centers, "
      "one linear solve — no grid anywhere)")

fig = go.Figure()
fig.add_trace(go.Scatter(x=np.array(u_true), y=np.array(u_hat), mode="markers",
                         marker=dict(size=5, color=PALETTE["primary"], opacity=0.6),
                         name="demo-support points"))
lim = float(jnp.max(jnp.abs(u_true))) * 1.1
fig.add_trace(go.Scatter(x=[-lim, lim], y=[-lim, lim], mode="lines",
                         line=dict(color=PALETTE["ink"], dash="dash"), name="ideal"))
flat_layout(fig, f"5-DoF: recovered vs true potential (R² = {r2_5:.3f})",
            height=460, width=520)
fig.update_xaxes(title="true gₛU (centered)"); fig.update_yaxes(title="recovered (centered)")
fig.show()
""")

md(r"""
The reading for world models: when the metric lives on a learned latent space, the
observables are exactly what this section used — **trajectories and their costs** —
and nothing else. The identifiability structure of §6 says which parts of the latent
cost those observables determine (the rotational part from shape, the potential from
timing/asymmetry, the full form from round trips), and the estimators that extract
them are linear algebra, not fragile inverse optimization. The 2-DoF sections are the
laboratory where every claim could be cross-examined against grids and pictures; this
section is the payoff, where the same tools run blind.
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Validation and honest limits

**Validation.** Everything above is gated by a runnable ladder
(`python -m experiments.arm.validate`, rungs L0-L7e) and the test suite
(`pytest tests/test_arm.py`): exact FlatTorus ops; the implicit-diff adjoint against
finite differences (the gate for every fit in §6); AVBD vs eikonal vs spray-shooting
cross-checks; barrier continuation vs cold-start; the exactness of the gravity form
$\lambda\beta = g_s\,dU$; the round-trip identity on arbitrary (non-geodesic) paths;
the 1st/2nd-order scaling law; the channel structure; and the convex recoveries.

**Honest limits.**

- **First-order gauge, second-order leak.** Projective invisibility is exact in the
  additive Randers form; the Zermelo representation leaks at $O(g_s^2)$ (measured:
  $\mathrm{dev}/g_s^2\approx 1.7$). That leak is precisely what makes shape-channel
  fits erratic rather than exactly zero.
- **The Hodge split is soft.** Round-trip data pins the drift *form*; separating it
  into exact and rotational parts from finitely many curves has near-harmonic
  ambiguity. The split R²/cosines are reported as measured.
- **Solver-in-the-loop shape inversion is fragile.** Gradient descent through the
  geodesic BVP basin-hops on these weak signals; the convex estimators replace it
  wherever the theory allows.
- **Tunneling, and barrier saturation.** The midpoint-rule energy lets a long
  segment straddle a thin obstacle body with no barrier gradient at all — and the
  barrier *repels vertices into* exactly that configuration (§7's discretization
  lesson; clearance is therefore always densified here). Separately, the softplus
  wall's hardness is fixed by its width: beyond $\alpha\approx 1$, raising it only
  inflates free space until cutting through beats detouring. The honest barrier
  range ends near $\alpha=0.6$, which is where the L4 stress rung now lives.
- **Instrumentation thins with dimension.** Eikonal fields and the a-priori
  diagnostic are low-DoF instruments (grids; solver floors that grow with basin
  multiplicity). The estimators and the planner itself are dimension-blind — that is
  §7's point — but they inherit AVBD's fixed-step stiffness limit (step 0.02 at
  5 DoF).
- **Scope.** Energy/cost geometry only: torque and velocity limits are inequality
  constraints, outside AVBD's exact-equality ALM. Timing data here is synthetic
  per-segment cost; a real robot logs it from execution.

**Next.** A real robot behind the same four protocols (`providers/real.py`); loop
demos on the Clifford torus to expose the *harmonic* corner of the gauge story
($H^1\neq 0$: closed-but-not-exact drifts, visible only through route class); and
the same estimators on a learned latent metric, where §7 says they should work
unchanged.
""")

# ===========================================================================
nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}

OUT.write_text(nbf.writes(nb), encoding="utf-8")
print(f"wrote {OUT}  ({len(cells)} cells)")

if "--run" in sys.argv:
    from nbclient import NotebookClient

    print("executing...")
    client = NotebookClient(nb, timeout=1800, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    client.execute()
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"executed and saved {OUT}")
