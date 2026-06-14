"""Assemble (and optionally execute) the marine-navigation walkthrough notebook.

Usage:
    python -m experiments.marine.build_notebook          # write unexecuted
    python -m experiments.marine.build_notebook --run    # write + execute

Produces ``experiments/marine/marine_navigation.ipynb``. The notebook reuses the
experiment package (no duplicated physics) and renders interactive Plotly 3-D
scenes, including a time-player (Play/Pause + slider) for the evolving current and
the glider traversal.
"""

import sys
from pathlib import Path

import nbformat as nbf

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OUT = HERE / "marine_navigation.ipynb"

cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def co(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ===========================================================================
md(r"""
# Time-Dependent Zermelo Routing for a 3-D Underwater Glider

This notebook is a walkthrough of the `experiments/marine` study built on the HAM
framework. It develops one idea carefully: **finding the time-optimal path of a
slow vehicle through a moving ocean current is a problem in Finsler geometry**, and
HAM's differentiable Randers machinery solves it — including the case the classical
machinery cannot, a current that *changes while you travel through it*.

The plan:

1. **Geometry** — why time-optimal navigation through a current is a Randers metric.
2. **The medium** — a physically-grounded synthetic ocean (geostrophic eddies, a
   baroclinic thermocline, Ekman drift), shown as an interactive 3-D field.
3. **Forward planning** — the global arrival-time field and a single optimal route;
   the slow glider rides a favorable deep layer.
4. **Time** — the current evolving, with a Play button.
5. **Time-dependent planning** — the contribution: a route that accounts for the
   evolving current, animated as the glider flies it.
6. **Reconstruction** — recovering the current from drifters, with an honest look at
   what is and isn't identifiable.
7. **Validation and caveats.**

Nothing here is a black box: every number comes from the same small package, and the
claims are checked against independent solutions.
""")

# ---------------------------------------------------------------------------
md(r"""
## 1. Setup

We import the experiment package and HAM's shared plotting style so the figures match
the rest of the example suite. Plotly renders the interactive 3-D scenes.
""")

co(r"""
import pathlib
import sys

# Locate the repository root (the folder containing `experiments/` and `src/`).
_p = pathlib.Path.cwd()
while not (_p / "experiments").exists() and _p != _p.parent:
    _p = _p.parent
sys.path.insert(0, str(_p))

import jax
import jax.numpy as jnp
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

from ham.vis.style import PALETTE, plotly_cones, use_ham_style

from experiments.marine import FrozenMedium, Glider, OceanMedium, build_snapshot_metric
from experiments.marine.constraints import depth_envelope
from experiments.marine.drifters import fit_kernel, fit_streamfunction, simulate_drifters
from experiments.marine.evaluate import (
    executed_arrival_time,
    navigability_map,
    recovery_metrics,
    time_saved,
    uniform_shooting_time,
)
from experiments.marine.planners import StationaryPlanner, TimeLiftedPlanner, thread_clock

pio.renderers.default = "plotly_mimetype"
use_ham_style()
jax.config.update("jax_enable_x64", False)

# Coordinates: x = east, y = north, z = depth (downward, 0 = surface). All
# quantities are non-dimensional — speeds are normalized so the glider's
# through-water speed is order 1.
EXTENT = (0.0, 10.0, 0.0, 10.0, 0.0, 1.0)
print("ready")
""")

# ---------------------------------------------------------------------------
md(r"""
## 2. The geometry: Zermelo's problem is a Randers metric

A vehicle moves with bounded speed through its surrounding water (the *sea*) and is
carried by a current $W$. Zermelo's navigation problem asks for the path that reaches
the destination in least time. Its solutions are the geodesics of a **Randers metric** —
an asymmetric Finsler metric

$$F(x, v) \;=\; \sqrt{v^\top H(x)\, v} \;+\; \beta(x)\cdot v ,$$

a Riemannian "sea" term $\sqrt{v^\top H v}$ plus a linear drift term from the current.
Travel time along a path is its $F$-length, and the least-time path is the $F$-geodesic.

Two consequences matter throughout:

- **Asymmetry.** Going *with* the current is cheaper than going *against* it:
  $F(x,v)\neq F(x,-v)$. This is the whole point — and it is exactly the asymmetry a
  Randers metric encodes.
- **A speed limit.** The construction is valid only in the *mild-wind* regime
  $\lVert W\rVert_H < 1$: the vehicle must be able to make headway against the current.
  Where the current is stronger than the vehicle, the water is genuinely non-navigable.

HAM represents this directly. `build_snapshot_metric` returns a `Randers` metric for the
current frozen at a chosen time; `randers_cost` is the per-segment travel time. We can
read the speed limit off the metric as $\lambda = 1 - \lVert W\rVert_H^2$.
""")

co(r"""
# A reusable styling for the ocean scenes: depth points downward, axes labelled.
def ocean_layout(fig, title, height=560, z_aspect=0.45, eye=(1.6, 1.5, 1.1)):
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center",
                   font=dict(size=16, color=PALETTE["ink"])),
        scene=dict(
            xaxis=dict(title="east", showspikes=False),
            yaxis=dict(title="north", showspikes=False),
            zaxis=dict(title="depth", autorange="reversed", showspikes=False),
            aspectmode="manual", aspectratio=dict(x=1, y=1, z=z_aspect),
            camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2])),
        ),
        margin=dict(l=0, r=0, t=42, b=0), height=height, paper_bgcolor="white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="#d8dee6", borderwidth=1),
    )
    return fig


def current_grid(medium, t, depth, n=11, lo=0.6, hi=9.4):
    xs = np.linspace(lo, hi, n)
    ys = np.linspace(lo, hi, n)
    base = np.array([[x, y, depth] for x in xs for y in ys], dtype=np.float32)
    W = np.array(jax.vmap(lambda p: medium.physical_current(p, jnp.asarray(float(t))))(
        jnp.asarray(base)))
    return base, W
""")

# ---------------------------------------------------------------------------
md(r"""
## 3. The medium: a physically-grounded synthetic ocean

`OceanMedium` is not an arbitrary vector field. It is the sum of named oceanographic
components, each chosen because it is real — not because the solver can handle it:

| Component | Form | Why |
|---|---|---|
| Geostrophic | $W_g=\nabla^\perp\psi$ | the mesoscale flow is the curl of sea-surface height; divergence-free |
| Baroclinic structure | two-layer (thermocline) | the deep layer flows *opposite* to the surface — the depth-riding opportunity |
| Ekman | curl-free, surface-trapped | wind-driven, divergent — a part no stream function can represent |
| Time variation | meander drift; a deep window opening and closing | the mesoscale evolves |

The vehicle is a buoyancy-driven glider: no propeller, slow, and therefore operating
*near* the mild-wind boundary. Its sea tensor is $H(x)=I/(s_\text{max}\,\text{sf}(x))^2$,
where a cold/dense "lens" lowers the achievable speed (more drag).

Below, the surface current (blue) and the deep current (orange) are shown as 3-D cones.
Note that they point in roughly opposite directions — that opposition is what a diving
glider can exploit.
""")

co(r"""
medium = OceanMedium()          # fully time-varying
glider = Glider(glide_angle_max_deg=None)

# Surface vs deep current at a moment when the deep window is open.
t_show = 2.0
base_s, W_s = current_grid(medium, t_show, depth=0.05)
base_d, W_d = current_grid(medium, t_show, depth=0.92)

fig = go.Figure()
fig.add_trace(plotly_cones(base_s, W_s, name="surface current", color=PALETTE["primary"],
                           sizeref=0.6))
fig.add_trace(plotly_cones(base_d, W_d, name="deep current", color=PALETTE["accent"],
                           sizeref=0.6))
# A faint thermocline sheet separating the two layers.
gx = np.linspace(0.6, 9.4, 2); gy = np.linspace(0.6, 9.4, 2)
GX, GY = np.meshgrid(gx, gy)
fig.add_trace(go.Surface(x=GX, y=GY, z=np.full_like(GX, medium.z_thermocline),
                         colorscale=[[0, PALETTE["surface"]], [1, PALETTE["surface"]]],
                         opacity=0.25, showscale=False, hoverinfo="skip", name="thermocline"))
ocean_layout(fig, "Surface vs deep current (the layers oppose each other)")
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
### Navigability: where the glider can and cannot go

Because the glider is slow, the strong cores of the eddies are genuinely non-navigable:
$\lambda = 1 - \lVert W\rVert_H^2 < 0$ there. This is not a defect to hide — it is the
reason a slow vehicle must route *with* the flow. HAM's metric stays well-posed in these
regions (a smooth squash caps the effective current), so the planner simply avoids them.
""")

co(r"""
n = 60
ax = np.linspace(0, 10, n)
pts = jnp.array([[x, y, 0.05] for x in ax for y in ax])
lam = np.array(navigability_map(medium, glider, pts, t=t_show)).reshape(n, n)

fig = go.Figure(go.Heatmap(
    x=ax, y=ax, z=lam.T, colorscale="RdBu", zmid=0.0, zmin=-1, zmax=1,
    colorbar=dict(title="λ", thickness=14, len=0.7)))
fig.add_trace(go.Contour(x=ax, y=ax, z=lam.T, showscale=False,
                         contours=dict(start=0, end=0, coloring="none"),
                         line=dict(color="black", width=2), hoverinfo="skip"))
fig.update_layout(title="Navigability  lam = 1 - |W|^2_H  (blue < 0: non-navigable cores)",
                  height=480, width=560, xaxis_title="east", yaxis_title="north",
                  paper_bgcolor="white", plot_bgcolor="white",
                  yaxis=dict(scaleanchor="x", scaleratio=1))
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 4. Forward planning: the arrival-time field and one optimal route

For a *frozen* current the problem is stationary, and HAM offers two complementary
solvers:

- the **volumetric eikonal** solver (an Eulerian PDE solve) returns the global
  time-to-arrival field $T(x)$ from the source — every wavefront at once;
- the **AVBD** solver (a Lagrangian path optimizer) returns a single optimal route
  between two points.

We use a steady medium with the deep favorable layer switched on, and an *eastward*
mission — against the (westward) surface jet, so diving to the reversed deep layer pays
off. The arrival-time isosurfaces below are the wavefronts; the route threads them.
""")

co(r"""
START = jnp.array([1.0, 5.0, 0.05])
END = jnp.array([9.0, 5.0, 0.05])

# Steady medium with an explicitly open favorable deep layer.
steady = OceanMedium(meander_c=0.0, eddy_drift=0.0, bc_base=0.8, bc_pulse=0.0,
                     ekman_omega=0.0)
metric = build_snapshot_metric(steady, glider, t=0.0)

stat = StationaryPlanner(max_iters=120, tol=1e-5, avbd_iters=300)
shape = (40, 40, 14)
T = np.array(stat.arrival_field(metric, START, EXTENT, shape))
print(f"arrival-time field solved on {shape} grid; T in [0, {T.max():.2f}]")
""")

co(r"""
# Depth-locked (surface) plan vs depth-free plan.
tl = TimeLiftedPlanner(n_iters=500, lr=0.03, penalty_weight=80.0)
surf = tl.plan(steady, glider, START, END, t0=0.0, n_steps=28,
               constraints=depth_envelope(0.0, 0.1))

n_steps = 28
base_line = np.linspace(np.array(START), np.array(END), n_steps + 1)
dive_init = base_line.copy()
dive_init[:, 2] = np.clip(base_line[:, 2] + 0.85 * np.sin(np.linspace(0, np.pi, n_steps + 1)), 0, 1)
deep = tl.plan(steady, glider, START, END, t0=0.0, n_steps=n_steps,
               constraints=depth_envelope(0.0, 1.0), init_path=jnp.asarray(dive_init),
               n_restarts=1)

saved = time_saved(surf.arrival_time, deep.arrival_time)
print(f"surface-locked arrival time : {float(surf.arrival_time):.2f}")
print(f"depth-riding  arrival time : {float(deep.arrival_time):.2f}")
print(f"depth-riding saves          : {100 * saved:.1f}%")
""")

co(r"""
# Interactive 3-D: arrival-time isosurfaces + the two routes.
axx = np.linspace(EXTENT[0], EXTENT[1], shape[0])
axy = np.linspace(EXTENT[2], EXTENT[3], shape[1])
axz = np.linspace(EXTENT[4], EXTENT[5], shape[2])
GX, GY, GZ = np.meshgrid(axx, axy, axz, indexing="ij")

fig = go.Figure()
fig.add_trace(go.Isosurface(
    x=GX.flatten(), y=GY.flatten(), z=GZ.flatten(), value=T.flatten(),
    isomin=float(np.quantile(T, 0.08)), isomax=float(np.quantile(T, 0.6)),
    surface_count=5, opacity=0.25, colorscale="Tealgrn", showscale=True,
    caps=dict(x_show=False, y_show=False, z_show=False),
    colorbar=dict(title="arrival time", thickness=14, len=0.6)))

for path, color, name in [(surf.path, PALETTE["muted"], "surface-locked"),
                          (deep.path, PALETTE["accent"], "depth-riding")]:
    p = np.array(path)
    fig.add_trace(go.Scatter3d(x=p[:, 0], y=p[:, 1], z=p[:, 2], mode="lines",
                               line=dict(color=color, width=7), name=name))
for pt, lbl, sym in [(START, "start", "circle"), (END, "end", "diamond")]:
    pt = np.array(pt)
    fig.add_trace(go.Scatter3d(x=[pt[0]], y=[pt[1]], z=[pt[2]], mode="markers",
                               marker=dict(size=5, color=PALETTE["ink"], symbol=sym),
                               name=lbl))
ocean_layout(fig, f"Arrival-time fronts and routes (depth-riding saves {100 * saved:.0f}%)",
             height=600)
fig.show()
""")

md(r"""
The depth-riding route dips below the thermocline where the current reverses and helps it
along, then surfaces near the destination. The two solvers agree on the cost (we check
this quantitatively in §7), which is the kind of cross-check that makes the result
trustworthy rather than merely plausible.
""")

# ---------------------------------------------------------------------------
md(r"""
## 5. Time: the current is not standing still

The real difficulty is that the current evolves. In this medium the eddies and jet drift,
and a favorable deep layer **opens and closes** with period $\tau$. A stationary
arrival-time field is only a snapshot; by the time a slow glider has crossed the basin,
the field it planned against is gone.

Press **Play** to watch the surface current evolve over one and a half periods. Drag the
slider to scrub through time.
""")

co(r"""
def animate_current(medium, times, depth=0.05, sizeref=0.6, title="Surface current over time"):
    base, _ = current_grid(medium, times[0], depth=depth, n=12)

    def cone_at(t):
        _, W = current_grid(medium, t, depth=depth, n=12)
        spd = np.linalg.norm(W, axis=-1)
        return go.Cone(x=base[:, 0], y=base[:, 1], z=base[:, 2],
                       u=W[:, 0], v=W[:, 1], w=W[:, 2], anchor="tail",
                       sizemode="scaled", sizeref=sizeref, cmin=0, cmax=float(spd.max()),
                       colorscale="Blues", showscale=False, hoverinfo="skip")

    frames = [go.Frame(name=f"{t:.1f}", data=[cone_at(t)]) for t in times]
    fig = go.Figure(data=[frames[0].data[0]], frames=frames)

    play = dict(label="▶ Play", method="animate",
                args=[None, {"frame": {"duration": 180, "redraw": True},
                             "fromcurrent": True, "transition": {"duration": 0}}])
    pause = dict(label="⏸ Pause", method="animate",
                 args=[[None], {"frame": {"duration": 0, "redraw": False},
                                "mode": "immediate"}])
    steps = [dict(method="animate", label=f.name,
                  args=[[f.name], {"frame": {"duration": 0, "redraw": True},
                                   "mode": "immediate"}]) for f in frames]
    ocean_layout(fig, title, height=560)
    fig.update_layout(
        updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                          xanchor="left", yanchor="bottom", buttons=[play, pause])],
        sliders=[dict(active=0, x=0.18, len=0.78, y=0.02, yanchor="bottom",
                      currentvalue=dict(prefix="t = ", font=dict(size=13)), steps=steps)],
    )
    return fig


t_frames = np.linspace(0.0, 1.5 * float(medium.tau), 24)
animate_current(medium, t_frames).show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 6. Time-dependent planning

This is the part the stationary solvers cannot do. The cost of crossing a segment depends
on *when* you cross it, so the metric is no longer a fixed object on space — it is threaded
through a clock. HAM's `TimeLiftedPlanner` discretizes the route and accumulates the clock
causally:

$$\Delta T_k = \texttt{randers\_cost}\big(H(m_k),\, W(m_k, t_k),\, \Delta x_k\big),
\qquad t_{k+1} = t_k + \Delta T_k,$$

then minimizes the true arrival time $\sum_k \Delta T_k$ over the interior waypoints by
gradient descent. It returns the path, so it composes with everything downstream.

We compare two plans, then **execute both under the same true, evolving current**:

- a **frozen-field** plan, optimized as if the current stayed fixed at departure — the
  implicit assumption of any stationary planner;
- a **time-aware** plan, optimized against the real clock-threaded cost.

We depart while the deep favorable window is *closed*; it reopens partway through the
crossing. The frozen planner sees no reason to dive; the time-aware planner waits and dives
into the window when it opens.
""")

co(r"""
T0 = 4.0          # depart during a closed window (it reopens at t = tau = 8)
N = 28


def dive_init(center, amp=0.8):
    b = np.linspace(np.array(START), np.array(END), N + 1)
    s = np.linspace(0, 1, N + 1)
    b[:, 2] = np.clip(b[:, 2] + amp * np.exp(-((s - center) ** 2) / (2 * 0.18 ** 2)), 0, 1)
    return jnp.asarray(b)


tl = TimeLiftedPlanner(n_iters=600, lr=0.03, penalty_weight=80.0)

frozen_medium = FrozenMedium(medium, T0)
frozen = tl.plan(frozen_medium, glider, START, END, t0=T0, n_steps=N,
                 init_path=dive_init(0.5), n_restarts=1)
frozen_executed = float(executed_arrival_time(frozen.path, medium, glider, T0))

aware = tl.plan(medium, glider, START, END, t0=T0, n_steps=N,
                init_path=dive_init(0.7), n_restarts=2)
aware_executed = float(aware.arrival_time)

adv = time_saved(frozen_executed, aware_executed)
print(f"frozen plan, executed under the true current : {frozen_executed:.2f}")
print(f"time-aware plan, executed                     : {aware_executed:.2f}")
print(f"time-aware advantage                          : {100 * adv:.1f}% faster")
""")

co(r"""
# Depth-vs-time: the time-aware plan delays its dive until the window opens.
_, times_frozen = thread_clock(frozen.path, medium, glider, T0)
_, times_aware = thread_clock(aware.path, medium, glider, T0)
times_frozen = np.array(times_frozen); times_aware = np.array(times_aware)
pf, pa = np.array(frozen.path), np.array(aware.path)

# Window open where sin(2π t / tau) > 0.
tt = np.linspace(T0, max(times_frozen[-1], times_aware[-1]), 200)
open_band = np.sin(2 * np.pi * tt / medium.tau) > 0

fig = go.Figure()
fig.add_trace(go.Scatter(x=tt, y=np.where(open_band, 1.0, np.nan), mode="lines",
                         line=dict(color=PALETTE["teal"], width=12), opacity=0.25,
                         name="deep window open", hoverinfo="skip"))
fig.add_trace(go.Scatter(x=times_frozen, y=pf[:, 2], mode="lines",
                         line=dict(color=PALETTE["muted"], width=3, dash="dash"),
                         name=f"frozen (exec {frozen_executed:.2f})"))
fig.add_trace(go.Scatter(x=times_aware, y=pa[:, 2], mode="lines",
                         line=dict(color=PALETTE["accent"], width=3),
                         name=f"time-aware (exec {aware_executed:.2f})"))
fig.update_layout(title="Depth over time — the time-aware plan dives into the open window",
                  xaxis_title="time", yaxis_title="depth",
                  yaxis=dict(autorange="reversed"), height=420, paper_bgcolor="white",
                  plot_bgcolor="white", legend=dict(x=0.02, y=0.02))
fig.show()
""")

md(r"""
Now the same comparison in space, animated. Both gliders depart together and move on the
*same* real clock; the current cones update with time. Watch the time-aware glider (orange)
drop below the thermocline as the window opens, while the frozen glider (grey) stays shallow
and arrives later. Press **Play**.
""")

co(r"""
def interp_at(route, times, t):
    if t <= times[0]:
        return route[0]
    if t >= times[-1]:
        return route[-1]
    k = int(np.searchsorted(times, t) - 1)
    f = (t - times[k]) / (times[k + 1] - times[k] + 1e-9)
    return route[k] + f * (route[k + 1] - route[k])


t_end = max(times_frozen[-1], times_aware[-1])
anim_t = np.linspace(T0, t_end, 28)
cone_base, _ = current_grid(medium, T0, depth=0.05, n=9)
cone_base_d, _ = current_grid(medium, T0, depth=0.92, n=9)


def cones(t, base, depth, scale):
    _, W = current_grid(medium, t, depth=depth, n=9)
    return go.Cone(x=base[:, 0], y=base[:, 1], z=base[:, 2], u=W[:, 0], v=W[:, 1], w=W[:, 2],
                   anchor="tail", sizemode="scaled", sizeref=scale, colorscale="Blues",
                   showscale=False, hoverinfo="skip", opacity=0.6)


def marker(pos, color):
    return go.Scatter3d(x=[pos[0]], y=[pos[1]], z=[pos[2]], mode="markers",
                        marker=dict(size=7, color=color), showlegend=False, hoverinfo="skip")


frames = []
for t in anim_t:
    fa = interp_at(pa, times_aware, t)
    ff = interp_at(pf, times_frozen, t)
    frames.append(go.Frame(name=f"{t:.1f}", data=[
        marker(fa, PALETTE["accent"]), marker(ff, PALETTE["muted"]),
        cones(t, cone_base, 0.05, 0.5), cones(t, cone_base_d, 0.92, 0.5)]))

# Static context: full routes, thermocline, endpoints.
static = [
    go.Scatter3d(x=pa[:, 0], y=pa[:, 1], z=pa[:, 2], mode="lines",
                 line=dict(color=PALETTE["accent"], width=5), name="time-aware route"),
    go.Scatter3d(x=pf[:, 0], y=pf[:, 1], z=pf[:, 2], mode="lines",
                 line=dict(color=PALETTE["muted"], width=5, dash="dash"), name="frozen route"),
]
gx = np.linspace(0.6, 9.4, 2)
GXm, GYm = np.meshgrid(gx, gx)
static.append(go.Surface(x=GXm, y=GYm, z=np.full_like(GXm, medium.z_thermocline),
                         colorscale=[[0, PALETTE["surface"]], [1, PALETTE["surface"]]],
                         opacity=0.18, showscale=False, hoverinfo="skip"))

fig = go.Figure(data=[frames[0].data[0], frames[0].data[1], frames[0].data[2],
                      frames[0].data[3], *static], frames=frames)
play = dict(label="▶ Play", method="animate",
            args=[None, {"frame": {"duration": 160, "redraw": True}, "fromcurrent": True}])
pause = dict(label="⏸ Pause", method="animate",
             args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}])
steps = [dict(method="animate", label=f.name,
              args=[[f.name], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}])
         for f in frames]
ocean_layout(fig, "Gliders flying their plans through the evolving current", height=620)
fig.update_layout(
    updatemenus=[dict(type="buttons", showactive=False, x=0.02, y=0.05,
                      xanchor="left", yanchor="bottom", buttons=[play, pause])],
    sliders=[dict(active=0, x=0.18, len=0.78, y=0.02, yanchor="bottom",
                  currentvalue=dict(prefix="t = ", font=dict(size=13)), steps=steps)])
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 7. Reconstructing the current from drifters

So far the current was known. In practice it is measured — often by **drifters**, buoys that
float passively with the flow. A drifter's track is an integral curve of $W$, so a drifter
*measures the current directly*; recovering $W$ is a regression problem, not an inverse
optimal-control problem. (This is worth stating plainly because it is a common confusion: you
do not need to invert the planner to learn the current from drifters.)

We release drifters, finite-difference their pings into velocity samples, and fit two models:

- a **geostrophic stream function** $W=\nabla^\perp\psi$ — divergence-free by construction,
  the physically correct prior;
- a **kernel smoother** — a flexible non-parametric baseline.

The honest finding: on-track, both do well. Off-track, recovery degrades — sparse drifters
simply do not constrain the current where they never went. And the stream function, by
construction, cannot represent the divergent Ekman drift. We show, not hide, this.
""")

co(r"""
recon_medium = OceanMedium(meander_c=0.0, eddy_drift=0.0, ekman_omega=0.0)  # static snapshot


def true_current(x):
    return recon_medium.physical_current(x, jnp.asarray(0.0))


obs = simulate_drifters(recon_medium, n_drifters=20, t_span=14.0, noise=0.01,
                        key=jax.random.PRNGKey(7))
psi_field = fit_streamfunction(obs, key=jax.random.PRNGKey(3), iters=1500)
kernel = fit_kernel(obs, sigma=0.8)

axg = np.linspace(1.0, 9.0, 22)
grid = jnp.array([[x, y, 0.0] for x in axg for y in axg])
dist = jnp.linalg.norm(grid[:, None, :2] - obs.positions[None, :, :], axis=-1)
on = jnp.min(dist, axis=1) < 0.9
for name, fn in [("stream-function", psi_field), ("kernel", kernel)]:
    a = recovery_metrics(fn, true_current, grid)["cosine"]
    on_c = recovery_metrics(fn, true_current, grid[on])["cosine"]
    off = recovery_metrics(fn, true_current, grid[~on])["cosine"]
    print(f"{name:16s} cosine  all={a:.3f}  on-track={on_c:.3f}  off-track={off:.3f}")
""")

co(r"""
# Quiver comparison: truth, stream-function, kernel (with drifter pings).
def quiver_trace(fn, color):
    W = np.array(jax.vmap(fn)(grid)).reshape(len(axg), len(axg), 3)
    X, Y = np.meshgrid(axg, axg, indexing="ij")
    # Build short line segments as a quiver.
    sc = 0.6
    xs, ys = [], []
    for i in range(len(axg)):
        for j in range(len(axg)):
            xs += [X[i, j], X[i, j] + sc * W[i, j, 0], None]
            ys += [Y[i, j], Y[i, j] + sc * W[i, j, 1], None]
    return go.Scatter(x=xs, y=ys, mode="lines", line=dict(color=color, width=1.4),
                      hoverinfo="skip", showlegend=False)


from plotly.subplots import make_subplots

fig = make_subplots(rows=1, cols=3,
                    subplot_titles=("true current", "geostrophic ψ", "kernel"))
for c, (fn, color) in enumerate([(true_current, PALETTE["ink"]),
                                 (psi_field, PALETTE["primary"]),
                                 (kernel, PALETTE["accent"])]):
    fig.add_trace(quiver_trace(fn, color), row=1, col=c + 1)
op = np.array(obs.positions)
for c in range(3):
    fig.add_trace(go.Scatter(x=op[:, 0], y=op[:, 1], mode="markers",
                             marker=dict(size=2.5, color=PALETTE["rose"]),
                             hoverinfo="skip", showlegend=False), row=1, col=c + 1)
fig.update_layout(height=380, width=960, paper_bgcolor="white", plot_bgcolor="white",
                  title="Reconstructing the surface current (red = drifter pings)")
fig.update_xaxes(scaleanchor="y", scaleratio=1)
fig.show()
""")

# ---------------------------------------------------------------------------
md(r"""
## 8. Validation

The claims rest on independent checks, all in `tests/test_marine.py`:

- the per-segment cost equals HAM's `Randers.metric_fn` (the planner uses the same geometry
  as the metric);
- for a spatially-uniform, time-varying current the planner reproduces an exact
  **shooting** solution;
- in the steady limit the time-lifted planner agrees with the AVBD route and the volumetric
  eikonal field — three independent solvers.

A couple of these, inline:
""")

co(r"""
# (a) Uniform time-varying current vs an exact shooting solution.
import equinox as eqx


class UniformMedium(eqx.Module):
    amp: float = eqx.field(static=True, default=0.12)
    omega: float = eqx.field(static=True, default=0.2)

    def physical_current(self, x, t):
        a = self.omega * t
        return jnp.array([self.amp * jnp.cos(a), self.amp * jnp.sin(a), 0.0])

    def speed_factor(self, x):
        return jnp.asarray(1.0)


um = UniformMedium()
g2 = Glider(s_max=0.85, glide_angle_max_deg=None)
t_star = float(uniform_shooting_time(lambda t: um.physical_current(START, t),
                                     g2.s_max, START, END, t0=0.0))
res = TimeLiftedPlanner(n_iters=500, lr=0.05, penalty_weight=0.0).plan(
    um, g2, START, END, t0=0.0, n_steps=20, constraints=[])
print(f"shooting solution      : {t_star:.3f}")
print(f"time-lifted planner    : {float(res.arrival_time):.3f}")
print(f"relative difference    : {abs(float(res.arrival_time) - t_star) / t_star * 100:.1f}%")
""")

# ---------------------------------------------------------------------------
md(r"""
## 9. What this is, and what it isn't

**What it is.** A demonstration that time-optimal navigation through an ocean current is a
Randers-geometry problem HAM solves directly, including the time-dependent case via a
clock-threaded path solver — with results cross-checked against independent solutions.

**Honest limits.**

- The time-lifted planner solves a non-convex boundary-value problem; the routes are *local*
  optima, warm-started and multi-started. Reported numbers are reproducible but not certified
  global.
- The glider operates near the mild-wind boundary by design; the smooth cap keeps the metric
  well-posed but distorts genuinely non-navigable currents. We map that regime rather than
  assume it away.
- Drifters constrain the current only where they travel; off-track recovery is limited, and a
  stream-function model cannot recover the divergent Ekman part.
- Vertical ocean currents are neglected (they are millimetres per second); the glider's own
  vertical motion is what changes its depth.

**Next.** The same frame (medium / vehicle / constraints / planners) supports a 2-D surface
vessel, where the sea tensor $H$ becomes *anisotropic* — a directional "speed polar" from wave
added resistance — exercising the asymmetric Randers structure further. New physics enters as a
single `Constraint`, which both planners already consume.
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
    client = NotebookClient(nb, timeout=900, kernel_name="python3",
                            resources={"metadata": {"path": str(REPO)}})
    client.execute()
    OUT.write_text(nbf.writes(nb), encoding="utf-8")
    print(f"executed and saved {OUT}")
