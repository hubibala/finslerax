"""A small, consistent visualization style for the HAM example notebooks.

The goal is a clean, modern, publication-grade look with a restrained palette.
Two backends share that palette so figures stay cohesive:

* **Matplotlib** (``use_ham_style``, ``axes3d``, ``tangent_arrows`` …) for the
  static 2-D analytical plots (convergence curves, indicatrices, image strips).
* **Plotly** (``plotly_layout``, ``plotly_sphere``, ``plotly_cones`` …) for the
  interactive 3-D scenes, which the reader can rotate to inspect a geodesic or a
  field from any angle. Vector fields use magnitude-scaled 3-D cones, so a field
  with varying speeds is shown as defined. Plotly is imported lazily so it is
  only required when a 3-D helper is actually called.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "PALETTE",
    "CYCLE",
    "use_ham_style",
    "axes3d",
    "style_axes3d",
    "draw_sphere",
    "draw_surface",
    "draw_path",
    "tangent_arrows",
    "plotly_layout",
    "plotly_sphere",
    "plotly_surface",
    "plotly_mesh",
    "plotly_path",
    "plotly_cones",
    "palette_colorscale",
]

# A restrained, colour-blind-aware palette. Colours carry consistent meaning
# across the notebook suite: PRIMARY for the "reference" object, ACCENT for the
# "result" of interest, the rest for additional series.
PALETTE = {
    "ink": "#1d1d1f",
    "primary": "#2f6db5",   # blue   — reference / baseline
    "accent": "#e8833a",    # orange — the result being highlighted
    "teal": "#2ca6a4",
    "violet": "#7b5bd6",
    "rose": "#d6456b",
    "green": "#2f9e44",
    "muted": "#9aa3ad",     # grids, secondary strokes
    "surface": "#cdd8e3",   # manifold surfaces
}
CYCLE = [PALETTE[k] for k in ("primary", "accent", "teal", "violet", "rose", "green")]


def use_ham_style() -> None:
    """Apply the HAM Matplotlib theme (idempotent)."""
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 120,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "semibold",
        "axes.titlepad": 10,
        "axes.labelsize": 11,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": "#c7ced6",
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": "#e6e9ee",
        "grid.linewidth": 1.0,
        "axes.prop_cycle": plt.cycler(color=CYCLE),
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.frameon": False,
        "legend.fontsize": 10,
        "lines.linewidth": 2.2,
        "image.cmap": "magma",
    })


def axes3d(figsize=(7, 6), elev=22, azim=-58, frame=False):
    """Create a cleanly styled 3-D axis (frameless backdrop by default)."""
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=elev, azim=azim)
    style_axes3d(ax, frame=frame)
    return fig, ax


def style_axes3d(ax, frame=False):
    """Strip 3-D chartjunk and equalize aspect.

    With ``frame=False`` (default) the axis box, panes and ticks are removed
    entirely — the manifold surface itself provides spatial context, which is
    the cleanest look for geometry figures. ``frame=True`` keeps faint panes for
    plots where a sense of scale helps (e.g. a paraboloid bowl).
    """
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    if not frame:
        ax.set_axis_off()
        return ax
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor((1, 1, 1, 0))
        axis.pane.set_edgecolor((0, 0, 0, 0))
        axis.line.set_color((0, 0, 0, 0.12))
    ax.grid(False)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    return ax


def draw_sphere(ax, radius=1.0, color=None, alpha=0.12, n=60):
    """Soft translucent sphere with a faint wireframe — a quiet backdrop."""
    color = color or PALETTE["surface"]
    u, v = np.mgrid[0:2 * np.pi:complex(0, n), 0:np.pi:complex(0, n // 2)]
    x = radius * np.cos(u) * np.sin(v)
    y = radius * np.sin(u) * np.sin(v)
    z = radius * np.cos(v)
    ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0,
                    antialiased=True, shade=False, zorder=0)
    ax.plot_wireframe(x, y, z, color="white", alpha=0.35, linewidth=0.5, zorder=0)


def draw_surface(ax, X, Y, Z, color=None, alpha=0.35, wire=True):
    """Soft translucent parametric surface for non-spherical manifolds."""
    color = color or PALETTE["surface"]
    ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0,
                    antialiased=True, shade=True, zorder=0)
    if wire:
        ax.plot_wireframe(X, Y, Z, color="white", alpha=0.25, linewidth=0.4,
                          rcount=24, ccount=24, zorder=0)


def draw_path(ax, xs, color=None, label=None, lw=3.0, marker_start=True, **kw):
    """Draw a 3-D path with an optional start marker."""
    xs = np.asarray(xs)
    color = color or PALETTE["accent"]
    ax.plot(xs[:, 0], xs[:, 1], xs[:, 2], color=color, lw=lw, label=label,
            solid_capstyle="round", **kw)
    if marker_start:
        ax.scatter(*xs[0], color=color, s=28, depthshade=False, zorder=5)


def tangent_arrows(ax, base, vecs, color=None, scale=0.22, lw=1.4, alpha=0.9,
                   label=None, normalize=False):
    """Draw classical arrow glyphs for a tangent field.

    Arrows are anchored at ``base`` points and drawn along ``vecs`` (which should
    already lie in the tangent space). ``scale`` sets a uniform visual length so
    the field reads as a clean quiver rather than a thicket.
    """
    base = np.asarray(base); vecs = np.asarray(vecs)
    color = color or PALETTE["primary"]
    if normalize:
        n = np.linalg.norm(vecs, axis=1, keepdims=True)
        vecs = vecs / np.where(n > 1e-9, n, 1.0)
    ax.quiver(base[:, 0], base[:, 1], base[:, 2],
              vecs[:, 0], vecs[:, 1], vecs[:, 2],
              length=scale, normalize=False, color=color, linewidth=lw,
              alpha=alpha, arrow_length_ratio=0.4, label=label)


# --------------------------------------------------------------------------- #
# Interactive Plotly helpers (3-D scenes the reader can rotate)
# --------------------------------------------------------------------------- #
def _solid(color):
    """A flat, single-colour Plotly colorscale."""
    return [[0.0, color], [1.0, color]]


def plotly_layout(fig, title=None, height=620, eye=(1.35, 1.25, 0.8)):
    """Apply the shared clean, frameless 3-D scene styling to a Plotly figure."""
    clean = dict(showbackground=False, showgrid=False, zeroline=False,
                 showticklabels=False, title="", visible=False)
    fig.update_layout(
        title=dict(text=title or "", x=0.5, xanchor="center",
                   font=dict(size=17, color=PALETTE["ink"])),
        scene=dict(xaxis=clean, yaxis=clean, zaxis=clean, aspectmode="data",
                   camera=dict(eye=dict(x=eye[0], y=eye[1], z=eye[2]))),
        margin=dict(l=0, r=0, t=44, b=0),
        height=height,
        paper_bgcolor="white",
        legend=dict(x=0.02, y=0.98, bgcolor="rgba(255,255,255,0.6)",
                    bordercolor="#d8dee6", borderwidth=1, font=dict(size=12)),
    )
    return fig


def plotly_sphere(radius=1.0, color=None, opacity=0.16, n=60):
    """A soft, evenly-lit sphere surface as a quiet backdrop."""
    import plotly.graph_objects as go
    color = color or PALETTE["surface"]
    u, v = np.mgrid[0:2 * np.pi:complex(0, n), 0:np.pi:complex(0, n // 2)]
    return go.Surface(
        x=radius * np.cos(u) * np.sin(v), y=radius * np.sin(u) * np.sin(v),
        z=radius * np.cos(v), colorscale=_solid(color), opacity=opacity,
        showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.9, diffuse=0.2, specular=0.05), name="")


def plotly_surface(X, Y, Z, color=None, opacity=0.42):
    """A soft single-colour parametric surface (paraboloid, torus, …)."""
    import plotly.graph_objects as go
    color = color or PALETTE["surface"]
    return go.Surface(x=X, y=Y, z=Z, colorscale=_solid(color), opacity=opacity,
                      showscale=False, hoverinfo="skip",
                      lighting=dict(ambient=0.75, diffuse=0.5, specular=0.08))


def plotly_mesh(verts, faces, color=None, opacity=0.45):
    """A flat-shaded triangle mesh surface."""
    import plotly.graph_objects as go
    verts = np.asarray(verts); faces = np.asarray(faces)
    color = color or PALETTE["surface"]
    return go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                     i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                     color=color, opacity=opacity, flatshading=True,
                     hoverinfo="skip", lighting=dict(ambient=0.7, diffuse=0.6),
                     name="")


def plotly_path(xs, color=None, name=None, width=6, dash=None, show_start=True):
    """A 3-D path as a Plotly line, optionally with a start marker."""
    import plotly.graph_objects as go
    xs = np.asarray(xs)
    color = color or PALETTE["accent"]
    mode = "lines+markers" if show_start else "lines"
    marker = dict(size=[5] + [0] * (len(xs) - 1), color=color) if show_start else None
    return go.Scatter3d(x=xs[:, 0], y=xs[:, 1], z=xs[:, 2], mode=mode,
                        line=dict(color=color, width=width, dash=dash),
                        marker=marker, name=name, showlegend=name is not None)


def palette_colorscale(color=None):
    """A light→``color`` sequential colorscale for magnitude shading."""
    return [[0.0, "#eef2f7"], [1.0, color or PALETTE["primary"]]]


def plotly_cones(base, vecs, name=None, color=None, colorscale=None,
                 sizeref=0.5, sizemode="scaled", showscale=False, colorbar_title=None):
    """A 3-D cone (vector) field that preserves the true vector magnitudes.

    Cone size follows the vector norm (``sizemode='scaled'``), so a field with
    varying speeds is shown as defined — no normalization. Pass ``color`` for a
    single hue, or ``colorscale`` to shade cones by magnitude. Returns one Plotly
    trace; add it with ``fig.add_trace(plotly_cones(...))``.
    """
    import plotly.graph_objects as go
    base = np.asarray(base, float); vecs = np.asarray(vecs, float)
    if color is not None and colorscale is None:
        colorscale = _solid(color)
    elif colorscale is None:
        colorscale = palette_colorscale()
    return go.Cone(
        x=base[:, 0], y=base[:, 1], z=base[:, 2],
        u=vecs[:, 0], v=vecs[:, 1], w=vecs[:, 2],
        anchor="tail", sizemode=sizemode, sizeref=sizeref, colorscale=colorscale,
        showscale=showscale, name=name, hoverinfo="skip",
        colorbar=dict(title=colorbar_title, thickness=14, len=0.6) if showscale else None,
    )
