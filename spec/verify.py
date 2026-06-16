"""Analytic ground-truth verification of the HAM core.

Runs a battery of checks where the geometry has a *known closed-form* answer and
compares the library's autodiff/solver output against it. This is the evidence
behind spec/verification_report.md. Reproduce with::

    JAX_PLATFORMS=cpu python spec/verify.py

Every check is independent of the library's own tests: the reference values are
computed here from textbook formulas (Christoffel symbols, Gaussian curvature,
Zermelo travel times, Poincaré distance), not from HAM.
"""

import jax
import jax.numpy as jnp

from ham.geometry import EuclideanSpace
from ham.geometry.curvature import sectional_curvature
from ham.geometry.zoo import Randers, Riemannian
from ham.solvers.avbd import AVBDSolver
from ham.solvers.gauss_newton import GaussNewtonGeodesic
from ham.utils import causal_wind_scale

jax.config.update("jax_platform_name", "cpu")

PLANE = EuclideanSpace(2)


def line(name, got, ref, tol, unit=""):
    err = abs(float(got) - float(ref))
    status = "PASS" if err <= tol else "FAIL"
    print(
        f"  [{status}] {name:<46} got={float(got):+.6f}{unit}  "
        f"ref={float(ref):+.6f}{unit}  |err|={err:.2e}"
    )
    return status == "PASS"


def main():
    ok = True
    print("=" * 78)
    print("HAM analytic verification")
    print("=" * 78)

    # 1. Geodesic spray on the Poincare upper half-plane (g = I / y^2).
    #    Geodesic acceleration a^k = -Gamma^k_ij v^i v^j with Christoffels
    #    Gamma^x_xy = -1/y, Gamma^y_xx = 1/y, Gamma^y_yy = -1/y gives
    #    a^x = (2/y) vx vy,  a^y = (1/y)(vy^2 - vx^2). The library returns the
    #    spray G with a = -2 G.
    print("\n1. Geodesic spray (Poincare half-plane, g = I/y^2)")
    poincare = Riemannian(PLANE, lambda x: jnp.eye(2) / x[1] ** 2)
    x = jnp.array([0.0, 1.0])
    v = jnp.array([1.0, 0.5])
    a_lib = -2.0 * poincare.spray(x, v)
    a_ref = jnp.array([2.0 * v[0] * v[1], v[1] ** 2 - v[0] ** 2])  # y = 1
    ok &= line("acceleration a^x", a_lib[0], a_ref[0], 1e-3)
    ok &= line("acceleration a^y", a_lib[1], a_ref[1], 1e-3)

    # 2. Randers / Zermelo cost: constant wind w along +x, identity sea. The
    #    time to go a unit step downwind is 1/(1+w), upwind 1/(1-w).
    print("\n2. Randers cost (Zermelo, identity sea, wind w=0.5 along +x)")
    w = 0.5
    randers = Randers(PLANE, lambda x: jnp.eye(2), lambda x: jnp.array([w, 0.0]))
    o = jnp.zeros(2)
    ok &= line("F(east)  [downwind]", randers.metric_fn(o, jnp.array([1.0, 0.0])),
               1.0 / (1 + w), 1e-4)
    ok &= line("F(west)  [upwind]", randers.metric_fn(o, jnp.array([-1.0, 0.0])),
               1.0 / (1 - w), 1e-4)
    ok &= line("F(north) [crosswind]", randers.metric_fn(o, jnp.array([0.0, 1.0])),
               1.0 / jnp.sqrt(1 - w * w), 1e-4)

    # 3. Sectional curvature against closed-form Gaussian curvature.
    print("\n3. Sectional curvature (autodiff vs analytic Gaussian curvature)")
    sphere = Riemannian(PLANE, lambda x: (4.0 / (1 + jnp.sum(x**2)) ** 2) * jnp.eye(2))
    ok &= line("stereographic sphere K", sectional_curvature(
        sphere, jnp.array([0.3, 0.0]), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])),
        1.0, 1e-2)
    surf = Riemannian(PLANE, lambda x: jnp.diag(jnp.array([1.0, 1.0 + x[0] ** 2])))
    ok &= line("surface of revolution K (x=0.5)", sectional_curvature(
        surf, jnp.array([0.5, 0.0]), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])),
        -1.0 / 1.25 ** 2, 1e-3)
    ok &= line("Poincare K", sectional_curvature(
        poincare, jnp.array([0.0, 1.0]), jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])),
        -1.0, 1e-2)

    # 4. Causal wind clamp: identity inside the physical region, strict bound
    #    above it (the fix replacing the tanh squash).
    print("\n4. Causal wind clamp (soft-min, max_speed = 1 - 1e-5)")
    c = 1.0 - 1e-5
    for r in (0.5, 0.9):
        s = causal_wind_scale(jnp.array(r), c)
        ok &= line(f"identity at ||W||={r}", s * r, r, 5e-3)
    for r in (1.5, 5.0):
        s = float(causal_wind_scale(jnp.array(r), c))
        passed = s * r < 1.0
        ok &= passed
        print(f"  [{'PASS' if passed else 'FAIL'}] strict bound at ||W||={r:<18} "
              f"clamped={s * r:.6f} < 1")

    # 5. Gauss-Newton geodesic length vs analytic Poincare distance.
    #    d((x1,y1),(x2,y2)) = arccosh(1 + ((dx)^2+(dy)^2)/(2 y1 y2)).
    print("\n5. Gauss-Newton geodesic length (Poincare)")
    p0 = jnp.array([-0.5, 1.0])
    p1 = jnp.array([0.5, 1.0])
    d_ref = jnp.arccosh(1 + ((p1[0] - p0[0]) ** 2) / (2 * p0[1] * p1[1]))
    gn = GaussNewtonGeodesic(iterations=60)
    traj_gn = gn.solve(poincare, p0, p1, n_steps=64, train_mode=False)
    len_gn = poincare.arc_length(traj_gn.xs)
    ok &= line("GN length vs arccosh(1.5)", len_gn, d_ref, 5e-3)

    # 6. AVBD on the same BVP (documented: converges slowly on stiff metrics;
    #    reported for honesty, not asserted tightly).
    print("\n6. AVBD geodesic length (same BVP) -- informational")
    avbd = AVBDSolver(iterations=400, step_size=0.01)
    traj_av = avbd.solve(poincare, p0, p1, n_steps=64)
    len_av = poincare.arc_length(traj_av.xs)
    print(f"  [INFO] AVBD length={float(len_av):.4f}  vs analytic={float(d_ref):.4f}  "
          f"rel.gap={float(abs(len_av - d_ref) / d_ref) * 100:.1f}%  "
          f"(GN rel.err={float(abs(len_gn - d_ref) / d_ref) * 100:.3f}%)")

    print("\n" + "=" * 78)
    print("OVERALL:", "ALL ANALYTIC CHECKS PASS" if ok else "SOME CHECKS FAILED")
    print("=" * 78)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
