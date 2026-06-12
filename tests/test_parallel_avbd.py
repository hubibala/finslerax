import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry import Sphere
from ham.geometry.zoo import Euclidean, Randers
from ham.solvers.avbd import AVBDSolver

# config.update("jax_enable_x64", True)
from ham.solvers.coloring import chain_coloring, greedy_coloring, mesh_vertex_coloring


class TestChainColoring(unittest.TestCase):

    def test_two_colors_cover_all_inner(self):
        """All inner vertices appear exactly once across both colors."""
        for n_inner in [1, 2, 5, 9, 10, 20]:
            c0, c1 = chain_coloring(n_inner)
            all_ids = jnp.sort(jnp.concatenate([c0, c1]))
            expected = jnp.arange(1, n_inner + 1)
            np.testing.assert_array_equal(all_ids, expected)

    def test_independence(self):
        """No two vertices in the same color are adjacent."""
        for n_inner in [5, 10, 15]:
            c0, c1 = chain_coloring(n_inner)
            for group in [c0, c1]:
                if len(group) > 1:
                    diffs = jnp.diff(group)
                    self.assertTrue(jnp.all(diffs >= 2),
                                    f"Adjacent vertices in same color: {group}")

    def test_single_vertex(self):
        c0, c1 = chain_coloring(1)
        self.assertEqual(len(c0), 1)
        self.assertEqual(len(c1), 0)
        self.assertEqual(int(c0[0]), 1)


class TestGreedyColoring(unittest.TestCase):

    def test_triangle(self):
        adj = {0: {1, 2}, 1: {0, 2}, 2: {0, 1}}
        groups = greedy_coloring(adj, 3)
        self.assertEqual(len(groups), 3)
        all_verts = sorted(v for g in groups for v in g)
        self.assertEqual(all_verts, [0, 1, 2])

    def test_path_graph(self):
        """A 5-vertex path should be 2-colorable."""
        adj = {0: {1}, 1: {0, 2}, 2: {1, 3}, 3: {2, 4}, 4: {3}}
        groups = greedy_coloring(adj, 5)
        self.assertLessEqual(len(groups), 2)
        for g in groups:
            for v in g:
                for u in g:
                    if v != u:
                        self.assertNotIn(u, adj[v])

    def test_isolated_vertices(self):
        adj = {0: set(), 1: set(), 2: set()}
        groups = greedy_coloring(adj, 3)
        self.assertEqual(len(groups), 1)
        self.assertEqual(sorted(groups[0]), [0, 1, 2])

    def test_complete_graph_k4(self):
        adj = {i: set(range(4)) - {i} for i in range(4)}
        groups = greedy_coloring(adj, 4)
        self.assertEqual(len(groups), 4)


class TestMeshVertexColoring(unittest.TestCase):

    def test_single_triangle(self):
        faces = jnp.array([[0, 1, 2]])
        groups = mesh_vertex_coloring(faces, 3)
        self.assertGreaterEqual(len(groups), 3)
        all_verts = sorted(int(v) for g in groups for v in g)
        self.assertEqual(all_verts, [0, 1, 2])

    def test_two_triangles_shared_edge(self):
        #   0--1
        #   |/ |
        #   2--3
        faces = jnp.array([[0, 1, 2], [1, 3, 2]])
        groups = mesh_vertex_coloring(faces, 4)
        # Verify independence: no two in same group share an edge
        adj = {0: {1, 2}, 1: {0, 2, 3}, 2: {0, 1, 3}, 3: {1, 2}}
        for g in groups:
            for v in g:
                for u in g:
                    if int(v) != int(u):
                        self.assertNotIn(int(u), adj[int(v)])


class TestParallelAVBDSolver(unittest.TestCase):

    def setUp(self):
        self.key = jax.random.PRNGKey(42)

    def test_parallel_euclidean_straight_line(self):
        """Parallel solver should find a straight line in Euclidean space."""
        from ham.geometry.manifolds.euclidean_space import EuclideanSpace
        metric = Euclidean(EuclideanSpace(dim=2))
        solver = AVBDSolver(iterations=50, step_size=0.1, parallel=True)

        p0 = jnp.array([0.0, 0.0])
        p1 = jnp.array([1.0, 0.0])
        traj = solver.solve(metric, p0, p1, n_steps=10)

        # All y-coordinates should stay near zero
        np.testing.assert_allclose(traj.xs[:, 1], 0.0, atol=1e-2)
        # x-coordinates should be approximately evenly spaced
        expected_x = jnp.linspace(0, 1, 11)
        np.testing.assert_allclose(traj.xs[:, 0], expected_x, atol=5e-2)

    def test_parallel_sphere_geodesic(self):
        """Parallel solver on a sphere should produce sensible energy."""
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        solver_seq = AVBDSolver(iterations=50, step_size=0.1, parallel=False)
        solver_par = AVBDSolver(iterations=50, step_size=0.1, parallel=True)

        p0 = jnp.array([1.0, 0.0, 0.0])
        p1 = jnp.array([0.0, 1.0, 0.0])

        traj_seq = solver_seq.solve(metric, p0, p1, n_steps=10, key=self.key)
        traj_par = solver_par.solve(metric, p0, p1, n_steps=10, key=self.key)

        # Both should produce valid trajectories on the sphere
        norms_par = jnp.sqrt(jnp.sum(traj_par.xs**2, axis=-1))
        np.testing.assert_allclose(norms_par, 1.0, atol=1e-2)

        # Energies should be in the same ballpark (parallel may converge
        # slightly differently due to Jacobi-within-color vs pure Gauss-Seidel)
        self.assertLess(abs(float(traj_par.energy) - float(traj_seq.energy)),
                        0.5 * float(traj_seq.energy) + 0.1)

    def test_parallel_jit(self):
        """Parallel solver must be JIT-compilable."""
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        solver = AVBDSolver(iterations=20, step_size=0.1, parallel=True)

        p0 = jnp.array([1.0, 0.0, 0.0])
        p1 = jnp.array([0.0, 1.0, 0.0])

        jit_solve = jax.jit(solver.solve, static_argnames=["n_steps"])
        traj = jit_solve(metric, p0, p1, n_steps=10)
        self.assertEqual(traj.xs.shape, (11, 3))
        self.assertTrue(jnp.isfinite(traj.energy))

    def test_parallel_vmap(self):
        """Parallel solver must work under vmap."""
        sphere = Sphere(radius=1.0)
        metric = Euclidean(sphere)
        solver = AVBDSolver(iterations=20, step_size=0.1, parallel=True)

        p0s = jnp.tile(jnp.array([1.0, 0.0, 0.0]), (3, 1))
        p1s = jnp.tile(jnp.array([0.0, 1.0, 0.0]), (3, 1))

        batched = jax.vmap(lambda p0, p1: solver.solve(metric, p0, p1, n_steps=8))
        trajs = batched(p0s, p1s)
        self.assertEqual(trajs.xs.shape, (3, 9, 3))

    def test_parallel_differentiability(self):
        """Gradients through the parallel solver should be finite."""
        sphere = Sphere(radius=1.0)

        def loss(wind_speed):
            def h_net(x): return jnp.eye(3)
            def w_net(x): return jnp.array([wind_speed, 0.0, 0.0])
            metric = Randers(sphere, h_net, w_net)
            solver = AVBDSolver(iterations=20, step_size=0.05, parallel=True)
            p0 = jnp.array([1.0, 0.0, 0.0])
            p1 = jnp.array([0.0, 1.0, 0.0])
            return solver.solve(metric, p0, p1, n_steps=8).energy

        grad_w = jax.grad(loss)(0.3)
        self.assertTrue(jnp.isfinite(grad_w))

    def test_parallel_matches_sequential_energy_order(self):
        """Parallel solver energy should be within 2x of sequential."""
        from ham.geometry.manifolds.euclidean_space import EuclideanSpace
        metric = Euclidean(EuclideanSpace(dim=3))
        solver_seq = AVBDSolver(iterations=80, step_size=0.1, parallel=False)
        solver_par = AVBDSolver(iterations=80, step_size=0.1, parallel=True)

        p0 = jnp.array([0.0, 0.0, 0.0])
        p1 = jnp.array([1.0, 1.0, 1.0])

        traj_seq = solver_seq.solve(metric, p0, p1, n_steps=15, key=self.key)
        traj_par = solver_par.solve(metric, p0, p1, n_steps=15, key=self.key)

        # In Euclidean space both should find the straight line
        ratio = float(traj_par.energy) / max(float(traj_seq.energy), 1e-10)
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)


if __name__ == "__main__":
    unittest.main()
