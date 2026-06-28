import unittest

import jax
import jax.numpy as jnp
import numpy as np

from ham.geometry.mesh import TriangularMesh
from ham.geometry.zoo import DiscreteRanders, Euclidean
from ham.solvers.avbd import AVBDSolver


class TestMeshSolver(unittest.TestCase):

    def setUp(self):
        # Increased iterations for convergence on sharp geometries
        # Using the new robust AVBD parameters (beta=10.0 for stiffness)
        self.solver = AVBDSolver(step_size=0.05, beta=10.0, iterations=200)
        self.key = jax.random.PRNGKey(42)

    def test_pyramid_surface_constraint(self):
        """
        Verifies that the solver respects the manifold constraint.
        Path: Over a pyramid from (-1, 0) to (1, 0).
        Expected: The path must climb the surface (z > 0), not tunnel through the base (z=0).
        """
        # Define 4-faced Pyramid vertices
        # 0: Left, 1: Right, 2: Back, 3: Front, 4: Apex
        verts = jnp.array([
            [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=float)

        # Faces connecting base points to apex
        # Face 0: Back-Left (0-2-4)
        # Face 1: Back-Right (2-1-4)
        # Face 2: Front-Right (1-3-4)
        # Face 3: Front-Left (3-0-4)
        faces = jnp.array([
            [0, 2, 4], [2, 1, 4], [1, 3, 4], [3, 0, 4]
        ])

        mesh = TriangularMesh(verts, faces)
        metric = Euclidean(mesh)

        # Start and End slightly elevated to ensure projection maps to surface faces
        start = jnp.array([-0.9, 0.0, 0.05])
        end = jnp.array([0.9, 0.0, 0.05])

        traj = self.solver.solve(metric, start, end, n_steps=20)

        # 1. Endpoint Fidelity
        np.testing.assert_allclose(traj.xs[0], start, atol=1e-2)
        np.testing.assert_allclose(traj.xs[-1], end, atol=1e-2)

        # 2. Surface Adherence
        # The straight line chord would have z=0.05.
        # The geodesic must climb the peak (z approaches 1.0).
        mid_z = jnp.max(traj.xs[:, 2])
        self.assertGreater(mid_z, 0.5, "Path failed to climb the pyramid surface.")

    def test_obstacle_avoidance(self):
        """
        Verifies that the solver respects varying metric costs on the mesh.
        Scenario: 
            - Same Pyramid geometry.
            - Path from Front (y=0.9) to Back (y=-0.9).
            - Right side (x > 0) has HIGH HEADWIND ("Lava").
            - Left side (x < 0) has NO WIND ("Grass").
            
        Expected: 
            The path should swing to the Left (negative x) to avoid the headwind on the right,
            breaking the geometric symmetry.
        """
        verts = jnp.array([
            [-1, 0, 0], [1, 0, 0], [0, -1, 0], [0, 1, 0], [0, 0, 1]
        ], dtype=float)

        faces = jnp.array([
            [0, 2, 4], # Face 0: Left (x < 0)
            [2, 1, 4], # Face 1: Right (x > 0)
            [1, 3, 4], # Face 2: Right (x > 0)
            [3, 0, 4]  # Face 3: Left (x < 0)
        ])

        mesh = TriangularMesh(verts, faces)

        # Define Winds: Faces 1 & 2 (Right side) have strong headwind against -Y movement.
        # Path is roughly moving in -Y direction (0.9 to -0.9).
        # A wind vector of (0, 0.95, 0) opposes this motion strongly.
        face_winds = jnp.array([
            [0.0, 0.0, 0.0],   # Face 0: Calm
            [0.0, 0.95, 0.0],  # Face 1: Strong Headwind (+Y)
            [0.0, 0.95, 0.0],  # Face 2: Strong Headwind (+Y)
            [0.0, 0.0, 0.0]    # Face 3: Calm
        ])

        metric = DiscreteRanders(mesh, face_winds)

        # Path from Front to Back
        start = jnp.array([0.0, 0.9, 0.1])
        end = jnp.array([0.0, -0.9, 0.1])

        # Higher iterations needed: the obstacle avoidance geometry with near-singular
        # 0.95 headwind requires more convergence steps to fully deviate the path,
        # especially since the AVBD permutation order is data-dependent.
        solver_hard = AVBDSolver(step_size=0.05, beta=10.0, iterations=400)

        traj = solver_hard.solve(metric, start, end, n_steps=40)

        # Calculate mean X position of the trajectory
        mean_x = jnp.mean(traj.xs[:, 0])

        # Visual check of the deviation
        print(f"Mean X deviation: {mean_x:.4f} (Expected < 0)")

        # Assert the path deviates significantly to the Left (Negative X)
        # Deviation might be slightly less than with scalar cost multiplier, but should still be significant
        self.assertLess(mean_x, -0.1, "Path did not avoid the high-wind region on the right.")

if __name__ == '__main__':
    unittest.main()
