"""Neural network building blocks for learned Finsler geometry.

Provides differentiable parameterizations for vector fields (wind) and
positive-definite matrix fields (Riemannian base metrics) used by the
learned metric classes in ham.models.learned.

All modules are compatible with JAX transforms (jit, vmap, grad).
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from ..utils.math import PSD_EPS


class RandomFourierFeatures(eqx.Module):
    """Random Fourier Feature (RFF) embedding (Rahimi & Recht, 2007).

    Maps input x ∈ R^D to a 2M-dimensional feature space via:
        γ(x) = [cos(Bx), sin(Bx)]
    where B ∈ R^{M×D} is sampled from N(0, scale²I). This approximates a
    shift-invariant kernel and mitigates spectral bias in coordinate-based
    MLPs, allowing them to learn high-frequency spatial variation.

    The frequency matrix B is frozen (non-trainable) via stop_gradient.

    See also:
        ham.models.learned.NeuralRanders for typical usage.
    """

    B: jnp.ndarray

    def __init__(self, in_dim: int, mapping_size: int, scale: float, key: jax.Array):
        """Initializes the Fourier frequency matrix.

        Args:
            in_dim: Dimensionality of the input vector D.
            mapping_size: Number of random frequencies M.
            scale: Standard deviation of the Gaussian sampling for B.
            key: JAX PRNG key.
        """
        # B is sampled from N(0, scale^2)
        # We wrap in stop_gradient to ensure it remains a frozen basis.
        self.B = jax.lax.stop_gradient(
            jax.random.normal(key, (mapping_size, in_dim)) * scale
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Computes the embedding γ(x).

        Args:
            x: Input vector of shape (D,).

        Returns:
            Fourier-embedded vector of shape (2*M,).
        """
        # x: (D,) -> Bx: (M,)
        projected = jnp.dot(self.B, x)
        return jnp.concatenate([jnp.cos(projected), jnp.sin(projected)], axis=0)


class VectorField(eqx.Module):
    """Neural-network approximation of a smooth vector field W: R^D → R^D.

    In the Zermelo parameterization of Randers metrics (see MATH_SPEC § 5),
    this network produces the raw wind field W^i(x). The strong-convexity
    constraint ||W||_h < 1 is enforced downstream by the RandersMetric.

    Uses tanh activation for C^∞ smoothness, which is critical for higher-order
    autodiff through the spray and Berwald connection.

    Note:
        use_fourier=True by default because wind fields typically exhibit
        high-frequency spatial variation. For smoother fields, use
        use_fourier=False.

    See also:
        PSDMatrixField — counterpart for the Riemannian base metric.
        ham.models.learned.NeuralRanders — primary consumer.
    """

    embedding: Optional[RandomFourierFeatures]
    mlp: eqx.nn.MLP

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int,
        key: jax.Array,
        use_fourier: bool = True,
        fourier_scale: float = 10.0,
    ):
        """Initializes the vector field network.

        Args:
            dim: Input and output dimensionality D.
            hidden_dim: Width of the hidden layers.
            depth: Number of hidden layers.
            key: JAX PRNG key.
            use_fourier: Whether to use RFF embedding.
            fourier_scale: Scale parameter for RFF frequencies.
        """
        k_emb, k_mlp = jax.random.split(key)

        if use_fourier:
            assert hidden_dim % 2 == 0, (
                f"hidden_dim must be even when use_fourier=True, got {hidden_dim}"
            )
            map_size = hidden_dim // 2
            self.embedding = RandomFourierFeatures(dim, map_size, fourier_scale, k_emb)
            in_size = map_size * 2
        else:
            self.embedding = None
            in_size = dim

        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=k_mlp,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the vector field at point x.

        Args:
            x: Point on the manifold, shape (D,).

        Returns:
            Vector value W(x), shape (D,).
            Operates on single points; use jax.vmap for batching.
        """
        if self.embedding is not None:
            x = self.embedding(x)
        return self.mlp(x)


class PSDMatrixField(eqx.Module):
    """Learns a Symmetric Positive Definite matrix field G(x).

    Reconstructs G(x) via a Cholesky-like factor A:
        G(x) = A(x) A(x)^T + ε I
    guaranteeing eigenvalues ≥ ε = PSD_EPS = 1e-4. The network outputs a
    flat vector of D² elements, reshaped to a D×D factor matrix A.

    Used to parameterize the Riemannian base metric h_{ij}(x) in Zermelo data
    (see MATH_SPEC § 5, ARCH_SPEC § 3.1).

    Note:
        The regularization constant ε = 1e-4 (PSD_EPS) is not configurable
        via constructor arguments. To use a different value, subclass and
        override __call__.
        use_fourier=False by default because metric fields are typically
        smoother than wind fields and benefit from direct coordinate input.

    See also:
        VectorField — counterpart for the wind field W^i(x).
        ham.utils.math.PSD_EPS — canonical epsilon constant.
    """

    embedding: Optional[RandomFourierFeatures]
    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int,
        key: jax.Array,
        use_fourier: bool = False,
    ):
        """Initializes the matrix field network.

        Args:
            dim: Dimension of the square matrix D.
            hidden_dim: Width of the hidden layers.
            depth: Number of hidden layers.
            key: JAX PRNG key.
            use_fourier: Whether to use RFF embedding (default False for smoother metrics).
        """
        k_emb, k_mlp = jax.random.split(key)
        self.dim = dim

        if use_fourier:
            # Output dim of RFF is 2 * map_size. Note: if hidden_dim is odd,
            # the effective embedding dimension will be hidden_dim - 1.
            map_size = hidden_dim // 2
            self.embedding = RandomFourierFeatures(dim, map_size, 1.0, k_emb)
            in_size = map_size * 2
        else:
            self.embedding = None
            in_size = dim

        # We output a full D*D matrix for factor A.
        # While Cholesky (D(D+1)/2) is more efficient, a full matrix A
        # simplifies implementation and ensures surjectivity over SPD matrices.
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim * dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.tanh,
            key=k_mlp,
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the SPD matrix at point x.

        Args:
            x: Point on the manifold, shape (D,).

        Returns:
            Symmetric positive-definite matrix G(x), shape (D, D).
            Operates on single points; use jax.vmap for batching.
        """
        if self.embedding is not None:
            x = self.embedding(x)

        flat_A = self.mlp(x)
        A = flat_A.reshape(self.dim, self.dim)

        # Construct G = A @ A.T + eps * I
        G = jnp.dot(A, A.T) + PSD_EPS * jnp.eye(self.dim)
        return G
