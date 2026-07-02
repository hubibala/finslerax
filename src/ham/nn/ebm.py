"""Energy-Based Model (EBM) neural networks.

Provides the ScalarEnergyField, which maps states in the ambient space to a
scalar energy value E(x); the induced density is ``p(x) \\propto exp(-E(x))``.
"""

from typing import Optional

import equinox as eqx
import jax
import jax.numpy as jnp

from .networks import RandomFourierFeatures


class QuadraticHead(eqx.Module):
    """Quadratic output head for EBMs.

    Ensures the energy function is bounded from below and structurally
    confining (E(x) -> infinity as ||x|| -> infinity).
    Calculates: w1(x) * w2(x) + w3(x^2).
    """

    w1: eqx.nn.Linear
    w2: eqx.nn.Linear
    w3: eqx.nn.Linear

    def __init__(self, in_features: int, key: jax.Array):
        k1, k2, k3 = jax.random.split(key, 3)
        self.w1 = eqx.nn.Linear(in_features, 1, key=k1)
        self.w2 = eqx.nn.Linear(in_features, 1, key=k2)
        self.w3 = eqx.nn.Linear(in_features, 1, key=k3)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        return self.w1(x) * self.w2(x) + self.w3(x**2)


class ScalarEnergyField(eqx.Module):
    r"""Neural-network approximation of a scalar energy field E(x): R^D -> R.

    Used for parameterizing Energy-Based Models (EBMs); the induced data
    distribution is p(x) \propto exp(-E(x)).

    Uses SiLU (Swish) activation for smooth, non-monotonic gradients.
    """

    embedding: Optional[RandomFourierFeatures]
    mlp: eqx.nn.MLP
    energy_out: eqx.nn.Linear

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        depth: int,
        key: jax.Array,
        use_fourier: bool = False,
        fourier_scale: float = 1.0,
    ):
        """Initializes the energy field network.

        Args:
            dim: Input dimensionality D.
            hidden_dim: Width of the hidden layers.
            depth: Number of hidden layers.
            key: JAX PRNG key.
            use_fourier: Whether to use RFF embedding. Often False for smoother
                global energy landscapes, but useful for high-frequency data.
            fourier_scale: Scale parameter for RFF frequencies.
        """
        k_emb, k_mlp, k_head = jax.random.split(key, 3)

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

        # The MLP outputs a feature vector, not a scalar directly
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=hidden_dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.silu,
            key=k_mlp,
        )
        self.energy_out = eqx.nn.Linear(hidden_dim, 1, key=k_head)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """Evaluates the energy field at point x.

        Args:
            x: Point in the state space, shape (D,).

        Returns:
            Scalar energy value E(x), shape ().
            Operates on single points; use jax.vmap for batching.
        """
        if self.embedding is not None:
            x_in = self.embedding(x)
        else:
            x_in = x
        features = self.mlp(x_in)
        base_energy = jnp.squeeze(self.energy_out(features), axis=-1)
        return base_energy
