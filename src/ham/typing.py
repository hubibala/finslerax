"""Type hints and protocols for the HAM framework."""

from typing import Any, Protocol

import jax


class Distribution(Protocol):
    def sample(self, seed: jax.Array) -> jax.Array: ...
    def kl_divergence_std_normal(self) -> jax.Array: ...

class GenerativeModel(Protocol):
    """Protocol for models that can encode to and decode from a latent manifold."""
    manifold: Any
    metric: Any

    def encode(self, x: jax.Array, key: jax.Array) -> jax.Array: ...
    def decode(self, z: jax.Array) -> jax.Array: ...
    def _get_dist(self, x: jax.Array) -> Distribution: ...
    def project_control(self, x: jax.Array, v: jax.Array) -> tuple[jax.Array, jax.Array]: ...

class EnergyModel(Protocol):
    """Protocol for models that map an input array to a scalar energy."""
    def __call__(self, x: jax.Array) -> jax.Array: ...
