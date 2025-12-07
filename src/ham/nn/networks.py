import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Optional

class RandomFourierFeatures(eqx.Module):
    """
    Maps input x to high-dimensional frequency space: [cos(Bx), sin(Bx)].
    Mitigates spectral bias in coordinate-based MLPs.
    """
    B: jnp.ndarray
    
    def __init__(self, in_dim: int, mapping_size: int, scale: float, key: jax.Array):
        # B is sampled from N(0, scale^2)
        self.B = jax.random.normal(key, (mapping_size, in_dim)) * scale

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        # x: (D,) -> Bx: (M,)
        projected = jnp.dot(self.B, x)
        return jnp.concatenate([jnp.cos(projected), jnp.sin(projected)], axis=0)

class VectorField(eqx.Module):
    """
    Learns a vector field W(x): R^D -> R^D.
    """
    embedding: Optional[RandomFourierFeatures]
    mlp: eqx.nn.MLP

    def __init__(self, dim: int, hidden_dim: int, depth: int, key: jax.Array, 
                 use_fourier: bool = True, fourier_scale: float = 10.0):
        
        k_emb, k_mlp = jax.random.split(key)
        
        if use_fourier:
            # Mapping size usually hidden_dim // 2 so output is hidden_dim
            map_size = hidden_dim // 2
            self.embedding = RandomFourierFeatures(dim, map_size, fourier_scale, k_emb)
            in_size = map_size * 2 # cos + sin
        else:
            self.embedding = None
            in_size = dim
            
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.gelu, # smoother than relu for gradients
            key=k_mlp
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.embedding is not None:
            x = self.embedding(x)
        return self.mlp(x)

class PSDMatrixField(eqx.Module):
    """
    Learns a Symmetric Positive Definite matrix field G(x).
    Method:
        1. Network outputs matrix A of shape (D, D).
        2. G = A @ A.T + epsilon * I
    """
    embedding: Optional[RandomFourierFeatures]
    mlp: eqx.nn.MLP
    dim: int = eqx.field(static=True)

    def __init__(self, dim: int, hidden_dim: int, depth: int, key: jax.Array, 
                 use_fourier: bool = False): # Metric usually smoother than wind
        
        k_emb, k_mlp = jax.random.split(key)
        self.dim = dim
        
        if use_fourier:
            map_size = hidden_dim // 2
            self.embedding = RandomFourierFeatures(dim, map_size, 1.0, k_emb)
            in_size = map_size * 2
        else:
            self.embedding = None
            in_size = dim
            
        # Output flattened D*D matrix
        self.mlp = eqx.nn.MLP(
            in_size=in_size,
            out_size=dim * dim,
            width_size=hidden_dim,
            depth=depth,
            activation=jax.nn.gelu,
            key=k_mlp
        )

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        if self.embedding is not None:
            x = self.embedding(x)
            
        # Predict Factor A
        flat_A = self.mlp(x)
        A = flat_A.reshape(self.dim, self.dim)
        
        # Construct PSD matrix
        # G = A A^T + eps I
        G = jnp.dot(A, A.T) + 1e-4 * jnp.eye(self.dim)
        
        return G