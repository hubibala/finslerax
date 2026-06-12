"""Learnable Finsler metric implementations.

Provides neural-network-parameterized and decoder-pullback metric classes that
inherit from the Riemannian and Randers base classes in ham.geometry.zoo. These
are the primary entry points for end-to-end geometry learning.

Classes:
    NeuralRiemannian -- Learnable Riemannian metric via PSDMatrixField.
    NeuralRanders    -- Learnable Randers metric via PSDMatrixField + VectorField.
    PullbackRanders  -- Decoder pullback metric (frozen H) with learned wind.
    PullbackRiemannian -- Decoder pullback metric (frozen H, no wind).
    DataDrivenPullbackRanders -- Pullback H with kernel-smoothed wind from data.
    KernelWindField  -- Non-parametric Nadaraya-Watson kernel smoother for latent velocities.
    PullbackGNet     -- Computes G = J^T J from a decoder Jacobian.

Note:
    spec/ARCH_SPEC.md § 3 refers to these collectively as 'LearnedFinsler'.
    The epsilon conventions used here:
    - 1e-5: Zermelo causality margin (||W||_h < 1 - 1e-5) in Randers subclasses.
    - PSD_EPS = 1e-4: metric tensor regularization in PullbackGNet.
    See also: spec/ARCH_SPEC.md § 3, § 5; spec/MATH_SPEC.md § 5.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
from typing import Any
from ..geometry.metric import FinslerMetric
from ..geometry.manifold import Manifold
from ..geometry.zoo import Randers, Riemannian
from ..nn.networks import VectorField, PSDMatrixField
from ..nn.ebm import ScalarEnergyField
from ..utils.math import PSD_EPS

class NeuralRiemannian(Riemannian):
    """A learnable Riemannian metric F(x,v) = sqrt(v^T G(x) v).

    G(x) is parameterized by a PSDMatrixField neural network, guaranteeing
    positive-definiteness. See also: spec/ARCH_SPEC.md § 3, spec/MATH_SPEC.md § 5.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array,
                 hidden_dim: int = 32, depth: int = 2):
        """Initializes the neural Riemannian metric.

        Args:
            manifold: The topological domain M.
            key: JAX PRNG key for network initialization.
            hidden_dim: Width of hidden layers in the PSDMatrixField. Default: 32.
            depth: Number of hidden layers. Default: 2.
        """
        self.dim = manifold.ambient_dim
        g_net = PSDMatrixField(self.dim, hidden_dim, depth, key)
        super().__init__(manifold, g_net)

class NeuralRanders(Randers):
    """A learnable Randers metric defined via Zermelo navigation.

    H(x) is a PSDMatrixField (the "sea" metric) and W(x) is a VectorField
    (the "wind"). The metric formula is from spec/MATH_SPEC.md § 5.
    The causality constraint ||W||_h < 1 is enforced by the parent class.

    See also:
        ham.geometry.zoo.Randers, ham.nn.networks.VectorField,
        ham.nn.networks.PSDMatrixField, examples/demo_learned_wind.py.
    """
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, key: jax.Array,
                 hidden_dim: int = 32, depth: int = 2,
                 use_fourier: bool = True, use_wind: bool = True):
        """Initializes the neural Randers metric.

        Args:
            manifold: The topological domain M.
            key: JAX PRNG key, split internally for the two sub-networks.
            hidden_dim: Width of hidden layers in both networks. Default: 32.
            depth: Number of hidden layers in both networks. Default: 2.
            use_fourier: If True, the wind network uses Random Fourier Features
                for high-frequency learning. Default: True. Note: Fourier scale
                for the wind network is hardcoded to 3.0.
        """
        k1, k2 = jax.random.split(key)
        self.dim = manifold.ambient_dim
        
        h_net = PSDMatrixField(self.dim, hidden_dim, depth, k1)
        w_net = VectorField(self.dim, hidden_dim, depth, k2, 
                            use_fourier=use_fourier, fourier_scale=3.0)
                            
        super().__init__(manifold, h_net, w_net, epsilon=1e-5, use_wind=use_wind)


class PullbackRanders(Randers):
    """A Randers metric where H(z) = J^T J is the pullback from the decoder.

    The Riemannian component H(z) is determined by the decoder's Jacobian
    (frozen, non-trainable). Only the wind W(z) is learned via a VectorField.

    Note:
        PullbackRanders(use_wind=False) is functionally equivalent to
        PullbackRiemannian. Use PullbackRiemannian when wind is never needed.
        The decoder should be frozen externally via eqx.partition.

    See also:
        PullbackGNet, PullbackRiemannian, examples/weinreb_smoke_test.py.
    """
    decoder: eqx.Module
    dim: int = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True, default=True)

    def __init__(self, manifold, decoder, key, hidden_dim=64, depth=3,
                 use_fourier=False, fourier_scale=1.0, use_wind=True):
        """Initializes the pullback Randers metric.

        Args:
            manifold: The latent-space manifold.
            decoder: A trained decoder network. Should be frozen (non-trainable)
                during metric learning via eqx.partition.
            key: JAX PRNG key for the wind VectorField initialization.
            hidden_dim: Width of hidden layers in the wind network. Default: 64.
            depth: Number of hidden layers in the wind network. Default: 3.
            use_fourier: If True, the wind network uses RFF embedding.
                Default: False.
            fourier_scale: Scale for RFF frequencies. Default: 1.0.
            use_wind: If False, the wind field is zeroed out, reducing this to
                a PullbackRiemannian metric. Default: True.
        """
        self.dim = int(manifold.ambient_dim)
        self.decoder = decoder
        self.use_wind = bool(use_wind)
        
        # We only need the VectorField for the Wind.
        w_net = VectorField(self.dim, hidden_dim, depth, key, 
                            use_fourier=use_fourier, fourier_scale=fourier_scale)
        
        # Proper eqx.Module for H(z)
        h_net = PullbackGNet(decoder=decoder, dim=self.dim)
        
        super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=self.use_wind)


class KernelWindField(eqx.Module):
    """Non-parametric Nadaraya-Watson kernel smoother for latent-space velocities.

    Computes a softmax-weighted average of anchor velocities:
        W(z) = sum_i w_i * v_i,  w_i = softmax(-||z - z_i||^2 / (2*sigma^2))

    This provides a data-driven, non-parametric wind field from observed
    pseudo-velocities projected into the latent space.
    """
    anchors_z: Any
    anchors_v: Any
    sigma: float = eqx.field(static=True)

    def __init__(self, anchors_z, anchors_v, sigma=0.5):
        """Initializes the kernel wind smoother.

        Args:
            anchors_z: Anchor latent points, shape (N, d).
            anchors_v: Velocity vectors at each anchor, shape (N, d).
            sigma: Bandwidth of the Gaussian kernel. Default: 0.5.
        """
        self.anchors_z = anchors_z
        self.anchors_v = anchors_v
        self.sigma = float(sigma)

    def __call__(self, z: jax.Array) -> jax.Array:
        """Returns the kernel-smoothed velocity at query point z.

        Args:
            z: Query latent point, shape (d,). Operates on single points;
               use jax.vmap for batched evaluation.

        Returns:
            Interpolated velocity, shape (d,).
        """
        # Optimized distance calculation for vmap scaling: (a-b)^2 = a^2 + b^2 - 2ab
        # Instead of (anchors - z)**2 which materializes a (B, N, D) array when vmapped over B queries.
        z_sq = jnp.sum(z**2, axis=-1, keepdims=True)
        anchors_sq = jnp.sum(self.anchors_z**2, axis=-1)
        dots = jnp.dot(self.anchors_z, z)
        
        dists_sq = z_sq + anchors_sq - 2 * dots
        # Clamp to avoid small negative values from floating-point cancellation
        dists_sq = jnp.maximum(dists_sq, 0.0)

        weights = jax.nn.softmax(-dists_sq / (2 * self.sigma**2))
        return jnp.dot(weights, self.anchors_v)

class DataDrivenPullbackRanders(Randers):
    """Pullback Randers metric with data-driven kernel-smoothed wind.

    H(z) = J^T J is the decoder pullback metric (frozen). W(z) is a
    Nadaraya-Watson kernel smoother over dataset RNA velocities projected
    into latent space. Primary metric for the Weinreb bio application
    (see spec/ARCH_SPEC.md § 6).

    See also:
        KernelWindField, PullbackGNet, examples/experiment_h2_directional.py.
    """
    decoder: Any
    dim: int = eqx.field(static=True)
    use_wind: bool = eqx.field(static=True, default=True)

    def __init__(self, manifold, decoder, anchors_z, anchors_v, sigma=0.5, use_wind=True):
        """Initializes the data-driven pullback Randers metric.

        Args:
            manifold: The latent-space manifold.
            decoder: A trained decoder network (frozen).
            anchors_z: Latent-space anchor points, shape (N, d).
            anchors_v: Velocity vectors at anchors (projected pseudo-velocities),
                shape (N, d).
            sigma: Kernel bandwidth. Default: 0.5.
            use_wind: If False, wind is zeroed out. Default: True.
        """
        self.dim = int(manifold.ambient_dim)
        self.decoder = decoder
        self.use_wind = bool(use_wind)
        h_net = PullbackGNet(decoder=decoder, dim=self.dim)
        w_net = KernelWindField(anchors_z, anchors_v, sigma)
        super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=self.use_wind)


class PullbackGNet(eqx.Module):
    """Pullback metric tensor from a decoder network.

    Computes G(z) = J(z)^T J(z) + PSD_EPS * I, where J(z) = d(decoder)/dz
    is the decoder Jacobian computed via jax.jacfwd. The diagonal term
    ensures positive-definiteness when the Jacobian is rank-deficient.

    Note:
        The decoder must be a pure function of z (no internal mutable state).
        For decoders with D >> d (typical in VAEs), jacfwd computes d forward
        passes and materialises the full (D, d) Jacobian in memory.

    Attributes:
        decoder: The decoder network f: R^d -> R^D.
        dim: Latent dimension d (used for the identity regularization).
    """
    decoder: Any
    dim: int = eqx.field(static=True)

    def __call__(self, z: jax.Array) -> jax.Array:
        """Computes the pullback metric tensor at latent point z.

        Args:
            z: Latent-space point, shape (d,).

        Returns:
            Positive-definite matrix G(z), shape (d, d).
        """
        J = jax.jacfwd(self.decoder)(z)
        H = jnp.dot(J.T, J)
        return H + PSD_EPS * jnp.eye(self.dim)

class PullbackRiemannian(Riemannian):
    """A Riemannian metric defined by the decoder's pullback G(z) = J^T J + eps I.

    No trainable parameters beyond the frozen decoder. For a version with a
    learned wind field, see PullbackRanders.

    Note:
        PullbackRanders(use_wind=False) is functionally equivalent to this
        class. Use this when wind is never needed (simpler API).
    """
    decoder: eqx.Module
    dim: int = eqx.field(static=True)

    def __init__(self, manifold: Manifold, decoder: eqx.Module):
        """Initializes the pullback Riemannian metric.

        Args:
            manifold: The latent-space manifold.
            decoder: A trained decoder network (frozen externally via
                eqx.partition before passing to an optimizer).
        """
        self.dim = manifold.ambient_dim
        self.decoder = decoder
        g_net = PullbackGNet(decoder=decoder, dim=self.dim)
        super().__init__(manifold, g_net)

class ConformalEnergyBase(eqx.Module):
    """A conformal base metric H(x) = c(x) I.
    
    The scaling factor c(x) increases with the scalar energy field E(x),
    ensuring that high-energy regions (mountains) cost more Riemannian 
    distance to traverse. This forces geodesics to curve around voids.
    """
    ebm: eqx.Module
    dim: int = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        # Use softplus for stability.
        energy = self.ebm(x)
        c = 1.0 + jax.nn.softplus(self.beta * energy)
        return c * jnp.eye(self.dim, dtype=x.dtype)

class EBMWindField(eqx.Module):
    """Wind field computed from the gradient of an Energy-Based Model."""
    ebm: eqx.Module
    scale: float = eqx.field(static=True)

    def __call__(self, x: jax.Array) -> jax.Array:
        # W(x) = -scale * \nabla E(x)
        grad_fn = jax.grad(self.ebm)
        return -self.scale * grad_fn(x)

class EnergyBasedRanders(Randers):
    """Energy-Based Randers Metric for Waddington's Landscape.

    H(x) = c(x) I (Conformal base, scaled by energy to bend geodesics)
    W(x) = -scale * \nabla E(x) (Negative gradient of biological potential)

    This metric formulates trajectory inference as a route down the biological
    energy landscape.
    """
    ebm: eqx.Module
    dim: int = eqx.field(static=True)
    wind_scale: float = eqx.field(static=True)

    def __init__(self, manifold: Manifold, ebm: eqx.Module, wind_scale: float = 1.0, beta: float = 1.0):
        """Initializes the Energy-Based Randers metric.

        Args:
            manifold: The topological domain M (usually flat Euclidean in PCA space).
            ebm: A trained ScalarEnergyField model.
            wind_scale: Scalar factor to multiply the energy gradient (lambda). Default: 1.0.
            beta: Conformal scaling factor for the base metric. Default: 1.0.
        """
        self.dim = manifold.ambient_dim
        self.ebm = ebm
        self.wind_scale = float(wind_scale)
        
        h_net = ConformalEnergyBase(ebm, self.dim, beta=float(beta))
        w_net = EBMWindField(ebm, self.wind_scale)
        
        super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=True)

class PseudotimeRanders(Randers):
    """Diffusion-Pseudotime (DPT) Randers Metric.
    
    H(x) = I (10D Euclidean Base metric)
    W(x) = \nabla DPT(x) (Gradient of the 1D pseudotime potential)
    """
    pseudotime_net: eqx.Module
    dim: int = eqx.field(static=True)
    wind_scale: float = eqx.field(static=True)
    
    def __init__(self, manifold: Manifold, pseudotime_net: eqx.Module, wind_scale: float = 1.0):
        self.dim = manifold.ambient_dim
        self.pseudotime_net = pseudotime_net
        self.wind_scale = float(wind_scale)
        
        # Base Metric H(x) = I
        class FlatHNet(eqx.Module):
            dim: int = eqx.field(static=True)
            def __call__(self, x: jax.Array) -> jax.Array:
                return jnp.eye(self.dim, dtype=x.dtype)
                
        h_net = FlatHNet(self.dim)
        
        # Wind W(x) = wind_scale * \nabla DPT(x)
        class DPTWindField(eqx.Module):
            net: eqx.Module
            scale: float = eqx.field(static=True)
            def __call__(self, x: jax.Array) -> jax.Array:
                grad_fn = jax.grad(self.net)
                return self.scale * grad_fn(x)
                
        w_net = DPTWindField(self.pseudotime_net, self.wind_scale)
        
        super().__init__(manifold, h_net=h_net, w_net=w_net, epsilon=1e-5, use_wind=True)