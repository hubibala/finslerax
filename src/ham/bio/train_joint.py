import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from ham.geometry import Sphere
from ham.models.learned import NeuralRanders
from ham.bio.vae import GeometricVAE

def train_learned_vae(data_x, epochs=1000):
    key = jax.random.PRNGKey(2025)
    k_model, k_metric = jax.random.split(key)

    # 1. Initialize the Learnable Geometry
    manifold = Sphere(radius=1.0) # We still fix the topology (S^2)
    
    # This metric has weights! (h_net, w_net)
    neural_metric = NeuralRanders(
        manifold=manifold, 
        key=k_metric,
        hidden_dim=32,
        use_fourier=True
    )

    # 2. Initialize the VAE with the Learnable Metric
    vae = GeometricVAE(
        data_dim=data_x.shape[1],
        latent_dim=3,
        metric=neural_metric,
        key=k_model
    )

    # 3. Optimizer
    # We train ALL parameters (VAE + Metric) together
    optimizer = optax.adam(learning_rate=1e-3)
    opt_state = optimizer.init(eqx.filter(vae, eqx.is_array))

    @eqx.filter_jit
    def train_step(model, x, state, k):
        def loss_fn(m):
            losses, stats = jax.vmap(m.loss_fn, in_axes=(0, None))(x, k)
            return jnp.mean(losses), stats
            
        (loss, (recon, action, reg)), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(model)
        updates, new_state = optimizer.update(grads, state, model)
        new_model = eqx.apply_updates(model, updates)
        return new_model, new_state, loss, recon, action, reg

    # 4. Loop
    print("Starting Joint Training (Geometry + VAE)...")
    for i in range(epochs):
        step_key = jax.random.fold_in(key, i)
        vae, opt_state, loss, r, a, reg = train_step(vae, data_x, opt_state, step_key)
        
        if i % 100 == 0:
            # Watch 'reg' - if it explodes, the metric is unstable.
            # Watch 'action' - it should decrease as the metric learns the flow.
            print(f"Epoch {i}: Loss={loss:.4f} | Recon={r:.4f} | Action={a:.4f} | Reg={reg:.4f}")
            
    return vae