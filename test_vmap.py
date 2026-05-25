import jax
import jax.numpy as jnp

def compute_loss(G):
    return jnp.sum(G ** 2)

def fd_gradient(G, c, i, j, eps):
    G_plus = G.at[c, i, j].add(eps)
    G_minus = G.at[c, i, j].add(-eps)
    return (compute_loss(G_plus) - compute_loss(G_minus)) / (2 * eps)

G = jnp.zeros((3, 10, 10))
pts = jnp.array([[0, 1, 1], [2, 5, 5]])

# Vmap over the points (axis 0 of pts)
vmap_fd = jax.vmap(fd_gradient, in_axes=(None, 0, 0, 0, None))
grads = vmap_fd(G, pts[:, 0], pts[:, 1], pts[:, 2], 1e-4)
print(grads)
