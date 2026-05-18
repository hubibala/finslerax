import jax
import jax.numpy as jnp
import equinox as eqx

class SimpleMLP(eqx.Module):
    linear: eqx.nn.Linear
    activation: jax.nn.tanh

    def __init__(self, key):
        self.linear = eqx.nn.Linear(2, 1, key=key)
        self.activation = jax.nn.tanh

    def __call__(self, x):
        return self.activation(self.linear(x))[0]

def test_scan_grad():
    key = jax.random.PRNGKey(0)
    model = SimpleMLP(key)
    xs = jnp.ones((4, 2))  # Batch size = 4
    ys = jnp.array([2.0, 2.0, 2.0, 2.0])

    @eqx.filter_jit
    def compute_loss_and_grads(m, x_batch, y_batch):
        def total_loss(model_in):
            def scan_fn(carry, inputs):
                x_i, y_i = inputs
                pred = model_in(x_i)
                loss_val = (pred - y_i) ** 2
                return carry, loss_val

            _, per_element_loss = jax.lax.scan(
                scan_fn,
                None,
                (x_batch, y_batch)
            )
            return jnp.mean(per_element_loss)

        return eqx.filter_value_and_grad(total_loss)(m)

    loss, grads = compute_loss_and_grads(model, xs, ys)
    print("Loss:", loss)
    print("Weight grads:", grads.linear.weight)
    print("Bias grads:", grads.linear.bias)

if __name__ == "__main__":
    test_scan_grad()
