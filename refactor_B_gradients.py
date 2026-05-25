import re

with open('experiments/wildfire/synthetic/exp_B_gradients.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Replace finite_difference_gradient definition with batched ones
old_fd_def = """def finite_difference_gradient(G, B, source_mask, dL_dT, param, indices, eps=1e-5, hx=1.0, hy=1.0, max_iters=50, tol=1e-4):
    \"\"\"Compute gradient via central finite differences.\"\"\"
    c, i, j = indices
    
    if param == 'G':
        G_plus = G.at[c, i, j].add(eps)
        G_minus = G.at[c, i, j].add(-eps)
        L_plus = compute_loss_jitted(G_plus, B, source_mask, dL_dT, hx, hy, max_iters, tol)
        L_minus = compute_loss_jitted(G_minus, B, source_mask, dL_dT, hx, hy, max_iters, tol)
    else:
        B_plus = B.at[c, i, j].add(eps)
        B_minus = B.at[c, i, j].add(-eps)
        L_plus = compute_loss_jitted(G, B_plus, source_mask, dL_dT, hx, hy, max_iters, tol)
        L_minus = compute_loss_jitted(G, B_minus, source_mask, dL_dT, hx, hy, max_iters, tol)
        
    return float((L_plus - L_minus) / (2 * eps))"""

new_fd_def = """@functools.partial(jax.jit, static_argnames=('hx', 'hy', 'max_iters', 'tol'))
def batched_fd_G(G, B, source_mask, dL_dT, pts_c, pts_i, pts_j, eps, hx, hy, max_iters, tol):
    def single_fd(c, i, j):
        G_plus = G.at[c, i, j].add(eps)
        G_minus = G.at[c, i, j].add(-eps)
        L_plus = compute_loss_jitted(G_plus, B, source_mask, dL_dT, hx, hy, max_iters, tol)
        L_minus = compute_loss_jitted(G_minus, B, source_mask, dL_dT, hx, hy, max_iters, tol)
        return (L_plus - L_minus) / (2 * eps)
    return jax.vmap(single_fd)(pts_c, pts_i, pts_j)

@functools.partial(jax.jit, static_argnames=('hx', 'hy', 'max_iters', 'tol'))
def batched_fd_B(G, B, source_mask, dL_dT, pts_c, pts_i, pts_j, eps, hx, hy, max_iters, tol):
    def single_fd(c, i, j):
        B_plus = B.at[c, i, j].add(eps)
        B_minus = B.at[c, i, j].add(-eps)
        L_plus = compute_loss_jitted(G, B_plus, source_mask, dL_dT, hx, hy, max_iters, tol)
        L_minus = compute_loss_jitted(G, B_minus, source_mask, dL_dT, hx, hy, max_iters, tol)
        return (L_plus - L_minus) / (2 * eps)
    return jax.vmap(single_fd)(pts_c, pts_i, pts_j)

def evaluate_fd_gradients(G, B, source_mask, dL_dT, test_points, eps, hx=1.0, hy=1.0, max_iters=50, tol=1e-4):
    pts_G_c, pts_G_i, pts_G_j = [], [], []
    for i, j in test_points:
        for c in [0, 2]:
            pts_G_c.append(c); pts_G_i.append(i); pts_G_j.append(j)
            
    pts_B_c, pts_B_i, pts_B_j = [], [], []
    for i, j in test_points:
        for c in [0, 1]:
            pts_B_c.append(c); pts_B_i.append(i); pts_B_j.append(j)
            
    fd_G = batched_fd_G(G, B, source_mask, dL_dT, jnp.array(pts_G_c), jnp.array(pts_G_i), jnp.array(pts_G_j), eps, hx, hy, max_iters, tol)
    fd_B = batched_fd_B(G, B, source_mask, dL_dT, jnp.array(pts_B_c), jnp.array(pts_B_i), jnp.array(pts_B_j), eps, hx, hy, max_iters, tol)
    
    return np.array(fd_G), np.array(fd_B)"""

content = content.replace(old_fd_def, new_fd_def)

# 2. Replace the loop in each experiment
old_loop_pattern = re.compile(r"""        for i, j in test_points:
            for c in \[0, 2\]:
                fd = finite_difference_gradient\(G, B, source_mask, dL_dT, 'G', \(c, i, j\), self\.eps\)
                impl = float\(dL_dG\[c, i, j\]\)
                rel_err = abs\(fd - impl\) / abs\(fd\) if abs\(fd\) > 1e-10 else abs\(impl\)
                results_G\.append\(\{'fd': fd, 'impl': impl, 'rel_err': rel_err\}\)
            
            for c in \[0, 1\]:
                fd = finite_difference_gradient\(G, B, source_mask, dL_dT, 'B', \(c, i, j\), self\.eps\)
                impl = float\(dL_dB\[c, i, j\]\)
                rel_err = abs\(fd - impl\) / abs\(fd\) if abs\(fd\) > 1e-10 else abs\(impl\)
                results_B\.append\(\{'fd': fd, 'impl': impl, 'rel_err': rel_err\}\)""")

new_loop_code = """        fd_G_all, fd_B_all = evaluate_fd_gradients(G, B, source_mask, dL_dT, test_points, self.eps)
        
        idx_G, idx_B = 0, 0
        for i, j in test_points:
            for c in [0, 2]:
                fd = float(fd_G_all[idx_G])
                idx_G += 1
                impl = float(dL_dG[c, i, j])
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results_G.append({'fd': fd, 'impl': impl, 'rel_err': rel_err})
            
            for c in [0, 1]:
                fd = float(fd_B_all[idx_B])
                idx_B += 1
                impl = float(dL_dB[c, i, j])
                rel_err = abs(fd - impl) / abs(fd) if abs(fd) > 1e-10 else abs(impl)
                results_B.append({'fd': fd, 'impl': impl, 'rel_err': rel_err})"""

content = old_loop_pattern.sub(new_loop_code, content)

with open('experiments/wildfire/synthetic/exp_B_gradients.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("Refactored exp_B_gradients.py!")
