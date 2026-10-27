import json

with open('notebooks/closed_vs_gd.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(True)  # keep newlines
    }

# Theory
new_cells.append(md_cell('# P1: Polynomial Regression — Closed Form vs Gradient Descent Family\n\n**Objective:** Implement polynomial regression both via the closed-form solution and via the full gradient descent family (Batch GD, Stochastic GD, Mini-batch GD) from scratch, verify that all methods agree, and study how the choice of learning rate governs convergence.\n\n**Dataset:** `poly_train.csv` with N=10 points $(x_i,y_i)$, $x_i$ evenly spaced in $[0,1]$. Target generated from noisy sine wave. Load directly, do not generate data. True function $\sin(2\pi x)$ may be used only for plotting.'))
new_cells.append(md_cell('## 1.1 Polynomial Model and Objective\n\n**Model:**\n$$\n\\hat{y} = w_0 + w_1 x + w_2 x^2 + \\cdots + w_d x^d = \\sum_{j=0}^{d} w_j x^j = \\phi(x)^T \\mathbf{w}\n$$\nwhere $\\phi(x)^T = [1, x, x^2, \\ldots, x^d]$. Design matrix $\\Phi \\in \\mathbb{R}^{N\\times(d+1)}$ has $n$-th row $\\phi(x_n)^T$.\n\n**Objective (SSE):**\n$$\nE(\\mathbf{w}) = \\frac{1}{2}\\sum_{n=1}^{N} (y(x_n,\\mathbf{w}) - t_n)^2 = \\frac{1}{2}\\|\\Phi \\mathbf{w} - \\mathbf{t}\\|^2\n$$\n\n**Gradient:**\n$$\n\\nabla E(\\mathbf{w}) = \\Phi^T (\\Phi \\mathbf{w} - \\mathbf{t})\n$$'))
new_cells.append(md_cell('## 1.2 Closed Form Solution\n\nSetting gradient to zero gives **normal equations:**\n$$\n\\Phi^T \\Phi\\, \\mathbf{w} = \\Phi^T \\mathbf{t}\n$$\n\n**Numerical stability (d=9, N=10):** $\\Phi$ is a Vandermonde matrix and is severely ill-conditioned ($\\operatorname{cond}(\\Phi) \\approx 10^7$). Computing $(\\Phi^T\\Phi)^{-1}$ explicitly is unstable. We use a robust **SVD-based** solver (`np.linalg.lstsq`) to solve directly.'))
new_cells.append(md_cell('## 1.3 Gradient Descent Family\n\nWe compare three variants. **One epoch** = one full pass over all N points.\n\n**Batch GD:** Uses full dataset in one update.\n$$\n\\mathbf{w}^{(k+1)} \\leftarrow \\mathbf{w}^{(k)} - \\eta\\, \\nabla E(\\mathbf{w})\\Big|_{\\mathbf{w}^{(k)}}, \\quad \\nabla E(\\mathbf{w}) = \\Phi^T(\\Phi \\mathbf{w} - \\mathbf{t})\n$$\n(1 update/epoch)\n\n**Stochastic GD (SGD):** Shuffle data per epoch, for each point $n$ do update using only that point.\n$$\n\\mathbf{w}^{(k+1)} \\leftarrow \\mathbf{w}^{(k)} - \\eta\\, (y(x_n,\\mathbf{w}^{(k)}) - t_n)\\, \\phi(x_n)\n$$\n(N updates/epoch)\n\n**Mini-batch GD:** Shuffle per epoch, update using average gradient over mini-batch (size = 3). Since N=10, batches of sizes 3,3,3,1 giving 4 updates/epoch.\n\n**Random seed:** We fix `np.random.default_rng(seed)` for per-epoch shuffling (seed stated in code).'))
new_cells.append(md_cell('## 1.4 QR Reparametrisation (for d=9)\n\nFor $d=9, N=10$, GD in original monomial basis converges extremely slowly due to ill-conditioning. To make optimisation tractable without changing the problem:\n\n1. Compute thin QR decomposition: $\\Phi = QR$, with $Q$ having orthonormal columns ($Q^T Q = I$), $R$ upper triangular.\n2. Define $\\boldsymbol{\\alpha} = R\\mathbf{w}$, so $\\Phi \\mathbf{w} = QR\\mathbf{w} = Q\\boldsymbol{\\alpha}$.\n3. Objective becomes well-conditioned: $E(\\boldsymbol{\\alpha}) = \\frac{1}{2}\\|Q\\boldsymbol{\\alpha} - \\mathbf{t}\\|^2$, gradient $\\nabla_\\alpha E(\\boldsymbol{\\alpha}) = Q^T (Q\\boldsymbol{\\alpha} - \\mathbf{t})$.\n4. Run Batch/SGD/Mini-batch GD in $\\alpha$-space starting from $\\boldsymbol{\\alpha}=0$ with same update rules (using $Q$).\n5. Recover original weights: $\\mathbf{w} = R^{-1}\\boldsymbol{\\alpha}$ (solve upper triangular system).'))

new_cells.extend(nb.cells)
with open('notebooks/closed_vs_gd.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('done')
