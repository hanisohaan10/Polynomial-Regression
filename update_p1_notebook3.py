import json

with open('notebooks/closed_vs_gd.ipynb', encoding='utf-8') as f:
    nb = json.load(f)

new_cells = []

def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.split('\n')
    }

new_cells.append(md_cell('# P1: Polynomial Regression — Closed Form vs Gradient Descent Family\n\n**Objective:** Implement polynomial regression both via the closed-form solution and via the full gradient descent family (Batch GD, Stochastic GD, Mini-batch GD) from scratch, verify that all methods agree, and study how the choice of learning rate governs convergence.\n\n**Dataset:** `poly_train.csv` with N=10 points $(x_i,y_i)$, $x_i$ evenly spaced in $[0,1]$. Target from noisy sine wave. Load directly. True function $\sin(2\pi x)$ used only for plotting.'))
new_cells.append(md_cell('## 1.1 Polynomial Model and Objective\n\n**Model:** $\\hat{y} = \\sum_{j=0}^{d} w_j x^j = \\phi(x)^T \\mathbf{w}$, $\\phi(x)^T=[1,x,...,x^d]$. $\\Phi \\in \\mathbb{R}^{N\\times(d+1)}$ has rows $\\phi(x_n)^T$.\n\n**Objective:** $E(\\mathbf{w}) = \\frac{1}{2}\\|\\Phi \\mathbf{w} - \\mathbf{t}\\|^2$. **Gradient:** $\\nabla E(\\mathbf{w}) = \\Phi^T (\\Phi \\mathbf{w} - \\mathbf{t})$.'))
new_cells.append(md_cell('## 1.2 Closed Form\n\nNormal equations: $\\Phi^T \\Phi\\, \\mathbf{w} = \\Phi^T \\mathbf{t}$. For $d=9,N=10$, $\\Phi$ is ill-conditioned ($\\operatorname{cond}(\\Phi)\\approx 10^7$), so we use `np.linalg.lstsq` (SVD-based) rather than explicit matrix inverse.'))
new_cells.append(md_cell('## 1.3 GD Family\n\n**One epoch** = one pass over all N points.\n\n- **Batch GD:** $\\mathbf{w}^{(k+1)} \\leftarrow \\mathbf{w}^{(k)} - \\eta \\Phi^T(\\Phi \\mathbf{w}^{(k)} - \\mathbf{t})$ (1 update/epoch)\n- **SGD:** shuffle per epoch; for each $n$: $\\mathbf{w}^{(k+1)} \\leftarrow \\mathbf{w}^{(k)} - \\eta (\\phi(x_n)^T \\mathbf{w}^{(k)} - t_n)\\phi(x_n)$ (N updates)\n- **Mini-batch GD:** shuffle per epoch, batch size 3 -> batches (3,3,3,1), 4 updates/epoch. Use mean gradient over batch.\n\nSeed fixed via `np.random.default_rng(seed)`.'))
new_cells.append(md_cell('## 1.4 QR Reparametrisation for d=9\n\nSince GD in monomial basis is slow for ill-conditioned $\\Phi$: do thin QR $\\Phi=QR$ ($Q^TQ=I$). Let $\\boldsymbol{\\alpha}=R\\mathbf{w}$, $E(\\boldsymbol{\\alpha})=\\frac{1}{2}\\|Q\\boldsymbol{\\alpha}-\\mathbf{t}\\|^2$, $\\nabla_\\alpha E = Q^T(Q\\boldsymbol{\\alpha}-\\mathbf{t})$. Run all three GD variants in $\\alpha$-space from $\\boldsymbol{\\alpha}=0$, then recover $\\mathbf{w}=R^{-1}\\boldsymbol{\\alpha}$.'))
new_cells.append(md_cell('## 1.5 Deliverables (Implementation Plan)\n\n1. Fit d=1..9 by closed form, plot curves vs training points.  \n2. For d=9 do Batch/SGD/Mini-batch in QR space.  \n3. Loss curves $E$ vs epoch (log y) for all three with same $\eta$, run until converged.  \n4. Batch GD learning rate study: too small ($\\eta=0.01$), well-chosen ($\\eta\\in[0.1,1.0]$), too large ($\\eta>2$, e.g. 2.1) - report $\\mathbf{w},E$ after 100 epochs for large $\eta$.  \n5. Verification: after convergence, report $\|\\mathbf{w}_{GD}-\\mathbf{w}_{closed}\\|_2$ and relative error $\|\\mathbf{w}_{GD}-\\mathbf{w}_{closed}\\|_2/\\|\\mathbf{w}_{closed}\\|_2$.  \n6. Discussion: compare curves, explain why SGD resembles Batch (unbiased stochastic gradient), why Mini-batch slower than both in some cases, relate to rows/structure of $Q$.'))

new_cells.extend(nb['cells'])
nb['cells'] = new_cells
with open('notebooks/closed_vs_gd.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print('done')
