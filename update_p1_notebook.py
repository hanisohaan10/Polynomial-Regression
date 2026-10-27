import json
import nbformat as nbf

nb = nbf.read('notebooks/closed_vs_gd.ipynb', as_version=4)

# Create new cells with theory
new_cells = []

# Title
new_cells.append(nbf.new_markdown_cell('# P1: Polynomial Regression — Closed Form vs Gradient Descent Family\n\n**Objective:** Implement polynomial regression via closed form and full GD family (Batch GD, Stochastic GD, Mini-batch GD) from scratch, verify agreement and study effect of learning rate.\n\n**Dataset:** `poly_train.csv` (N=10). True function $\\sin(2\\pi x)$ used only for plotting, not fitting.'))

# Theory 1
new_cells.append(nbf.new_markdown_cell('## 1.1 Model and Objective\n\n**Polynomial model:**\n$$\ny \\approx \\hat{y} = w_0 + w_1 x + w_2 x^2 + \\cdots + w_d x^d = \\sum_{j=0}^{d} w_j x^j = \\mathbf{w}^T \\phi(x)\n$$\n\n**Design matrix** $\\Phi \\in \\mathbb{R}^{N\\times (d+1)}$ with rows $\\phi(x_n)^T = [1, x_n, x_n^2, ..., x_n^d]$.\n\n**Sum of Squared Errors (SSE):**\n$$\nE(\\mathbf{w}) = \\frac{1}{2}\\sum_{n=1}^{N} (y(x_n,\\mathbf{w}) - t_n)^2 = \\frac{1}{2}\\|\\Phi \\mathbf{w} - \\mathbf{t}\\|^2\n$$\n\n**Gradient:**\n$$\n\\nabla E(\\mathbf{w}) = \\Phi^T (\\Phi \\mathbf{w} - \\mathbf{t})\n$$'))

# Theory 2
new_cells.append(nbf.new_markdown_cell('## 1.2 Closed Form Solution\n\nSetting $\\nabla E(\\mathbf{w}) = 0$ gives the **normal equations:**\n$$\n\\Phi^T \\Phi\\, \\mathbf{w} = \\Phi^T \\mathbf{t}\n$$\n\nFor general case $\\mathbf{w} = (\\Phi^T\\Phi)^{-1}\\Phi^T \\mathbf{t}$.\n\n**Note on numerical stability (d=9, N=10):** For $d=9$, $N=10$, the Vandermonde matrix $\\Phi$ is severely ill-conditioned ($\\operatorname{cond}(\\Phi) \\approx 10^7$). Forming $(\\Phi^T\\Phi)^{-1}$ explicitly is unstable. Hence we use **SVD-based solver** `np.linalg.lstsq(Phi, t, rcond=None)` to compute weights robustly.'))

# Add existing cells (imports onwards)
new_cells.extend(nb.cells)

nb.cells = new_cells
nbf.write(nb, 'notebooks/closed_vs_gd.ipynb')
print('done')
