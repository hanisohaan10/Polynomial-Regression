import numpy as np
import matplotlib.pyplot as plt
import os

train_data = np.loadtxt("data/poly_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("data/poly_test.csv", delimiter=",", skiprows=1)
x, t = train_data[:, 0], train_data[:, 1]
x_test, t_test = test_data[:, 0], test_data[:, 1]

OUT = "notebooks/Outputs/Lasso"
os.makedirs(OUT, exist_ok=True)

d = 9
def Phi_legendre(x, d):
    u = 2*x - 1
    return np.polynomial.legendre.legvander(u, d)

Phi = Phi_legendre(x, d)
Phi_test = Phi_legendre(x_test, d)

def E_lasso(w, Phi, t, lam):
    r = Phi @ w - t
    return 0.5 * np.sum(r**2) + lam * np.sum(np.abs(w))

def rmse(Phi, w, t):
    pred = Phi @ w
    return np.sqrt(np.mean((pred - t)**2))

def lasso_subgrad(Phi, t, lam, eta=0.05, n_iter=20000):
    N, M = Phi.shape
    w = np.zeros(M)
    for it in range(n_iter):
        r = Phi @ w - t
        grad_smooth = Phi.T @ r
        g = grad_smooth + lam * np.sign(w)
        w = w - eta * g
    return w

# P3.2 RMSE sweep
lams = np.logspace(-4, 0, 21)
eta = 0.05
train_rmse = []
test_rmse = []
W_list = []
for lam in lams:
    w = lasso_subgrad(Phi, t, lam, eta=eta, n_iter=20000)
    W_list.append(w)
    train_rmse.append(rmse(Phi, w, t))
    test_rmse.append(rmse(Phi_test, w, t_test))

plt.figure(figsize=(6,4))
plt.semilogx(lams, train_rmse, marker='o', label='train RMSE')
plt.semilogx(lams, test_rmse, marker='s', label='test RMSE')
plt.axhline(np.sqrt(0.01), color='k', linestyle='--', label='noise sqrt(0.01)=0.1')
plt.xlabel('lambda'); plt.ylabel('RMSE'); plt.title('Lasso RMSE sweep'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,'lasso_rmse_sweep.png'))
plt.close()

# P3.3 coefficient paths
W = np.array(W_list)  # (n_lams, M)
plt.figure(figsize=(7,5))
for j in range(W.shape[1]):
    plt.semilogx(lams, W[:,j], label=f'w{j}')
plt.xlabel('lambda'); plt.ylabel('coefficient'); plt.title('Lasso coefficient paths'); plt.legend(fontsize=6, ncol=2)
plt.tight_layout()
plt.savefig(os.path.join(OUT,'lasso_coef_paths.png'))
plt.close()

# count ~0
thr = 2*eta*lams  # depends on lam
# compute per lam
zero_counts = []
for i, lam in enumerate(lams):
    w = W[i]
    thr_l = 2*eta*lam
    zero_counts.append(np.sum(np.abs(w) < thr_l))
print("zero counts:", list(zip(lams, zero_counts)))

# P3.4 best lambda
best_idx = np.argmin(test_rmse)
lam_star = lams[best_idx]
w_star_lasso = W[best_idx]
thr_star = 2*eta*lam_star
zero_idx = np.where(np.abs(w_star_lasso) < thr_star)[0]
print(f"lam*: {lam_star:.6f}, test RMSE*: {test_rmse[best_idx]:.6f}")
print(f"#~0 coeffs at lam*: {len(zero_idx)}, indices: {zero_idx.tolist()}")

# P3.5 Ridge in same Legendre basis
def ridge_legendre(Phi, t, lam):
    A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
    b = Phi.T @ t
    return np.linalg.solve(A,b)

ridge_train = []
ridge_test = []
W_ridge = []
for lam in lams:
    w = ridge_legendre(Phi, t, lam)
    W_ridge.append(w)
    ridge_train.append(rmse(Phi,w,t))
    ridge_test.append(rmse(Phi_test,w,t_test))
W_ridge = np.array(W_ridge)
best_r = np.argmin(ridge_test)
lam_star_r = lams[best_r]
w_star_ridge = W_ridge[best_r]
print(f"Ridge lam*: {lam_star_r:.6f}, test RMSE*: {ridge_test[best_r]:.6f}")

# table
def count_small(w, thresh=0.01):
    return np.sum(np.abs(w) < thresh)

print("\n=== Comparison at best ===")
print(f"Lasso (lam*={lam_star:.4g}): testRMSE={test_rmse[best_idx]:.6f}, ||w||2={np.linalg.norm(w_star_lasso):.6f}, |w|<0.01: {count_small(w_star_lasso)}")
print(f"Ridge (lam*={lam_star_r:.4g}): testRMSE={ridge_test[best_r]:.6f}, ||w||2={np.linalg.norm(w_star_ridge):.6f}, |w|<0.01: {count_small(w_star_ridge)}")
np.savetxt(os.path.join(OUT,'lasso_w_at_star.csv'), w_star_lasso)
np.savetxt(os.path.join(OUT,'ridge_w_at_star.csv'), w_star_ridge)
print("done")
