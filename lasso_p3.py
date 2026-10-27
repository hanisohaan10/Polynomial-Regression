import numpy as np
import matplotlib.pyplot as plt
import os

# Data
train_data = np.loadtxt("data/poly_train.csv", delimiter=",", skiprows=1)
test_data = np.loadtxt("data/poly_test.csv", delimiter=",", skiprows=1)
x, t = train_data[:, 0], train_data[:, 1]
x_test, t_test = test_data[:, 0], test_data[:, 1]

OUT = "notebooks/Outputs/Lasso"
os.makedirs(OUT, exist_ok=True)

# Legendre basis
d = 9
def Phi_legendre(x, d):
    u = 2*x - 1
    return np.polynomial.legendre.legvander(u, d)

Phi = Phi_legendre(x, d)
Phi_test = Phi_legendre(x_test, d)

def E_lasso(w, Phi, t, lam):
    r = Phi @ w - t
    return 0.5 * np.sum(r**2) + lam * np.sum(np.abs(w))

def lasso_subgrad(Phi, t, lam, eta=0.05, n_iter=20000):
    N, M = Phi.shape
    w = np.zeros(M)
    E_hist = []
    E_best = np.inf
    w_best = w.copy()
    for it in range(n_iter):
        r = Phi @ w - t
        grad_smooth = Phi.T @ r
        g = grad_smooth + lam * np.sign(w)
        w = w - eta * g
        E = E_lasso(w, Phi, t, lam)
        E_hist.append(E)
        if E < E_best:
            E_best = E
            w_best = w.copy()
    return w, np.array(E_hist), E_best, w_best

# P3.1
lam_v = 0.1
eta = 0.05
w_v, E_hist_v, E_best_v, w_best_v = lasso_subgrad(Phi, t, lam_v, eta=eta, n_iter=20000)
w_vf = w_v
r_check = t - Phi @ w_vf
PhiT_r = Phi.T @ r_check
thr = 2*eta*lam_v
abs_w = np.abs(w_vf)
is_zero = abs_w < thr
max_nonzero = 0
for j in range(len(w_vf)):
    if not is_zero[j] and abs(w_vf[j]) > 1e-18:
        val = PhiT_r[j] - lam_v * np.sign(w_vf[j])
        max_nonzero = max(max_nonzero, abs(val))
max_zero = 0
for j in range(len(w_vf)):
    if is_zero[j]:
        max_zero = max(max_zero, abs(PhiT_r[j]) / lam_v)
print("=== P3.1 ===")
print(f"best E: {E_best_v:.6f} (ref ~0.2625)")
print(f"max |Phi^T r - lam sign(w)| nonzero: {max_nonzero:.6e}")
print(f"max |Phi^T r|/lam zero: {max_zero:.6f}")
w_star = np.array([0, -0.928, 0.045, 1.117, 0.032, -0.201, 0, -0.026, 0, 0])
print(f"||w - w*||2: {np.linalg.norm(w_vf - w_star):.6f}")

best_so_far = np.minimum.accumulate(E_hist_v)
plt.figure(figsize=(6,4))
plt.semilogy(E_hist_v, alpha=0.5, label='E(w)')
plt.semilogy(best_so_far, label='best-so-far E')
plt.xlabel('iteration'); plt.ylabel('E(w)'); plt.title('Lasso conv (lambda=0.1)'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,'lasso_conv_lambda01.png'))
plt.close()
print("saved lasso_conv_lambda01.png")
