import numpy as np
import matplotlib.pyplot as plt
import os

OUT = "notebooks/Outputs/Bayesian"
os.makedirs(OUT, exist_ok=True)

# fixed hyperparams
sigma2 = 0.01
beta = 1.0/sigma2  # 100
alpha = 0.05
lam_map = alpha/beta  # 0.0005

d = 3
def phi(x):
    # [1, x, x^2, x^3]
    x = np.atleast_1d(x)
    Phi = np.column_stack([x**i for i in range(d+1)])
    return Phi

# load N10
train10 = np.loadtxt("data/p4_train_N10.csv", delimiter=",", skiprows=1)
x10, t10 = train10[:,0], train10[:,1]
Phi10 = phi(x10)

# posterior
def bayes_posterior(Phi, t, alpha, beta):
    M = Phi.shape[1]
    SN_inv = alpha*np.eye(M) + beta*(Phi.T @ Phi)
    SN = np.linalg.inv(SN_inv)
    mN = beta * SN @ (Phi.T @ t)
    return mN, SN

mN10, SN10 = bayes_posterior(Phi10, t10, alpha, beta)

# predictive
x_grid = np.linspace(0,1,200)
Phi_g = phi(x_grid)
m_pred = Phi_g @ mN10  # mean
# s^2(x) = beta^-1 + phi^T SN phi
s2 = 1.0/beta + np.sum((Phi_g @ SN10) * Phi_g, axis=1)  # (Phi_g @ SN10) is (N, M), elementwise times Phi_g gives quadratic form
s = np.sqrt(s2)

# true function
y_true_g = np.sin(2*np.pi*x_grid)

plt.figure(figsize=(7,5))
plt.scatter(x10, t10, label='train (N=10)', color='k')
plt.plot(x_grid, y_true_g, label='sin(2pi x)', linestyle='--')
plt.plot(x_grid, m_pred, label='predictive mean m(x)')
plt.fill_between(x_grid, m_pred - 1.96*s, m_pred + 1.96*s, alpha=0.2, label='95% pred interval')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Bayesian predictive (N=10)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,'bayes_predictive_N10.png'))
plt.close()

# MAP equivalence
# wMAP = (Phi^T Phi + lam I)^-1 Phi^T t
A = Phi10.T @ Phi10 + lam_map * np.eye(Phi10.shape[1])
b = Phi10.T @ t10
wMAP = np.linalg.solve(A,b)
mN_check = mN10
print("=== P4.2 equivalence ===")
print(f"||wMAP - mN||2 = {np.linalg.norm(wMAP - mN_check):.3e}")
# also plot comparison
y_map = Phi_g @ wMAP
plt.figure(figsize=(6,4))
plt.plot(x_grid, m_pred, label='m(x)=phi^T mN')
plt.plot(x_grid, y_map, linestyle='--', label='y(x)=phi^T wMAP')
plt.title('MAP vs posterior mean'); plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,'map_vs_mean.png'))
plt.close()
print('done p4 part1-2')
