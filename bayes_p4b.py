import numpy as np
import matplotlib.pyplot as plt
import os

OUT = "notebooks/Outputs/Bayesian"
os.makedirs(OUT, exist_ok=True)

sigma2 = 0.01
beta = 1.0/sigma2
alpha = 0.05
d = 3
def phi(x):
    x = np.atleast_1d(x)
    return np.column_stack([x**i for i in range(d+1)])

def bayes_posterior(Phi, t, alpha, beta):
    M = Phi.shape[1]
    SN_inv = alpha*np.eye(M) + beta*(Phi.T @ Phi)
    SN = np.linalg.inv(SN_inv)
    mN = beta * SN @ (Phi.T @ t)
    return mN, SN

x_grid = np.linspace(0,1,200)
Phi_g = phi(x_grid)

Ns = [5,10,20,50]
files = {
    5: "data/p4_train_N5.csv",
    10: "data/p4_train_N10.csv",
    20: "data/p4_train_N20.csv",
    50: "data/p4_train_N50.csv",
}

plt.figure(figsize=(7,5))
for N in Ns:
    data = np.loadtxt(files[N], delimiter=",", skiprows=1)
    xN, tN = data[:,0], data[:,1]
    PhiN = phi(xN)
    mN, SN = bayes_posterior(PhiN, tN, alpha, beta)
    s2 = 1.0/beta + np.sum((Phi_g @ SN) * Phi_g, axis=1)
    s = np.sqrt(s2)
    width = 2*1.96*s
    plt.plot(x_grid, width, label=f'N={N}')
# irreducible floor
floor = 2*1.96*np.sqrt(1.0/beta)
plt.axhline(floor, color='k', linestyle='--', label='irreducible 2*1.96/sqrt(beta)')
plt.xlabel('x'); plt.ylabel('95% pred interval width'); plt.title('Predictive uncertainty vs N')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,'bayes_uncertainty_vs_N.png'))
plt.close()
print('done')
