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

# load weights
w_l = np.loadtxt(os.path.join(OUT,'lasso_w_at_star.csv'))
w_r = np.loadtxt(os.path.join(OUT,'ridge_w_at_star.csv'))

x_dense = np.linspace(0,1,200)
Phi_dense = Phi_legendre(x_dense, d)
y_l = Phi_dense @ w_l
y_r = Phi_dense @ w_r
y_true = np.sin(2*np.pi*x_dense)

plt.figure(figsize=(7,5))
plt.scatter(x,t,label='train', color='k')
plt.plot(x_dense, y_true, label='sin(2pi x)', linestyle='--')
plt.plot(x_dense, y_l, label=f'Lasso lam*=0.1')
plt.plot(x_dense, y_r, label=f'Ridge lam*=0.1585')
plt.xlabel('x'); plt.ylabel('y'); plt.title('Lasso vs Ridge fits')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT,'lasso_vs_ridge_fits.png'))
plt.close()
print('done')
