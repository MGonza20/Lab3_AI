
import numpy as np 
import matplotlib.pyplot as plt
import quad 
from linreg import linreg 

DATASET_SIZE = 100 
DATASET_SPARSE_RATIO = 20
DATASET_X_LIM = 10

# Imaginemos que X es dado
X = np.linspace(0, DATASET_X_LIM, DATASET_SIZE).reshape((DATASET_SIZE, 1))
Xr = np.hstack((
    np.ones((DATASET_SIZE, 1)),
    X
))
y = 3 + 2 * X + np.random.rand(DATASET_SIZE, 1) * DATASET_SPARSE_RATIO
to = np.random.rand(Xr.shape[1], 1)
tf = linreg(Xr, y, to, quad.cost, quad.grad, a=0.025, n=500)
print("Tf: ", tf)

xm = np.array([[1], [DATASET_X_LIM]])
xmr = np.hstack((
    np.ones((2, 1)),
    xm
))
ym = xmr @ tf

plt.plot(X, y, 'ro')
plt.plot(xm, ym)
plt.show()
