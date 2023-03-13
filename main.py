
import numpy as np 
import matplotlib.pyplot as plt
import quad 
from linreg import linreg 

DATASET_SIZE = 100 
DATASET_SPARSE_RATIO = 20
DATASET_X_LIM = 10

# Imaginemos que X es dado
def get_random_dataset():
    X = np.linspace(0, DATASET_X_LIM, DATASET_SIZE).reshape((DATASET_SIZE, 1))
    Xr = np.hstack((
        np.ones((DATASET_SIZE, 1)),
        X
    ))
    y = 3 + 2 * X + np.random.rand(DATASET_SIZE, 1) * DATASET_SPARSE_RATIO
    return Xr, y


Xr, y = get_random_dataset()
to = np.random.rand(Xr.shape[1], 1)

tf, costs = linreg(
    Xr,
    y,
    to,
    quad.cost,
    quad.grad, 
    a=0.025, 
    n=20
)

print("Tf: ", tf)

xm = np.array([[0], [DATASET_X_LIM]])
xmr = np.hstack((
    np.ones((2, 1)),
    xm
))
ym = xmr @ tf

plt.plot(Xr[:, 1], y, 'ro')
plt.plot(xm, ym)
plt.show()

plt.plot(costs)
plt.show()
