
def linreg(X, y, t, cost, grad, a=0.1, n=100):
    for i in range(n):
        t -= a * grad(X, y, t)
        print(cost(X, y, t))
    return t  