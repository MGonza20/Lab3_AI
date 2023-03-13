
def linreg(X, y, t, cost, grad, a=0.1, n=100, on_step=None):
    for i in range(n):
        t -= a * grad(X, y, t)
        # print(cost(X, y, t))
        if on_step:
            on_step(t)

    return t  