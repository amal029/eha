#!/usr/bin/env python3
import matplotlib.pyplot as plt
from numpy.polynomial import Polynomial as P
import numpy as N


# fxt and gxt are slope functions of x(t), t
def compute_step(fxt, gxt, dq, R):
    # First get the number of random variables
    dWt = N.random.standard_normal(R)
    Winc = sum(dWt)
    gn = gxt * Winc
    # This is the max dq we can take
    dq2 = gn**2 / (4 * fxt * R)
    odq = dq
    dq = dq if dq <= dq2 else dq2
    print('Given dq: %f, chosen dq: %f' % (odq, dq))
    # First coefficient
    a = R * (fxt**2)
    # Second coefficient
    b = ((2 * fxt * dq * R) - (gn**2))
    # Third coefficient
    c = R*(dq**2)

    # The first polynomial ax² + bx + c = 0
    # Build the polynomial to solve for dt
    p = P([a, b, c])
    roots = p.roots()
    roots = [i for i in roots if N.isreal(i)]
    root1 = min(roots) if len(roots) != 0 else None
    print(p, root1)

    # The second polynomial ax² - bx + cx = 0
    b = ((2 * fxt * dq * R) + (gn**2))
    p = P([a, -b, c])
    roots = p.roots()
    roots = [i for i in roots if N.isreal(i)]
    root2 = min(roots) if len(roots) != 0 else None
    print(p, root2)

    # Now get Δt and δt
    Dt = root1 if root1 is not None else root2
    Dt = min(Dt, root2) if root2 is not None else Dt
    dt = Dt/R
    print('Δt: %f, δt: %f' % (Dt, dt))

    assert False
    return (Dt, dt)


def main(delta=1e-10):
    """This is an example of scalar SDE solution using quantised state
    integration.

    """
    # dx = λx(t)dt + μx(t)dW(t) scalar SDE example
    # λ = 2, μ = 1, x(0) = 1

    fxt = (lambda x: (lambda t: 2 * x))
    gxt = (lambda x: (lambda t: 1 * x))
    # Variables, and initial values
    x = 1                       # Initial value
    xf = 1.1
    t = 0
    vs = [x]
    ts = [t]
    while(True):
        dt = compute_step(fxt(x)(t), gxt(x)(t), (xf-x), 4)
        t = t + dt
        # Now compute x(t) using Euler-Maruyama solution to get x(t)
        vs.append(x)
        ts.append(t)
        # Check if δ <= |x(t) - xf| <= δ
        if abs(x - xf) <= delta:
            break
    return vs, ts


if __name__ == '__main__':
    xs, ts = main()
    plt.plot(ts, xs)
