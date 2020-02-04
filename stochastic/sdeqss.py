#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as N
import mpmath as M


# TODO: Check for condition that you always get at least one real
# positive root.

# fxt and gxt are slope functions of x(t), t
def compute_step(fxt, gxt, dq, R):
    # First get the number of random variables
    dWt = N.random.randn(R)
    print('dWt:', dWt)
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

    # Use mpmath to get the roots
    f = (lambda x: a*(x**2) + (b*x) + c)
    try:
        root1 = M.findroot(f, 0, tol=1e-3)
        # Debug
        print('root1:', root1)
        root1 = root1 if root1 > 0 else None
    except ValueError as e:
        print(e)
        root1 = None

    # The second polynomial ax² - bx + cx = 0
    b = ((2 * fxt * dq * R) + (gn**2))
    f = (lambda x: a*(x**2) + (b*x) + c)
    try:
        root2 = M.findroot(f, 0, tol=1e-3)
        print('root2:', root2)
        root2 = root2 if root2 > 0 else None
    except ValueError as e:
        print(e)
        root2 = None

    # Now get Δt and δt
    Dt = root1 if root1 is not None else root2
    Dt = min(Dt, root2) if root2 is not None else Dt
    dt = Dt/R
    print('Δt: %f, δt: %f' % (Dt, dt))

    # assert False
    return (Dt, dt, dWt)


def main(delta=1e-10):
    """This is an example of scalar SDE solution using quantised state
    integration.

    """
    # dx = λx(t)dt + μx(t)dW(t) scalar SDE example
    # λ = 2, μ = 1, x(0) = 1

    fxt = (lambda x: (lambda t: 2 * 1))
    gxt = (lambda x: (lambda t: 1 * 1))
    # Variables, and initial values
    x = 1                       # Initial value
    xf = 2
    t = 0
    vs = [x]
    ts = [t]
    # count = 0                   # Debug
    while(True):
        curr_fxt = fxt(x)(t)
        curr_gxt = gxt(x)(t)
        Dt, dt, dWt = compute_step(curr_fxt, curr_gxt, abs(xf-x), R=4)
        # Now compute x(t) using Euler-Maruyama solution to get x(t)
        # First build the weiner process
        dWt = N.sqrt(dt) * dWt
        print('New dWt:', dWt)
        Winc = sum(dWt)
        # EM
        x = x + (Dt * curr_fxt) + (curr_gxt * Winc)
        # Increment the time-step
        t = t + Dt
        # Append to plot later on
        vs.append(x)
        ts.append(t)
        # This does not guarantee first time
        # XXX: Check
        if abs(x - xf) <= delta:
            break
        # elif count > 100:
        #     break
        # count += 1
    return vs, ts


if __name__ == '__main__':
    # N.random.seed(0)
    xs, ts = main()
    plt.plot(ts, xs)
    plt.show()
