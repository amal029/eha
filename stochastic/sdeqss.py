#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as N
import mpmath as M


# TODO: Check for condition that you always get at least one real
# positive root.

# fxt and gxt are slope functions of x(t), t
def compute_step(fxt, gxt, dq, R, dWt):

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
    # There can be only one root.
    f = (lambda x: a*(x**2) + (b*x) + c)
    try:
        root1 = M.findroot(f, 0)
        # Debug
        print('root1:', root1)
        root1 = root1 if root1 > 0 else None
    except ValueError as e:
        print(e)
        root1 = None

    # The second polynomial ax² - bx + cx = 0
    b = ((2 * fxt * dq * R) + (gn**2))
    f = (lambda x: a*(x**2) - (b*x) + c)
    try:
        root2 = M.findroot(f, 0)
        print('root2:', root2)
        root2 = root2 if root2 > 0 else None
    except ValueError as e:
        print(e)
        root2 = None

    # Now get Δt and δt
    Dt = root1 if root1 is not None else root2
    Dt = min(Dt, root2) if root2 is not None else Dt
    dt = Dt/R
    # print('Δt: %f, δt: %f' % (Dt, dt))

    # assert False
    return Dt, dt


def main(delta=1e-10, R=4):
    """This is an example of scalar SDE solution using quantised state
    integration.

    """
    # dx = λx(t)dt + μx(t)dW(t) scalar SDE example
    # λ = 2, μ = 1, x(0) = 1
    assert R > 1

    fxt = (lambda x: (lambda t: 2 * x))
    gxt = (lambda x: (lambda t: 1 * x))
    # Variables, and initial values
    x = 1                       # Initial value
    xf = 4.5
    t = 0
    vs = [x]
    ts = [t]

    while(True):
        curr_fxt = fxt(x)(t)    # The current slope
        curr_gxt = gxt(x)(t)    # The current dWt slope

        # First get the number of random variables
        dWt = N.random.randn(R)

        dq = abs(xf-x)          # Initially
        # This needs to be done iteratively
        while(True):
            xtemp = x
            xt = x
            Dt, dt = compute_step(curr_fxt, curr_gxt, dq=dq, dWt=dWt,
                                  R=R)
            # Now compute x(t) using Euler-Maruyama solution to get x(t)
            # First build the weiner process
            Winc = sum(N.sqrt(dt) * dWt)

            # EM
            xtemp += (Dt * curr_fxt) + (curr_gxt * Winc)

            for i in dWt:
                xt += (dt * fxt(xt)(t)) + (gxt(xt)(t) * N.sqrt(dt) * i)

            dt = float(dt)
            tol = N.sqrt(1 + N.log(1/dt))*N.sqrt(dt)
            print('tutu:', abs(xtemp - xt), xtemp, xt, tol,
                  abs(xtemp - xt) <= tol)
            # Now compute the value at the smallest steps of δt (we can
            # make this better, by doing it at δt*R/2)

            # XXX: Here we break it, if error is met,
            # else we half the dq
            if abs(xtemp - xt) <= tol:
                break
            else:
                # Can we do better than this?
                dq = dq/2       # Half it and try again

        # Now we can set the real value
        x = xt
        # Increment the time-step
        t = t + Dt

        # Append to plot later on
        vs.append(x)
        ts.append(t)

        # XXX: Equality seems to work, but we need a prove that says we
        # converge to the discontinuity.

        # The proof depends upon the following:

        # 1.) Get all the roots of the polynomial, and take the minimum from
        # amongst those roots.
        # 2.) Make sure that the obtained Dt is such that the local truncation
        # error met
        # 3.) The guard is a taylor expansion of some function.
        # 4.) The value at any step x(t + Dt) <= x(t) + Dq, where Dq =
        # x(T) - x(t), where x(T) satisfies the guard .

        # Maintain the brownian path again when restarting the step --
        # but this should not happen for us.

        if abs(x - xf) <= delta:
            break

    return vs, ts


if __name__ == '__main__':
    # N.random.seed(0)
    xs, ts = main(R=10)
    # print(list(zip(ts, xs)))
    plt.plot(ts, xs)
    plt.show()

# Make this iterative with local error ≤ sqrt(1 + log(1/Δt)) * sqrt(Δt)

# Compare with all the micro-benchmarks from the paper: THE EULER SCHEME
# FOR STOCHASTIC DIFFERENTIAL EQUATIONS WITH DISCONTINUOUS DRIFT
# COEFFICIENT: A NUMERICAL STUDY OF THE CONVERGENCE RATE
