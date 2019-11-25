#!/usr/bin/env python3
from math import factorial
import sympy as S
from sympy.abc import t
import mpmath as M

START_SIM_TIME = 0

MAX_STEP_SIZE = 0.05

# This can also make the series diverge!
STOP_SIM_TIME = 0.07              # user defined

h = 0                           # The step size


def getN(expr=dict(), epsilon=1e-12, method='r+s', Debug=0):
    """Gives the number of terms needed in the taylor polynomial to
    correctly bound the local truncation error given the step size

    h ≤ max(START_SIM_TIME - STOP_SIM_TIME)/50, 0.2) like simulink

    args:

    expr: Ode expr dictionary:

    [x(t): ([first smooth token], {all ode exprs}, {replace func with
    values at Tₙ})], all in sympy format, default None

    epsilon: The remainder of the taylor series should be bounded by
    epsilon, default=1e-12.


    return: heuristic (n ∈ ℕ) such that Lagrange error:
    f⁽ⁿ⁺¹)(η)/(n+1)!⋆(h⁽ⁿ⁺¹⁾) ≤ epsilon, n ≥ 2, η ∈ (Tₙ, x)

    """
    assert(len(list(expr)) == 1)
    values = list(*(expr.values()))
    assert(len(values) == 3)
    if Debug in (1, 2, 3):
        print(values)

    global h
    if MAX_STEP_SIZE is None:
        h = max(abs((START_SIM_TIME - STOP_SIM_TIME)/50), 0.2)
    else:
        h = MAX_STEP_SIZE

    def build(tokens):
        slope = tokens[-1].diff(t)
        # 2.) Replace Derivative(deps(t), t) → exprs
        if Debug in [3]:
            print(slope)
        for i, k in values[1].items():
            slope = slope.replace(i, k)
        if Debug in [3]:
            print(slope)

        # XXX: This is important
        tokens.append(slope)
        # print(slope)
        return slope

    def replace(slope):
        for (i, k) in values[2].items():
            slope = slope.replace(i, k)
            if Debug in [3]:
                print('replacing %s → %s' % (i, k))
                print(slope)
        return slope

    def computeN(n):

        hn = h**(n)
        fn = factorial(n)
        slope = build(tokens)
        if Debug in [3]:
            print(slope)
        slope = replace(slope)
        if Debug in [3]:
            print(slope)
        v = (slope.evalf()*hn)/fn
        if Debug in [2, 3]:
            print('v:', abs(v))
        return abs(v)

    n = len(values[0]) + 1
    ps = None                   # Previous sum
    while True:
        tokens = values[0]
        s = abs(M.nsum(computeN, [n, M.inf], method=method))
        if Debug in (1, 2, 3):
            print('Σₙᵒᵒ:', s, 'n:', n)
        # This is the convergence test
        if n > 2 and (s/ps > 1):
            raise AssertionError('Series diverges %s/%s' % (ps, s))
        ps = s
        if s <= epsilon:
            # Replace the final value with taylor term
            values[0] = values[0][:n]
            diff_coeffs = values[0]
            values[0] = [replace(x).evalf() for x in values[0]]
            # Insert the initial value, constant term of taylor
            values[0].insert(0, (values[2][list(expr)[0].args[0]]))
            # Build the taylor polynomial for the ode
            polynomial = sum([c*S.abc.x**p/factorial(p)
                              for c, p in zip(values[0], range(n))])
            return (diff_coeffs, polynomial, n)
        else:
            # Automatic append inside build. This is optimised, but does
            # not give the supremum for n.
            n = len(values[0]) + 1


def solve():
    """Solving cos(x) <= -0.99, dx/dt=1, x(0) = 0
    # Basic steps:
    # 1. First compute the n terms for each ode
    # 2. Next replace the guard with ode(t), so that it is only in t
    # 3. Then compute the number of terms needed for g(t)
    # 4. Finally, compute g(t) = 0 and g(t)-2g(0) = 0

    # 5. Note that computing number of terms "n" in taylor essentially
    # guarantees that tᵣ - t ≤ floating point error only, specified by the
    # polynomial solver.
    """

    # XXX: These are different ode examples.
    def test_multivariate():
        # LTI is easy to solve
        # Xdiff = S.sympify('(2*x(t) + y(t))')

        # L, but time varying
        Xdiff = S.sympify('(5*x(t) + 2*y(t) + (t-9)**3 - sin(t))')

        # A sqrt
        # Xdiff = S.sympify('sqrt(x(t)+1)')

        # A power, does not seem to converge
        # Xdiff = S.sympify('sqrt(x(t)**2+1)')

        # Xdiff = S.sympify('x(t) + y(t)')

        # Xdiff = S.sympify('(y(t)) + cos(t) + (x(t)+1)')
        # Xdiff = S.sympify('t*(x(t)-2) + y(t)')

        # This seems to diverge
        # Xdiff = S.sympify('sin(y(t))')

        # Non linear with periodic functions
        # Xdiff = S.sympify('sin(sqrt(x(t)+1))')

        # more complex ode
        # Xdiff = S.sympify('sin(sin(y(t)+1))')

        # This one diverges too
        # Xdiff = S.sympify('exp(x(t))')

        # XXX: The below one does not converge
        # Xdiff = S.sympify('x(t)*y(t)')

        return Xdiff

    # The robot example
    # Some constants
    v1 = 30
    v2 = -10.0
    le = 1

    # xxx: In the next iteration (integration step) start with n which
    # currently holds, else increase n.
    # tomaximize = test_multivariate()
    xt = S.sympify('x(t)')
    xtdt = S.sympify(S.sympify('cos(th(t))')*v1)
    yt = S.sympify('y(t)')
    ytdt = S.sympify(S.sympify('sin(th(t))')*v1)
    pht = S.sympify('ph(t)')
    phtdt = S.sympify(v2)
    tht = S.sympify('th(t)')
    thtdt = S.sympify((S.sin(pht)/S.cos(pht))/le*v1)
    epsilon = 1e-4
    # Dependent odes
    dodes = {xt.diff(t): xtdt, yt.diff(t): ytdt, tht.diff(t): thtdt,
             pht.diff(t): phtdt}
    # Initial values
    # t is the current time (Tₙ)
    initivals = {xt: 0, yt: 1, tht: 0, pht: 1, t: 0}
    # Integrating xt
    (final_tokens, xpoly, nx) = getN({xt.diff(t): ([xtdt], dodes, initivals)},
                                     epsilon=epsilon, method='r+s', Debug=0)
    # print('required terms xtdt: %s satisfying ε: %s: %s' % (xtdt, epsilon,
    #                                                         nx))
    # print('∫', xt, 'dt: ', xpoly)
    # Integrating yt
    (final_tokens, ypoly, ny) = getN({yt.diff(t): ([ytdt], dodes, initivals)},
                                     epsilon=epsilon, method='r+s', Debug=0)
    # print('required terms ytdt: %s satisfying ε: %s: %s' % (ytdt, epsilon,
    #                                                         nx))
    # print('∫', yt, 'dt: ', ypoly)
    # Integrating tht
    (final_tokens, thpoly, nt) = getN({tht.diff(t): ([thtdt], dodes,
                                                     initivals)},
                                      epsilon=epsilon, method='r+s', Debug=0)
    # print('required terms thtdt: %s satisfying ε: %s: %s' % (thtdt, epsilon,
    #                                                          nx))
    # print('∫', tht, 'dt: ', thpoly)
    # print('∫', tht, 'dt: ', tokens.replace(S.abc.h, h).evalf())

    # Integrating pht
    (final_tokens, phpoly, np) = getN({pht.diff(t): ([phtdt], dodes,
                                                     initivals)},
                                      epsilon=epsilon, method='r+s', Debug=0)
    # print('required terms thtdt: %s satisfying ε: %s: %s' % (phtdt, epsilon,
    #                                                          nx))
    # print('∫', pht, 'dt: ', phpoly)

    print('With max step size: %s, we have these values' % h)
    print('∫', xt, 'Δh: ', xpoly.replace(S.abc.x, h).evalf())
    print('∫', yt, 'Δh: ', ypoly.replace(S.abc.x, h).evalf())
    print('∫', tht, 'Δh: ', thpoly.replace(S.abc.x, h).evalf())
    print('∫', pht, 'Δh: ', phpoly.replace(S.abc.x, h).evalf())

    # XXX: Solve for the guard step-size

    # I am just doing yt, because I know it hits first in this example.

    # The first case y(t)-yt0 - 1.8 = 0, 1.8 is the guard
    # Use mpmath to find root, because numpy gives a wrong root!
    # g(t) = 0 case
    gpolylambds = (lambda x: (ypoly-1.8).replace(S.abc.x, x).evalf())
    M.mp.dps = 10
    root = M.findroot(gpolylambds, 0.04199999, tol=epsilon, verify=True,
                      method='newton', verbose=False, maxsteps=1e3)
    print('With root step size: %s, we have these values' % root)
    print('∫', xt, 'Δroot: ', xpoly.replace(S.abc.x, root).evalf())
    print('∫', yt, 'Δroot: ', ypoly.replace(S.abc.x, root).evalf())
    print('∫', tht, 'Δroot: ', thpoly.replace(S.abc.x, root).evalf())
    print('∫', pht, 'Δroot: ', phpoly.replace(S.abc.x, root).evalf())

    dt = min(h, root)
    # Compute the values from amongst them all
    print('Value with correct step size: %s' % dt)
    print('∫', xt, 'dt: ', xpoly.replace(S.abc.x, dt).evalf())
    print('∫', yt, 'dt: ', ypoly.replace(S.abc.x, dt).evalf())
    print('∫', tht, 'dt: ', thpoly.replace(S.abc.x, dt).evalf())
    print('∫', pht, 'dt: ', phpoly.replace(S.abc.x, dt).evalf())


if __name__ == '__main__':
    M.mp.dps = 4               # Decimal precision
    solve()
