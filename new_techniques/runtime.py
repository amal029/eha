#!/usr/bin/env python3
from math import factorial
import sympy as S
from sympy.abc import t
import mpmath as M
from itertools import count


START_SIM_TIME = 0
STOP_SIM_TIME = 60              # user defined


def getN(expr=dict(), epsilon=1e-12, method='r+s', Debug=0):
    """Gives the number of terms needed in the taylor polynomial to
    correctly bound the local truncation error given the step size

    h ≤ max(START_SIM_TIME - STOP_SIM_TIME)/50, 0.2) like simulink

    args:

    expr: Ode expr dictionary:

    [x(t): ([first smooth token], {dep ode exprs}, {replace func with
    values at Tₙ})], all in sympy format, default None

    epsilon: The remainder of the taylor series should be bounded by
    epsilon, default=1e-12.


    return: heuristic (n ∈ ℕ) such that Lagrange error:
    f⁽ⁿ⁺¹)(η)/(n+1)!⋆(h⁽ⁿ⁺¹⁾) ≤ epsilon, n ≥ 2, η ∈ (Tₙ, x)

    """
    assert(len(list(expr)) == 1)
    values = list(*(expr.values()))
    assert(len(values) == 3)
    if Debug in (1, 2):
        print(values)

    h = max(abs((START_SIM_TIME - STOP_SIM_TIME)/50), 0.2)

    def build(tokens):
        slope = tokens[-1].diff(t)
        # 1.) Replace Derivative(x(t), t) → token[0]
        slope = slope.replace(list(expr)[0], tokens[0])
        # 2.) Replace Derivative(deps(t), t) → exprs
        for i, k in values[1].items():
            slope = slope.replace(i, k)

        # XXX: This is important
        tokens.append(slope)
        # print(slope)
        return slope

    def replace(slope):
        for (i, k) in values[2].items():
            slope = slope.replace(i, k)
        return slope

    def computeN(n):

        hn = h**(n)
        fn = factorial(n)
        slope = build(tokens)
        slope = replace(slope)
        v = (slope.evalf()*hn)/fn
        if Debug in [2]:
            print('v:', abs(v))
        return abs(v)

    n = len(values[0]) + 1
    ps = None                   # Previous sum
    while True:
        tokens = values[0]
        s = M.nsum(computeN, [n, M.inf], method=method)
        assert (s/ps < 1) if n > 2 else True
        ps = s
        # This is the convergence test
        if Debug in (1, 2):
            print('Σₙᵒᵒ:', s, 'n:', n)
        if s <= epsilon:
            # Replace the final value with taylor term
            values[0] = values[0][:n]
            diff_coeffs = values[0]
            values[0] = [replace(x).evalf() for x in values[0]]
            # Insert the initial value, constant term of taylor
            values[0].insert(0, (values[2][list(expr)[0].args[0]]))
            # Build the taylor polynomial for the ode
            polynomial = sum([c*h**p/factorial(p)
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

        # a power, does not seem to converge
        # Xdiff = S.sympify('sqrt(x(t)**2+1)')

        # Xdiff = S.sympify('x(t) + y(t)')

        # Xdiff = S.sympify('(y(t)) + cos(t) + (x(t)+1)')
        # Xdiff = S.sympify('t*(x(t)-2) + y(t)')

        # Xdiff = S.sympify('sin(y(t))')

        # Non linear with periodic functions
        # Xdiff = S.sympify('sin(sqrt(x(t)+1))')

        # more complex ode
        # Xdiff = S.sympify('sin(sin(y(t)+1))')

        # Xdiff = S.sympify('exp(x(t))')  # This seems to fuck up

        # XXX: The below one does not converge
        # Xdiff = S.sympify('x(t)*y(t)')

        return Xdiff

    # xxx: In the next iteration (integration step) start with n which
    # currently holds, else increase n.
    tomaximize = test_multivariate()
    xt = S.sympify('x(t)')
    yt = S.sympify('y(t)')
    dydt = 2*xt - 1
    epsilon = 1e-6
    # Coupled ode example
    (final_tokens,
     tokens, nx) = getN({xt.diff(t): ([tomaximize],
                                      {yt.diff(t): dydt},
                                      # Always list all
                                      # initial values at
                                      # Tₙ
                                      {xt: 5.6, yt: 1, t: 0})},
                        epsilon=epsilon, method='r+s', Debug=1)
    print('required terms for dx/dt: %s satisfying ε: %s: %s' %
          (tomaximize, epsilon, nx))
    print('Taylor polynomial for dx/dt: %s with dy/dt: %s is %s' %
          (tomaximize, dydt, tokens))
    # print(final_tokens)          # Only needed for the next iteration


if __name__ == '__main__':
    M.mp.dps = 2               # Decimal precision
    solve()
