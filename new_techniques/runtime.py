#!/usr/bin/env python3
from math import factorial
import sympy as S
from sympy.abc import t
import mpmath as M
from itertools import count


START_SIM_TIME = 0
STOP_SIM_TIME = 50              # user defined


def getN(expr=dict(), epsilon=1e-12, method='r+s+e', Debug=0):
    """Gives the number of terms needed in the taylor polynomial to
    correctly bound the local truncation error given the step size

    h ≤ max(START_SIM_TIME - STOP_SIM_TIME)/50, 0.2) like simulink

    args:

    expr: Ode expr dictionary:

    [x(t): ([first smooth token], {dep ode exprs}, {replace func with
    values at Tₙ})], all in sympy format, default None

    epsilon: The remainder of the taylor series should be bounded by
    epsilon, default=1e-12.


    return: supremum(n ∈ ℕ) such that Lagrange error:
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

    def computeN(n):

        hn = h**(n)
        fn = factorial(n)
        slope = build(tokens)
        # Now replace all x(t) → initial values
        for (i, k) in values[2].items():
            slope = slope.replace(i, k)

        # print('ss:', slope)
        v = (slope.evalf()*hn)/fn
        if Debug in [2]:
            print('v:', (v))
        return v

    for n in count(2):
        tokens = values[0].copy()
        # print('values:', tokens)
        s = M.nsum(computeN, [n, M.inf], method=method)
        if Debug in (1, 2):
            print('Σₙᵒᵒ:', s, 'n:', n)
        if s <= epsilon:
            # tokens[:len(tokens)-1]
            return (values[0], n)
        else:
            # Automatic append inside build
            build(values[0])


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

    # XXX: This is the ode
    def test_multivariate():
        # LTI is easy to solve
        # Xdiff = S.sympify('(2*x(t) + y(t))')

        # L, but time varying
        # Xdiff = S.sympify('(5*x(t) + 2*y(t) + t**3)')

        # A sqrt
        # Xdiff = S.sympify('sqrt(x(t)+1)')

        # A power, does not seem to converge
        Xdiff = S.sympify('sqrt(x(t)**2+1)')

        # Xdiff = S.sympify('x(t)*t')

        # Xdiff = S.sympify('(y(t)) + cos(t) + x(t)')

        # These are better done with lipschitz constants.
        # Periodic functions keep on oscillating, so never seem to
        # converge.

        # Xdiff = S.sympify('sin(y(t))')

        # Non linear with periodic functions
        # XXX: Needs to be fixed
        # Xdiff = S.sympify('sin(sqrt(x(t)+1))')

        # more complex ode
        # XXX: Needs fixing
        # Xdiff = S.sympify('sin(sin(x(t)+1))')
        # the angles can only be between 0 and 2π

        # Xdiff = S.sympify('exp(x(t))')  # This seems to fuck up

        # XXX: The below one does not converge
        # Xdiff = S.sympify('x(t)*y(t)')

        return Xdiff

    tomaximize = test_multivariate()
    xt = S.sympify('x(t)')
    yt = S.sympify('y(t)')
    epsilon = 1e-6
    # Coupled ode example
    (tokens, nx) = getN({xt.diff(t): ([tomaximize],
                                      {yt.diff(t): (2*1)},
                                      # Always list all initial values
                                      # at Tₙ
                                      {xt: 5, yt: 1, t: 1})},
                        epsilon=epsilon, method='a', Debug=1)
    print(tokens)
    print('required terms for ode satisfying ε: %s: %s' % (epsilon, nx))


if __name__ == '__main__':
    M.mp.dps = 20               # Decimal precistion
    solve()
