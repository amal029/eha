#!/usr/bin/env python3
from math import cos, factorial, ceil
from scipy.optimize import minimize
import math
import sympy as S
from sympy.abc import t


def getLipschitz(fun, x0min=0, x0max=0):
    """
    args:
    fun: The function whole lipschitz constant is needed
    x0min: The start point near the min lipschitz constant
    x0max: The start point near the max lipschitz constant

    return:
    The lipschitz constant L for the function if on exists
    """

    # Get the min lipschitz constant
    resmin = minimize(fun, x0min, args=(True))
    # Now get the max lipschitz constant
    resmax = minimize(fun, x0max, args=(False))
    # Lipschitz constant is max of resmin and resmax
    L = max(abs(resmax.fun), abs(resmin.fun))
    return L


def getN(epsilon, C, n=0):
    """Gives the number of terms needed in the taylor polynomial to
    correctly bound the local truncation error given the step size h ∈
    (0, 1)

    args:

    epsilon: The remainder of the taylor series should be bounded by
    epsilon.

    C: Absolute value of Lipschitz constant
    n0: starting value of n, default = 0

    return:
    supremum(n ∈ ℕ), C/(n+1)! ≤ epsilon

    Theorems used:

    1.) fⁿ(x) ≤ f¹(x) ≤ C, see lipschitz value of higher derivatives and
    sobolov spaces.

    2.) Lagrange error: f⁽ⁿ⁺¹)(x₀)/(n+1)!⋆(h⁽ⁿ⁺¹⁾), where h = 1, in the worst
    case and fⁿ(x₀) ≤ C form 1.) above.

    """
    X = int(ceil(C/epsilon))         # The bound

    # First see if the starting value of n ≥ X, if yes then binary
    # else increase it by 1
    def computeN(n):
        return n+1 if factorial(n+1) >= X else computeN(n+1)

    return computeN(n)


# This is all done at compile time.
if __name__ == '__main__':
    """Solving cos(x) <= -0.99, dx/dt=1, x(0) = 0
    """
    # The guard function that needs the lipschitz constant
    def guard(x, m):
        return cos((5*math.pi/2)+x)+0.99 if m else -(cos(x)+0.99)

    # n is the number of taylor terms we need, can be computed at
    # compile time. But epsilon seems to effect the result (the root)
    # quite massively.
    C = getLipschitz(guard)
    n = getN(epsilon=1e-12, C=C)
    print('Lipschitz constant for cos(5*π/2+t)+0.99:', C)
    print('Number of terms in taylor needed: ', n)

    # I think it is important to find all roots and then take the
    # minimum here.

    # Now we do the example of the ode with taylor polynomial
    cosseries1 = S.fps(S.cos((5*S.pi)/2 + t)+0.99, x0=0).polynomial(n=n)
    # print(S.simplify(cosseries1))
    root = None
    try:
        root1 = S.nsolve(cosseries1, t, 0, dict=True)[0][t]
    except ValueError:
        print('No root for g(t)=0')
    else:
        root = root1

    # Now the second one, this one fails
    # g(t) - 2*g(0) = 0
    cosseries2 = S.fps(S.cos((5*S.pi/2) + t)-1.98, x0=0).polynomial(n=n)
    # print(S.simplify(cosseries2))
    try:
        root2 = S.nsolve(cosseries2, t, 0, dict=True)[0][t]
    except ValueError:
        print('No root for g(t)-2*g(0) = 0')
    else:
        root = min(root, root2) if root2 is not None else root2

    print('guard Δt:', root)
