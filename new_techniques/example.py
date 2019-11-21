#!/usr/bin/env python3
from math import factorial, ceil
from scipy.optimize import minimize
import sympy as S
from sympy.abc import t
# import inspect

FLT_MAX = 3.402823466e+38


def getLipschitz(fun, x0=[0], bounds=[(-FLT_MAX, FLT_MAX)]):
    """args:
    fun: The function whose lipschitz constant is needed
    x0: The start point near the max lipschitz constant
    bounds: Sequence of (min, max) pairs for each element in x. None is
    used to specify no bound.

    return: The lipschitz constant L for the function if one exists

    """

    # XXX: Always call the minimize function with this, because it
    # handles the number of arguments correctly.
    def lambdify_wrapper(x):
        """
        args:
        x: The argument list, which will be used by scipy
        func: The actual function generated by sympy
        """
        return fun(*tuple(x))

    # # Now get the max lipschitz constant
    resmax = minimize(lambdify_wrapper, x0, bounds=bounds)
    # print(resmax)
    # print('---------')
    # Lipschitz constant is the abs value
    return abs(resmax.fun)


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
        fn = factorial(n+1)
        # TODO: Check and prove correctness of this fn*fn
        return n+1 if fn >= X else computeN(n+1)

    return computeN(n)


# This is all done at compile time.
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

    # XXX: This is the theta
    def test_multivariate():
        X = S.abc.X
        # Y = S.abc.Y
        # Xdiff = -(X**2 + Y**3 + 1)
        Xdiff = -1              # Negative because -minimize ≡ maximize
        # Xdiff = -xt.diff(X)       # The partial derivative in X
        # Now to maximize them in each variable
        # Xdiffl = S.lambdify([X, Y], Xdiff)
        Xdiffl = S.lambdify([X], Xdiff)
        # print(inspect.getsource(Xdiffl))
        return Xdiffl

    # XXX: Just a test
    tomaximize = test_multivariate()
    res = getLipschitz(tomaximize, [0])
    print('Lipschitz value for θ:', res)
    nx = getN(epsilon=1e-12, C=res)
    print('required terms for θ satisfying Lipschitz constant:', nx)
    # Now make the taylor polynomial
    taylorxcoeffs = [5*S.pi/2, 1] + [0]*(nx-2)
    # These are the smooth tokens
    taylorxpoly = sum([t**i*v for i, v in zip(range(nx), taylorxcoeffs)])
    # The theta' taylor polynomial
    print('θ(t) = ', taylorxpoly)

    # The guard function that needs the lipschitz constant
    def guard():
        diff = -((S.cos(taylorxpoly)+0.99).diff(t))
        ldiff = S.lambdify(t, diff, 'scipy')
        return ldiff

    # n is the number of taylor terms we need, can be computed at
    # compile time.
    C = getLipschitz(guard(), bounds=[(-FLT_MAX, FLT_MAX)])
    n = getN(epsilon=1e-12, C=C)
    print('\nLipschitz constant for cos(%s)+0.99: %d' % (taylorxpoly, C))
    print('Number of terms in taylor needed: ', n)

    # Some of these parts will happen at runtime
    # Now we do the example of the ode with taylor polynomial
    cosseries1 = S.fps(S.cos((5*S.pi)/2 + t)+0.99, x0=0).polynomial(n=n)
    print('Guard taylor polynomial:', cosseries1, '\n')
    # print(S.simplify(cosseries1))
    root = None
    try:
        root1 = S.nsolve(cosseries1, t, 0, dict=True)[0][t]
        root = root1
    except ValueError:
        print('No root for g(t)=0')

    # Now the second one, this one fails
    # g(t) - 2*g(0) = 0
    cosseries2 = S.fps(S.cos((5*S.pi/2) + t)-1.98, x0=0).polynomial(n=n)
    # print(S.simplify(cosseries2))
    try:
        root2 = S.nsolve(cosseries2, t, 0, dict=True)[0][t]
        root = min(root, root2)
    except ValueError:
        print('No root for g(t)-2*g(0) = 0')

    print('guard Δt:', root)


if __name__ == '__main__':
    solve()
