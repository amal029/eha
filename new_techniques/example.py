#!/usr/bin/env python3
from math import factorial
from scipy.optimize import differential_evolution
import sympy as S
from sympy.abc import t
import math
# import inspect

# XXX: Can we not have a scaling factor, where all the variables can
# only be between [-1, 1] when computing the Lipschitz value, and
# running the example and then for plotting it we just multiply
# everything with that factor?
# FLT_MAX = 1

START_SIM_TIME = 0
STOP_SIM_TIME = 60              # user defined


def getLipschitz(fun, bounds):
    """args:
    fun: The function whose lipschitz constant is needed
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

    # Now get the max lipschitz constant
    resmax = differential_evolution(lambdify_wrapper, bounds)
    return abs(resmax.fun)


# XXX: This can be optimized for LTI systems. With LTI systems we have
# to compute the lipschitz constant just once. Because with LTI systems:
# Lⁿₓ = A*L⁽ⁿ⁻¹⁾ₓ

# XXX: In the general case, we have to compute the lipschitz constant
# every time, which is what I am doing here.

def getN(expr=dict(), epsilon=1e-12, FLT_MAX=3.402823466e+38,
         FLT_MIN=-3.402823466e+38):
    """Gives the number of terms needed in the taylor polynomial to
    correctly bound the local truncation error given the step size

    h ≤ max(START_SIM_TIME - STOP_SIM_TIME)/50, 0.2) like simulink

    args:

    expr: Ode expr dictionary:

    [x(t): ([first smooth token], {dep ode exprs}, {replace func with
    names}, [lambdify args])], all in sympy format, default None

    epsilon: The remainder of the taylor series should be bounded by
    epsilon, default=1e-12.


    return: supremum(n ∈ ℕ) such that Lagrange error:
    f⁽ⁿ⁺¹)(x₀)/(n+1)!⋆(h⁽ⁿ⁺¹⁾) ≤ epsilon, n ≥ 2

    """
    assert(len(list(expr)) == 1)
    values = list(*(expr.values()))
    assert(len(values) == 4)
    # print(values)

    h = max((START_SIM_TIME - STOP_SIM_TIME)/50, 0.2)

    def build(slope, tokens, Ls1=None):
        # 1.) Replace Derivative(x(t), t) → token[0]
        lslope = slope
        slope = slope.replace(list(expr)[0], tokens[0])
        if Ls1 is not None:
            k = list(expr)[0]
            lslope = lslope.replace(k, Ls1[k])
        else:
            lslope = slope
        # 2.) Replace Derivative(deps(t), t) → exprs
        for i, k in values[1].items():
            slope = slope.replace(i, k[0])
            if Ls1 is not None:
                lslope = lslope.replace(i, Ls1[i])
            else:
                lslope = slope
        # Now lambdify the argument and find its maximum
        # 1. Replace all func names with variable names
        # lslope = slope
        for (i, k) in values[2].items():
            lslope = lslope.replace(i, k)
        return slope, lslope

    def computeN(n, tokens, Ls1=None):
        hn = h**(n+1)
        fn = factorial(n+1)

        # XXX: Compute the lipschitz constant
        slope = tokens[-1].diff(t)
        # print('sending:', slope)
        slope, lslope = build(slope, tokens, Ls1)

        # Make bounds
        bounds = [(FLT_MIN, FLT_MAX)]*len(values[3])
        L = S.lambdify(values[3], (-lslope))
        print('\n', lslope, '\n')
        L = getLipschitz(L, bounds)
        # The remainder theorem
        if (L*hn)/fn <= epsilon:
            return (tokens, n)
        else:
            # Append the smooth tokens here
            tokens.append(slope)
            return computeN(n+1, tokens, Ls1)

    # Compute the first smooth tokens lipschitz constants
    _, dfx = build(values[0][0], values[0])
    # Make bounds
    bounds = [(FLT_MIN, FLT_MAX)]*len(values[3])
    L = S.lambdify(values[3], (-dfx))
    L = getLipschitz(L, bounds)
    Ls1 = {list(expr)[0]: L}

    # FIXME: This needs to be completed, need all the information for
    # this too.
    for i, k in values[1].items():
        _, dfi = build(k[0], [k[0]])
        # Make bounds
        bounds = [(FLT_MIN, FLT_MAX)]*len(k[1])
        L = S.lambdify(k[1], (-dfx))
        L = getLipschitz(L, bounds)
        Ls1[i] = L

    # Call with the first smooth token
    return computeN(2, values[0], Ls1)


# This is all done at compile time.
def solve(FLT_MIN, FLT_MAX):
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
        # LTI is easy to solve
        # Xdiff = S.sympify('(5*x(t) + 2*y(t) + 1)')

        # Time varying, takes more time in general,
        # with increasing power for t^n
        # Xdiff = S.sympify('(5*x(t) + 2*y(t) + t**3)')

        # Non linear with periodic functions
        # Xdiff = S.sympify('sin(sqrt(x(t)+1))')
        # FLT_MIN = 0
        # FLT_MAX = 2*math.pi

        # More complex ode
        # Xdiff = S.sympify('sin(sin(x(t)+1))')
        # The angles can only be between 0 and 2π
        # FLT_MIN = 0
        # FLT_MAX = 2*math.pi

        # A sqrt
        # Xdiff = S.sympify('sqrt(x(t)+1)')

        # The ones below need to have a reduced search space bound for
        # continous variables.

        # Another sqrt, does not seem to converge
        # Xdiff = S.sympify('x(t)*t')

        # Now multiplication
        Xdiff = S.sympify('x(t)*y(t)')

        # Using scaling factor, to reduce the bounds of the maximisation
        # problem.
        FLT_MIN = -1e1
        FLT_MAX = 1e1

        return FLT_MIN, FLT_MAX, Xdiff

    FLT_MIN, FLT_MAX, tomaximize = test_multivariate()
    xt = S.sympify('x(t)')
    x = S.abc.x
    yt = S.sympify('y(t)')
    y = S.abc.y
    # Coupled ode example
    (tokens, nx) = getN({xt.diff(t): ([tomaximize],
                                      {yt.diff(t): (xt,
                                                    # args
                                                    [x, y, t])},
                                      # Always list all the replacements
                                      {xt: x, yt: y},
                                      [x, y, t])},
                        FLT_MIN=FLT_MIN, FLT_MAX=FLT_MAX)
    # print(tokens)
    print('required terms for θ satisfying Lipschitz constant:', nx)

    # Now make the taylor polynomial
    taylorxcoeffs = [5*S.pi/2, 1] + [0]*(nx-2)
    # These are the smooth tokens
    taylorxpoly = sum([t**i*v for i, v in zip(range(nx), taylorxcoeffs)])
    # The theta' taylor polynomial
    print('θ(t) = ', taylorxpoly)

    # The guard function that needs the lipschitz constant
    def guard():
        gt = (S.cos(taylorxpoly)+0.99)
        return gt.diff(t)

    gt = S.sympify('g(t)')
    tokens, n = getN({gt.diff(t): ([guard()], dict(), dict(), [t])})
    # print(tokens)
    print('Number of terms for cos(%s)+0.99: %s' % (taylorxpoly, n))

    # Now we do the example of the ode with taylor polynomial
    cosseries1 = S.fps(S.cos(taylorxpoly)+0.99, x0=0).polynomial(n=n)
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
    FLT_MAX = 3.402823466e+38
    FLT_MIN = -FLT_MAX
    solve(FLT_MIN, FLT_MAX)
