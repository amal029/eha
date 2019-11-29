#!/usr/bin/env python3
from scipy.optimize import minimize
import sympy as S
import sympy.abc as ABC
from math import factorial


class Solver(object):
    """The solver for computing the integration step size.

    """
    epsilon = 1e-12
    t = ABC.t
    h = ABC.h
    n = 1
    DEBUG = 0

    def __init__(self, n=1, epsilon=1e-12, DEBUG=0):
        Solver.epsilon = epsilon
        assert n >= 1, "n < 1"
        Solver.n = n
        Solver.DEBUG = DEBUG

    @staticmethod
    def getLipschitz(fun, x0, bounds):
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
        resmax = minimize(lambdify_wrapper, x0, bounds=bounds)
        return abs(resmax.fun)

    @staticmethod
    def get_polynomial(k, tokens, vals_at_tn):
        # Insert initial value in tokens
        tokens.insert(0, vals_at_tn[k])
        poly = sum([c*Solver.h**p/factorial(p)
                    for c, p in zip(tokens, range(Solver.n+1))])
        return poly

    @staticmethod
    def get_vals_at_tn_h(poly, vals_at_tn, h):
        """tokens are the taylor derivative terms for k, excluding the constant
        term k is x(t)

        """
        # First replace all x(t) → initial values
        for k, i in vals_at_tn.items():
            poly = poly.replace(k, i)
        # Now replace Solver.h → h
        poly = poly.replace(Solver.h, h)
        # Now eval it
        return poly.evalf()

    @staticmethod
    def build_tokens(cont_var, odes):
        """cont_var: name of the function, e.g., x(t), you want the tokens for.

        odes are all xs(t) derivative terms, of all continuous vars, e.g.,
        {x(t): x(t)+y(t)+1, y(t): 1,...}

        """
        tokens = [odes[cont_var]]
        for _ in range(len(tokens), Solver.n):
            tokens.append(Solver.build(tokens, odes))
        return tokens

    @staticmethod
    def build(tokens, odes):
        # This gets the next derivative
        slope = tokens[-1].diff(Solver.t)
        # 2.) Replace Derivative(deps(t), t) → exprs
        for i, k in odes.items():
            slope = slope.replace(i, k)
        return slope

    @staticmethod
    def delta(values, h, curr_time):
        """Gives the time step needed in the taylor polynomial to correctly
        bound the local truncation error given the step size

        args:

        values: Continuous variable tuple

        ({all cont_var tokens obtained from Solver.build_tokens},
        {intial values of all continous variables})

        h: The step size that you want to take

        curr_time: The current time Tₙ


        return: h (seconds), such that f⁽ⁿ⁺¹)(η)/(n+1)!⋆(h⁽ⁿ⁺¹⁾) ≤
        Solver.epsilon, n ≡ 2, η ∈ (Tₙ, Tₙ + h),

        """
        assert(len(values) == 2)

        vals_at_tn = values[1]
        all_ode_taylor = values[0]
        odes = {k: i[0] for k, i in all_ode_taylor.items()}

        # XXX: Now compute the value of continous vars at Tₙ + h
        vals_at_tn_h = {k: Solver.get_vals_at_tn_h(
            Solver.get_polynomial(k, i, vals_at_tn), vals_at_tn, h)
                        for k, i in all_ode_taylor.items()}

        # Now compute the bounds for each continous variable
        bounds = {k: (min(i, vals_at_tn_h[k]), max(i, vals_at_tn_h[k]))
                  for k, i in vals_at_tn.items()}

        # Replace x(t) → x, will be needed for S.lambdify
        func_to_var = {k: S.Symbol(str(k.func)) for k in bounds.keys()}

        # Now we need to get the lipschitz constant for all continous
        # vars at the n+1 term
        taylor_n1_term = dict()
        for k, i in all_ode_taylor.items():
            slope = Solver.build(i, odes)
            # XXX: x(t) → x, etc
            for kk, ii in func_to_var.items():
                slope = slope.replace(kk, ii)

            # XXX: This should be negative, because we want to maximize
            # it.
            taylor_n1_term[k] = -slope

        # These are the lambdified python functions
        lambdified = {k: S.lambdify((list(func_to_var.values()) +
                                     [Solver.t]), i)
                      for k, i in taylor_n1_term.items()}

        # Now get the lipschitz constants for each ode
        lips = {k: Solver.getLipschitz(i, bounds[k],
                                       (list(bounds.values()) +
                                        [(curr_time, curr_time+h)]))
                for k, i in lambdified.items()}

        # Now check if lagrange error is satisfied
        facn_1 = factorial(Solver.n+1)
        sat = {k: (l*(h**(Solver.n+1))/facn_1) <= Solver.epsilon
               for k, l in lips.items()}

        # XXX: If all satisfy it then send back the h as the next
        # integration step
        if all(list(sat.values())):
            return h
        else:
            # XXX: Compute the new integration step size
            numerator = Solver.epsilon * facn_1
            steps = {k: (numerator/l)**(1/Solver.n+1)
                     for k, l in lips.items()}
            return min(list(steps.values()))


def example1():
    x = S.sympify('x(t)')       # The function

    # The odes for all continuous variables
    odes = {x: S.sympify('1')}

    # Initial values
    vals_at_tn = {x: 3}

    # The guard expression
    g = S.sympify('x(t) - 10')

    # Initiaise the solver
    solver = Solver()

    # Get the tokens for x
    dict_tokens = {x: solver.build_tokens(x, odes)}
    tokens = [j for i in dict_tokens.values() for j in i]

    # First get the polynomial expression from tokens
    xps = {x: solver.get_polynomial(x, tokens, vals_at_tn)
           for x in odes}

    # Substitute the polynomial in the guard expression
    og = g
    for x in xps:
        g = g.replace(x, xps[x])

    # XXX: This can be made efficient later on using scipy
    hs = (S.Poly(g).all_roots())
    hs = [h for h in hs if h >= 0]
    h = min(hs)

    # Now we have a starting "h", which we think we can jump as the
    # integration step.

    # Test to see if this "h" satisfies lagrange error. curr_time shoul
    # come using the event driven engine.
    h = solver.delta((dict_tokens, vals_at_tn), h, 0)

    # XXX: This example is guaranteed to have a single integration step,
    # which meets the guard.
    # Final result
    print('Initial value:', vals_at_tn)
    res = {k: solver.get_vals_at_tn_h(x, vals_at_tn, h)
           for k, x in xps.items()}
    print('Final value:', res, 'with guard: %s, odes: %s, integration step: %s'
          % (og, odes, h))


if __name__ == '__main__':
    example1()
