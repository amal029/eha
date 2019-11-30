from scipy.optimize import minimize
import sympy as S
import sympy.abc as ABC
from math import factorial
import mpmath as M


class Solver(object):
    """The solver for computing the integration step size.
    n: The number of terms in ODE → Taylor exapansion
    NUM_TERMS: The number of terms in transcendental → Taylor expansion
    epsilon: The max value error allowed
    DEBUG: No effect, yet

    """
    epsilon = 1e-12
    t = ABC.t
    h = ABC.h
    n = 1
    NUM_TERMS = 5
    DEBUG = 0
    TRIG_FUNCS = [S.sin, S.cos, S.tan, S.cot, S.sec, S.csc]
    INV_TRIG_FUNCS = [S.asin, S.acos, S.atan, S.acot, S.asec, S.acsc, S.atan2]
    HYPERBOLIC_FUNCS = [S.sinh, S.cosh, S.tanh, S.coth, S.sech, S.csch]
    INV_HYPERBOLIC_FUNCS = [S.asinh, S.acosh, S.atanh, S.acoth, S.asech,
                            S.acsch]
    EXP_LOG = [S.exp, S.ln]
    TRANSCEDENTAL_FUNCS = (TRIG_FUNCS + INV_TRIG_FUNCS + HYPERBOLIC_FUNCS +
                           INV_HYPERBOLIC_FUNCS + EXP_LOG)

    def __init__(self, n=1, NUM_TERMS=10, epsilon=1e-12, DEBUG=0):
        Solver.epsilon = epsilon
        assert n >= 1, "n < 1"
        Solver.n = n
        Solver.DEBUG = DEBUG
        Solver.NUM_TERMS = NUM_TERMS

    @staticmethod
    def taylor_expand(expr, around=0):
        assert around == 0, 'Taylor expansion only works around 0 for now'
        if expr.args is ():
            return expr
        args = [Solver.taylor_expand(a, around) for a in expr.args]
        if expr.func in Solver.TRANSCEDENTAL_FUNCS:
            if len(args) != 1:
                raise RuntimeError('Cannot create a taylor series '
                                   'approximation of: ', expr)
            else:
                # XXX: Build the polynomial for arg
                coeffs = M.taylor(expr.func, around, Solver.NUM_TERMS)
                # print(coeffs)
                coeffs = [(S.Mul(float(a), S.Mul(*[args[0]
                                                   for i in range(c)])))
                          for c, a in enumerate(coeffs)][::-1]
                # print(coeffs)
                return S.Add(*coeffs)
        else:
            return expr.func(*args)

    @staticmethod
    def getLipschitz(fun, x0, bounds):
        """args:
        fun: The function whose lipschitz constant is needed
        bounds: Sequence of (min, max) pairs for each element in x. None is
        used to specify no bound.

        return: The lipschitz constant L for the function if one exists

        """

        # import inspect
        # print(inspect.getsource(fun))

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
        tokens = tokens.copy()
        # Insert initial value in tokens
        tokens.insert(0, vals_at_tn[k])
        poly = sum([c*Solver.h**p/factorial(p)
                    for c, p in zip(tokens, range(Solver.n+1))])
        return poly

    @staticmethod
    def get_vals_at_tn_h(poly, vals_at_tn, h, curr_time):
        """tokens are the taylor derivative terms for k, excluding the constant
        term k is x(t)

        """
        # First replace all x(t) → initial values
        for k, i in vals_at_tn.items():
            poly = poly.replace(k, i)
        # Replace all Solver.t with curr_time
        poly = poly.replace(Solver.t, curr_time)
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
        odes = {k: Solver.taylor_expand(i) for k, i in odes.items()}
        tokens = [odes[cont_var.diff(Solver.t)]]
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
        {intial values of all continuous variables})

        h: The step size that you want to take

        curr_time: The current time Tₙ


        return: h (seconds), such that f⁽ⁿ⁺¹)(η)/(n+1)!⋆(h⁽ⁿ⁺¹⁾) ≤
        Solver.epsilon, n ≡ 2, η ∈ (Tₙ, Tₙ + h),

        """
        assert(len(values) == 2)

        vals_at_tn = values[1]
        all_ode_taylor = values[0]
        odes = {k.diff(Solver.t): i[0] for k, i in all_ode_taylor.items()}

        # XXX: Now compute the value of continuous vars at Tₙ + h
        vals_at_tn_h = {k: Solver.get_vals_at_tn_h(
            Solver.get_polynomial(k, i, vals_at_tn), vals_at_tn, h, curr_time)
                        for k, i in all_ode_taylor.items()}

        # Now compute the bounds for each continuous variable
        bounds = {k: (min(i, vals_at_tn_h[k]), max(i, vals_at_tn_h[k]))
                  for k, i in vals_at_tn.items()}
        # print('bounds:', bounds)
        x0s = [bounds[k][1] for k in bounds]

        # Replace x(t) → x, will be needed for S.lambdify
        func_to_var = {k: S.Symbol(str(k.func)) for k in bounds.keys()}

        # Now we need to get the lipschitz constant for all continuous
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
        # print('n+1 derivatives:', taylor_n1_term)

        # These are the lambdified python functions
        lambdified = {k: S.lambdify((list(func_to_var.values()) +
                                     [Solver.t]), i)
                      for k, i in taylor_n1_term.items()}

        # Now get the lipschitz constants for each ode
        lips = {k: Solver.getLipschitz(i, (x0s + [curr_time+h]),
                                       (list(bounds.values()) +
                                        [(curr_time, curr_time+h)]))
                for k, i in lambdified.items()}
        # print('lips:', lips)

        # Now check if lagrange error is satisfied
        facn_1 = factorial(Solver.n+1)
        # print({k: (l*(h**(Solver.n+1))/facn_1) for k, l in lips.items()},
        #       '<=', Solver.epsilon, '?')
        sat = {k: (l*(h**(Solver.n+1))/facn_1) <= Solver.epsilon
               for k, l in lips.items()}

        # TODO: Add the condition to check that the series
        # converges.
        while True:
            if all(list(sat.values())):
                break
            else:
                # Just half the given step size until it meets the
                # requrements.
                h = h/2
                sat = {k: (l*(h**(Solver.n+1))/facn_1) <= Solver.epsilon
                       for k, l in lips.items()}
        return h
