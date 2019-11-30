#!/usr/bin/env python3
import sympy as S
import numpy as N
import simpy
from src.solver import Solver


step = 0                        # The number of integration steps


def example1(env, solver, cstate=0):
    """Example of a ha being solved using the new technique.

    """
    # TODO: Need to fix so that this works.
    # odes = {x.diff(solver.t): (x+5)*solver.t}

    # First initialise the continuous variables.

    x = S.sympify('x(t)')       # The continuous variable

    # Initial values
    # We have to add the modulo operator
    vals_at_tn = {x: ((5/2)*S.pi) % (2 * S.pi)}

    def build_gtn(gtn, vals_at_tn):
        for k, i in vals_at_tn.items():
            gtn = gtn.replace(k, i)
        # Replace the "t" if it is there in the guard
        gtn = gtn.replace(solver.t, env.now).evalf()
        return gtn

    def build_gth(og, vals_at_tn, xps):
        for x in xps:
            og = og.replace(x, xps[x])
        # Replace x(t) → x(Tₙ)
        # Replace t → Tₙ
        for k, v in vals_at_tn.items():
            og = og.replace(k, v)
        og = og.replace(solver.t, env.now).evalf()
        return og

    def get_gh(og):
        # Now get the h, where you think the guard holds.
        nsoln = N.roots(S.poly(og).all_coeffs())
        nsoln = nsoln[N.isreal(nsoln)]
        nsoln = nsoln[N.where(nsoln >= 0)]
        # If you cannot find a root then set it to infinity
        h = N.real(N.min(nsoln)) if nsoln.size != 0 else N.inf
        return h

    # Returning state, delta, values, loc's_FT
    def location1(x, vals_at_tn):
        # The odes for all continuous variables in location1
        odes = {x.diff(solver.t): S.sympify('1')}

        # Get the tokens for x
        dict_tokens = {x: solver.build_tokens(x, odes)}

        # First get the polynomial expression from tokens
        xps = {x: solver.get_polynomial(x, tokens, vals_at_tn)
               for x, tokens in dict_tokens.items()}

        # Now check of the guard is satisfied, if yes jump
        # The guard expression
        g = S.sympify('cos(x(t)) + 0.99')

        # Need to taylor_expand, because there is a cos
        g = solver.taylor_expand(g)

        # Compute the value of g(t) at Tₙ
        gtn = build_gtn(g.copy(), vals_at_tn)
        # print('guard at Tₙ:', gtn)

        if (abs(gtn) <= solver.epsilon):           # If zero crossing happens
            # We can make a jump to the next location
            return 1, 0, vals_at_tn
        else:
            # This is the intra-location transition

            # Guard1 g(t) = 0
            og = build_gth(g.copy(), vals_at_tn, xps)
            # print('guard1:', og)
            h = get_gh(og)

            # TODO: Guard2 g(t) - 2×g(Tₙ) = 0
            og2 = og - 2*gtn
            # print('guard2:', og2)
            h2 = get_gh(og2)

            # Take the minimum from amongst the two
            h = min(h, h2) if h2 is not N.inf else h

            assert h is not N.inf, 'Cannot find h from guards'

            h = solver.delta((dict_tokens, vals_at_tn), h, 0)

            # Now compute the new values for continuous variables
            vals_at_tn = {k: solver.get_vals_at_tn_h(x, vals_at_tn, h)
                          for k, x in xps.items()}
            return 0, h, vals_at_tn

    def location2(x, vals_at_tn):
        # The odes for all continuous variables in location1
        odes = {x.diff(solver.t): S.sympify('-1')}

        # Get the tokens for x
        dict_tokens = {x: solver.build_tokens(x, odes)}

        # First get the polynomial expression from tokens
        xps = {x: solver.get_polynomial(x, tokens, vals_at_tn)
               for x, tokens in dict_tokens.items()}

        # Now check of the guard is satisfied, if yes jump
        # The guard expression
        g = S.sympify('cos(x(t)) - 0.99')
        g = solver.taylor_expand(g)

        # Compute the value of g(t) at Tₙ
        gtn = build_gtn(g.copy(), vals_at_tn)
        # print('guard at Tₙ:', gtn)

        if (abs(gtn) <= solver.epsilon):           # If zero crossing happens
            # We can make a jump to the next location
            return 0, 0, vals_at_tn
        else:
            # This is the intra-location transition

            # Guard1 g(t) = 0
            og = build_gth(g.copy(), vals_at_tn, xps)
            # print('guard1:', og)
            h = get_gh(og)

            # TODO: Guard2 g(t) - 2×g(Tₙ) = 0
            og2 = og - 2*gtn
            # print('guard2:', og2)
            h2 = get_gh(og2)

            # Take the minimum from amongst the two
            h = min(h, h2) if h2 is not N.inf else h

            assert h is not N.inf, 'Cannot find h from guards'

            h = solver.delta((dict_tokens, vals_at_tn), h, 0)

            # Now compute the new values for continuous variables
            vals_at_tn = {k: solver.get_vals_at_tn_h(x, vals_at_tn, h)
                          for k, x in xps.items()}
            return 1, h, vals_at_tn

    # The dictionary for the switch statement.
    switch_case = {
        0: location1,
        1: location2
    }

    # The initial values at time 0
    print('%f, %s' % (env.now, vals_at_tn))

    # Now start running the system until all events are done or
    # simulation time is over.
    while(True):
        (cstate, delta, vals_at_tn) = switch_case[cstate](x, vals_at_tn)
        # The new values of the continuous variables
        if delta != 0:
            print('%f: %s' % (env.now+delta, vals_at_tn))
        # This should always be the final statement in this function
        global step
        step += 1
        yield env.timeout(delta)


def main():
    # Initiaise the solver
    solver = Solver(epsilon=1e-6, NUM_TERMS=10)
    # For higher precision, default is 5

    env = simpy.Environment()
    env.process(example1(env, solver))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=5.0)
    print('total steps: ', step)


if __name__ == '__main__':
    main()
