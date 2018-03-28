#!/usr/bin/env python3

import simpy
import sympy as S
from src.ode import ODE


def ha(env, cstate=0):
    """This is the ha itself. This is very similar to the 'C' code that we
    generate from the haskell model, except that the whole thing is
    event drive.

    """
    delta = None               # None to cause failure
    # The continous variables used in this ha
    x = 2                       # The initial value
    loc1_ode = ODE(env, lvalue=S.sympify('diff(x(t))'),
                   rvalue=S.sympify('x(t)^2'))
    loc2_ode = ODE(env, S.sympify('diff(x(t))'),
                   S.sympify('-x(t)^3'))
    loc1_FT = False
    loc2_FT = False

    # XXX: DEBUG
    # print(loc1_ode, loc2_ode)

    # The computations in location1
    # Returning state, delta, value, loc1_FT, loc2_FT
    def location1(x, loc1_FT, loc2_FT):
        curr_time = env.now
        # The edge guard takes preference
        if x >= 10:
            print('%7.4f %7.4f %d' % (curr_time, x, cstate))
            print('b1')
            return 1, 0, x, None, True
        # The invariant
        elif x <= 10:
            # Compute the x value and print it.
            if not loc1_FT:
                x = loc1_ode.compute(x, curr_time)
                loc1_FT = True
            print('%7.4f %7.4f %d' % (curr_time, x, cstate))
            print('b2')
            # TODO: Call the ODE class that will give the delta back
            delta = loc1_ode.delta(x, (10 - x))
            print('predicted delta: ', delta)
            return 0, delta, x, False, None
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location1')

    # The computations in location2
    def location2(x, loc1_FT, loc2_FT):
        curr_time = env.now
        if x <= 1:
            print('%7.4f %7.4f %d' % (curr_time, x, cstate))
            print('b1')
            return 0, 0, x, True, None
        elif x >= 1:
            # TODO: Call the ODE class to get delta
            if not loc2_FT:
                x = loc2_ode.compute(x, curr_time)
            print('%7.4f %7.4f %d' % (curr_time, x, cstate))
            print('b2')
            delta = loc2_ode.delta(x, (1 - x))
            print('predicted delta: ', delta)
            return 1, delta, x, None, False
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location2')

    # The dictionary for the switch statement.
    switch_case = {
        0: location1,
        1: location2
    }

    while(True):
        cstate, delta, x, loc1_FT, loc2_FT = switch_case[cstate](x,
                                                                 loc1_FT,
                                                                 loc2_FT)
        # This should always be the final statement in this function
        yield env.timeout(delta)


def main():
    """
    """
    env = simpy.Environment()
    env.process(ha(env))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run()


if __name__ == '__main__':
    main()
