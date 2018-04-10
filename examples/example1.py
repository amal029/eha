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
                   rvalue=S.sympify('sin(cos(x(t)))+2'),
                   ttol=10**-3, iterations=100)
    loc2_ode = ODE(env, S.sympify('diff(x(t))'),
                   S.sympify('-2*x(t)'),
                   ttol=10**-3, iterations=100)
    loc1_FT = False
    loc2_FT = False

    # XXX: DEBUG
    # print(loc1_ode, loc2_ode)

    # The computations in location1
    # Returning state, delta, value, loc1_FT, loc2_FT
    def location1(x, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if x >= 5:
            print('%7.4f %7.4f' % (curr_time, x))
            # import sys
            # sys.exit(1)
            return 1, 0, x, None, True, curr_time
        # The invariant
        elif x <= 5:
            # Compute the x value and print it.
            if not loc1_FT:
                x = loc1_ode.compute({S.sympify('x(t)'): x},
                                     curr_time-prev_time)
                loc1_FT = True
            print('%7.7f %7.7f' % (curr_time, x))
            # XXX: Call the ODE class that will give the delta back iff
            # the calculated "x" is greater than the error.
            if abs(x-5) > loc1_ode.vtol:
                delta = loc1_ode.delta({S.sympify('x(t)'): x},
                                       quanta=(5-x))
            else:
                # If within the error bound just make it 10
                x = 5
                delta = 0
            return 0, delta, x, False, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location1')

    # The computations in location2
    def location2(x, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        if x <= 1:
            print('%7.7f %7.7f' % (curr_time, x))
            return 0, 0, x, True, None, curr_time
        elif x >= 1:
            # TODO: Call the ODE class to get delta
            if not loc2_FT:
                x = loc2_ode.compute({S.sympify('x(t)'): x},
                                     curr_time-prev_time)
            print('%7.7f %7.7f' % (curr_time, x))
            # If the output is outside the error margin then bother
            # recomputing a new delta.
            if abs(x-1) > loc2_ode.vtol:
                delta = loc2_ode.delta({S.sympify('x(t)'): x},
                                       quanta=(1 - x))
            else:
                # If within error bound then just make it the level.
                x = 1
                delta = 0
            return 1, delta, x, None, False, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location2')

    # The dictionary for the switch statement.
    switch_case = {
        0: location1,
        1: location2
    }

    prev_time = env.now
    while(True):
        (cstate, delta, x,
         loc1_FT, loc2_FT, prev_time) = switch_case[cstate](x,
                                                            loc1_FT,
                                                            loc2_FT,
                                                            prev_time)
        # This should always be the final statement in this function
        yield env.timeout(delta)


def main():
    """
    """
    env = simpy.Environment()
    env.process(ha(env))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=5)


if __name__ == '__main__':
    main()
