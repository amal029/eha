#!/usr/bin/env python3

import simpy
import sympy as S
from src.ode import ODE

step = 0


def ha(env, cstate=0):
    """This is the ha itself. This is very similar to the 'C' code that we
    generate from the haskell model, except that the whole thing is
    event drive.

    """
    delta = None               # None to cause failure
    # The continous variables used in this ha
    x = 20                     # The initial value
    loc1_xode = ODE(env, lvalue=S.sympify('diff(x(t))'),
                    rvalue=S.sympify('-0.25*x(t)'),
                    ttol=10**-3, iterations=100)
    loc2_xode = ODE(env, S.sympify('diff(x(t))'),
                    S.sympify('125-0.25*x(t)'),
                    ttol=10**-3, iterations=100)
    loc1_FT = False
    loc2_FT = False

    # The computations in location1
    # Returning state, delta, value, loc1_FT, loc2_FT
    def location1(x, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if x <= 18.5:
            print('%7.7f: %7.7f' % (curr_time, x))
            return 1, 0, x, None, True, curr_time
        # The invariant
        elif x >= 18.5:
            # Compute the x value and print it.
            if not loc1_FT:
                # All the dependent initial conditions
                x = loc1_xode.compute({S.sympify('x(t)'): x},
                                      curr_time-prev_time)
                loc1_FT = True
            print('%7.7f: %7.7f' % (curr_time, x))

            if abs(x-18.5) > loc1_xode.vtol:
                x_delta = loc1_xode.delta({S.sympify('x(t)'): x},
                                          quanta=(18.5-x))
            else:
                # If within the error bound just make it 10
                x = 18.5
                x_delta = 0
            return 0, x_delta, x, False, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location1')

    # The computations in location2
    def location2(x, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        if x >= 19.5:
            print('%7.7f: %7.7f' % (curr_time, x))
            return 0, 0, x, True, None, curr_time
        elif x <= 19.5:
            if not loc2_FT:
                x = loc2_xode.compute({S.sympify('x(t)'): x},
                                      curr_time-prev_time)
            print('%7.7f: %7.7f' % (curr_time, x))

            if abs(x-19.5) > loc2_xode.vtol:
                x_delta = loc2_xode.delta({S.sympify('x(t)'): x},
                                          quanta=(19.5 - x))
            else:
                # If within error bound then just make it the level.
                x = 19.5
                x_delta = 0

            return 1, x_delta, x, None, False, curr_time
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
        global step
        step += 1
        yield env.timeout(delta)


def main():
    """
    """
    env = simpy.Environment()
    env.process(ha(env))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=0.5)
    print('total steps: ', step)


if __name__ == '__main__':
    main()
