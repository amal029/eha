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
    x = 1                       # The initial value
    y = 0                       # The initial value
    loc1_xode = ODE(env, lvalue=S.sympify('diff(x(t))'),
                    rvalue=S.sympify('y(t)+x(t)+1'),
                    ttol=10**-3, iterations=100)
    loc1_yode = ODE(env, lvalue=S.sympify('diff(y(t))'),
                    rvalue=S.sympify('x(t)^2'),
                    ttol=10**-3, iterations=1000)
    loc2_xode = ODE(env, S.sympify('diff(x(t))'),
                    S.sympify('-2*y(t)'),
                    ttol=10**-3, iterations=100)
    loc2_yode = ODE(env, S.sympify('diff(x(t))'),
                    S.sympify('-x(t)+1'),
                    ttol=10**-3, iterations=100)
    loc1_FT = False
    loc2_FT = False

    # XXX: DEBUG
    # print(loc1_ode, loc2_ode)

    # The computations in location1
    # Returning state, delta, value, loc1_FT, loc2_FT
    def location1(x, y, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if x >= 5 and y >= 3:
            print('%7.4f %7.4f' % (curr_time, x))
            # import sys
            # sys.exit(1)
            return 1, 0, x, None, True, curr_time
        # The invariant
        elif x <= 5 and y <= 3:
            # Compute the x value and print it.
            if not loc1_FT:
                # All the dependent initial conditions
                x = loc1_xode.compute({S.sympify('y(t)'): y,
                                       S.sympify('x(t)'): x},
                                      curr_time-prev_time)
                y = loc1_yode.compute({S.sympify('y(t)'): y,
                                       S.sympify('x(t)'): x},
                                      curr_time-prev_time)
                loc1_FT = True
            print('%7.7f %7.7f %7.7f' % (curr_time, x, y))

            if abs(x-5) > loc1_xode.vtol:
                x_delta = loc1_xode.delta({S.sympify('y(t)'): y,
                                           S.sympify('x(t)'): x},
                                          quanta=(5-x),
                                          other_odes=[loc1_yode])
                # DEBUG
                print('xδ: ', x_delta)
            else:
                # If within the error bound just make it 10
                x = 5
                x_delta = 0

            if abs(y-3) > loc1_yode.vtol:
                y_delta = loc1_yode.delta({S.sympify('y(t)'): y,
                                           S.sympify('x(t)'): x},
                                          quanta=(3-y),
                                          other_odes=[loc1_xode])
                # DEBUG
                print('yδ: ', y_delta)
            else:
                # If within the error bound just make it 10
                y = 3
                y_delta = 0
            # DEBUG
            print('min δ: ', min(x_delta, y_delta))
            return 0, min(x_delta, y_delta), (x, y), False, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location1')

    # The computations in location2
    def location2(x, y, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        if x <= 1 and y <= 1:
            print('%7.7f %7.7f' % (curr_time, x))
            return 0, 0, x, True, None, curr_time
        elif x >= 1 and y >= 1:
            # TODO: Call the ODE class to get delta
            if not loc2_FT:
                x = loc2_xode.compute({S.sympify('y(t)'): y,
                                       S.sympify('x(t)'): x},
                                      curr_time-prev_time)
                y = loc2_yode.compute({S.sympify('y(t)'): y,
                                       S.sympify('x(t)'): x},
                                      curr_time-prev_time)
            print('%7.7f %7.7f %7.7f' % (curr_time, x, y))

            if abs(x-1) > loc2_xode.vtol:
                x_delta = loc2_xode.delta({S.sympify('y(t)'): y,
                                           S.sympify('x(t)'): x},
                                          quanta=(1 - x),
                                          other_odes=[loc2_yode])
            else:
                # If within error bound then just make it the level.
                x = 1
                x_delta = 0

            if abs(y-1) > loc2_yode.vtol:
                y_delta = loc2_yode.delta({S.sympify('y(t)'): y,
                                           S.sympify('x(t)'): x},
                                          quanta=(1 - y),
                                          other_odes=[loc2_xode])
            else:
                # If within error bound then just make it the level.
                y = 1
                y_delta = 0

            return 1, min(x_delta, y_delta), (x, y), None, False, curr_time
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
        (cstate, delta, (x, y),
         loc1_FT, loc2_FT, prev_time) = switch_case[cstate](x, y,
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
