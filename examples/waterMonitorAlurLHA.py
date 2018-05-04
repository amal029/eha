#!/usr/bin/env python3

import simpy
import sympy as S
from src.ode import ODE

# The variable holding the number of steps taken during simulation
step = 0


def ha(env, cstate=0):
    """This is the ha itself. This is very similar to the 'C' code that we
    generate from the haskell model, except that the whole thing is
    event drive.

    """
    delta = None               # None to cause failure
    # The continous variables used in this ha
    x = 0                       # clock variable
    y = 1                       # The initial value

    loc0_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc0_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc0_FT = False

    loc1_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc1_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc1_FT = False

    loc2_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc2_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('-2.0'),
                     ttol=10**-3, iterations=100)
    loc2_FT = False

    loc3_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc3_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('-2.0'),
                     ttol=10**-3, iterations=100)
    loc3_FT = False

    # Location 0
    def location0(x, y, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if y == 10:
            x = 0               # Reset relation on x
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            return 1, 0, x, y, None, True, None, None, curr_time
        # The invariant
        elif y <= 10:
            if not loc0_FT:
                x = loc0_ode_x.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                y = loc0_ode_y.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                loc0_FT = True
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            # XXX: Call the ODE class that will give the delta back iff
            # the calculated "x" is greater than the error.
            if abs(y-10) > loc0_ode_y.vtol:
                delta = loc0_ode_y.delta({S.sympify('x(t)'): x,
                                          S.sympify('y(t)'): y}, quanta=(10-y))
            else:
                # If within the error bound just make it 10
                y = 10
                delta = 0
            return 0, delta, x, y, False, None, None, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 0')

    # Location 1
    def location1(x, y, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if x == 2:
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            return 2, 0, x, y, None, None, True, None, curr_time
        # The invariant
        elif x <= 2:
            if not loc1_FT:
                x = loc1_ode_x.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                y = loc1_ode_y.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                loc1_FT = True
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            if abs(x-2) > loc1_ode_x.vtol:
                delta = loc1_ode_x.delta({S.sympify('x(t)'): x,
                                          S.sympify('y(t)'): y},
                                         quanta=(2-x))
            else:
                # If within the error bound just make it 2
                x = 2
                delta = 0
            return 1, delta, x, y, None, False, None, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 1')

    # Location 2
    def location2(x, y, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if y == 5:
            x = 0               # Reset relation on x
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            return 3, 0, x, y, None, None, None, True, curr_time
        # The invariant
        elif y >= 5:
            if not loc2_FT:
                x = loc2_ode_x.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                y = loc2_ode_y.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                loc2_FT = True
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            if abs(y-5) > loc2_ode_y.vtol:
                delta = loc2_ode_y.delta({S.sympify('x(t)'): x,
                                          S.sympify('y(t)'): y},
                                         quanta=(5-y))
            else:
                # If within the error bound just make it 10
                y = 5
                delta = 0
            return 2, delta, x, y, None, None, False, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 2')

    # Location 3
    def location3(x, y, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        curr_time = env.now
        # The edge guard takes preference
        if x == 2:
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            return 0, 0, x, y, None, None, True, None, curr_time
        # The invariant
        elif x <= 2:
            if not loc3_FT:
                x = loc3_ode_x.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                y = loc3_ode_y.compute({S.sympify('x(t)'): x,
                                        S.sympify('y(t)'): y},
                                       curr_time-prev_time)
                loc3_FT = True
            print('%7.4f %7.4f %7.4f' % (curr_time, x, y))
            if abs(x-2) > loc3_ode_x.vtol:
                delta = loc3_ode_x.delta({S.sympify('x(t)'): x,
                                          S.sympify('y(t)'): y},
                                         quanta=(2-x))
            else:
                # If within the error bound just make it 2
                x = 2
                delta = 0
            return 3, delta, x, y, False, None, None, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 2')
    # The dictionary for the switch statement.
    switch_case = {
        0: location0,
        1: location1,
        2: location2,
        3: location3
    }

    prev_time = env.now
    while(True):
        (cstate, delta, x, y,
         loc0_FT, loc1_FT, loc2_FT, loc3_FT,
         prev_time) = switch_case[cstate](x, y,
                                          loc0_FT,
                                          loc1_FT,
                                          loc2_FT,
                                          loc3_FT,
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
    env.run(until=30)
    print('total steps: ', step)


if __name__ == '__main__':
    main()
