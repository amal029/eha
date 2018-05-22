#!/usr/bin/env python3

import simpy
import sympy as S
import sys
from src.ode import ODE

# The variable holding the number of steps taken during simulation
step = 0


def ha(env, cstate=0):
    """This is the ha itself. This is very similar to the 'C' code that we
    generate from the haskell model, except that the whole thing is
    event drive.

    """
    T1 = 10
    T2 = 10
    thM = 20
    thm = 5
    vr = 10.5
    v1 = -1.3
    v2 = -2.7
    assert(T1 == T2)

    delta = None               # None to cause failure
    # The continous variables used in this ha
    x = T1                       # clock1 variable
    y = T2                       # clock2 variable
    th = 11.5                    # The reactor temperature

    # You need vtol here, because of floating point error.
    loc0_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc0_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc0_ode_th = ODE(env, S.sympify('diff(th(t))'), S.sympify(vr),
                      ttol=10**-3, iterations=100, vtol=10**-10)
    loc0_FT = False

    loc1_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc1_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc1_ode_th = ODE(env, S.sympify('diff(th(t))'), S.sympify(v1),
                      ttol=10**-3, iterations=100, vtol=10**-10)
    loc1_FT = False

    loc2_ode_x = ODE(env, S.sympify('diff(x(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc2_ode_y = ODE(env, S.sympify('diff(y(t))'), S.sympify('1.0'),
                     ttol=10**-3, iterations=100)
    loc2_ode_th = ODE(env, S.sympify('diff(th(t))'), S.sympify(v2),
                      ttol=10**-3, iterations=100, vtol=10**-10)
    loc2_FT = False

    # Location 3 is reactor shutdown
    loc3_FT = False

    # Location 0
    def location0(x, y, th, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        vals = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th}
        curr_time = env.now
        # The edge guard takes preference
        if th == thM and x >= T1:
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            return 1, 0, x, y, th, None, True, None, None, curr_time
        elif th == thM and y >= T2:
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            return 2, 0, x, y, th, None, None, True, None, curr_time
        elif th == thM and x < T1 and y < T2:
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            return 3, 0, x, y, th, None, None, None, True, curr_time
        # The invariant
        elif th <= thM:
            if not loc0_FT:
                x = loc0_ode_x.compute(vals, curr_time-prev_time)
                y = loc0_ode_y.compute(vals, curr_time-prev_time)
                th = loc0_ode_th.compute(vals, curr_time-prev_time)
                loc0_FT = True
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            if abs(th-thM) > loc0_ode_th.vtol:
                deltath = loc0_ode_th.delta(vals, quanta=(thM-th))
            else:
                th = thM
                deltath = 0
            return 0, deltath, x, y, th, False, None, None, None, curr_time
        else:
            print('th:', th)
            raise RuntimeError('Reached unreachable branch'
                               ' in location 0')

    def location1(x, y, th, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        vals = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th}
        curr_time = env.now
        # The edge guard takes preference
        if th == thm:
            x = 0               # Reset
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            return 0, 0, x, y, th, True, None, None, None, curr_time
        # The invariant
        elif th >= thm:
            if not loc1_FT:
                x = loc1_ode_x.compute(vals, curr_time-prev_time)
                y = loc1_ode_y.compute(vals, curr_time-prev_time)
                th = loc1_ode_th.compute(vals, curr_time-prev_time)
                loc1_FT = True
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            if abs(th-thm) > loc1_ode_th.vtol:
                deltath = loc1_ode_th.delta(vals, quanta=(thm-th))
            else:
                th = thm
                deltath = 0
            return 1, deltath, x, y, th, False, None, None, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 1')

    def location2(x, y, th, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        vals = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th}
        curr_time = env.now
        # The edge guard takes preference
        if th == thm:
            y = 0               # Reset
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            return 0, 0, x, y, th, True, None, None, None, curr_time
        # The invariant
        elif th >= thm:
            if not loc2_FT:
                x = loc2_ode_x.compute(vals, curr_time-prev_time)
                y = loc2_ode_y.compute(vals, curr_time-prev_time)
                th = loc2_ode_th.compute(vals, curr_time-prev_time)
                loc2_FT = True
            print('%7.4f %7.4f %7.4f %7.4f' % (curr_time, x, y, th))
            if abs(th-thm) > loc2_ode_th.vtol:
                deltath = loc2_ode_th.delta(vals, quanta=(thm-th))
            else:
                th = thm
                deltath = 0
            return 2, deltath, x, y, th, False, None, None, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 2')

    def location3(x, y, th, loc0_FT, loc1_FT, loc2_FT, loc3_FT, prev_time):
        global step
        print('total steps: ', step)
        # Done
        sys.exit(1)

    # The dictionary for the switch statement.
    switch_case = {
        0: location0,
        1: location1,
        2: location2,
        3: location3
    }

    prev_time = env.now
    while(True):
        (cstate, delta, x, y, th,
         loc0_FT, loc1_FT, loc2_FT, loc3_FT,
         prev_time) = switch_case[cstate](x, y, th,
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
