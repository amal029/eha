#!/usr/bin/env python3

import simpy
import sympy as S
import sys
from src.ode import ODE

step = 0


def ha(env, cstate=0):
    """This is the ha itself. This is very similar to the 'C' code that we
    generate from the haskell model, except that the whole thing is
    event drive.

    """
    delta = None               # None to cause failure

    # Some constants
    v1 = 30
    v2 = -10.0
    le = 1

    # The continous variables used in this ha
    x = 0                       # The initial value
    y = 1                       # The initial value
    th = 0                       # The initial value
    ph = 1                       # The initial value

    loc1_ode_x = ODE(env, lvalue=S.sympify('diff(x(t))'),
                     rvalue=S.sympify(S.sympify('cos(th(t))')*v1),
                     ttol=10**-2, iterations=1000)
    loc1_ode_y = ODE(env, S.sympify('diff(y(t))'),
                     S.sympify(S.sympify('sin(th(t))')*v1),
                     ttol=10**-2, iterations=1000)
    loc1_ode_th = ODE(env, S.sympify('diff(th(t))'),
                      S.sympify((S.sympify('tan(ph(t))')/le)*v1),
                      ttol=10**-2, iterations=1000)
    loc1_ode_ph = ODE(env, S.sympify('diff(ph(t))'),
                      S.sympify(v2),
                      ttol=10**-2, iterations=1000)
    loc1_FT = False
    loc2_FT = False

    # XXX: DEBUG
    # print(loc1_ode, loc2_ode)

    # The computations in location1
    # Returning state, delta, value, loc1_FT, loc2_FT
    def location1(x, y, th, ph, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th,
                S.sympify('ph(t)'): ph}

        # The edge guard takes preference
        if ((y >= 1.8 and x <= 2.8) or (y <= 0.8 and x <= 2.8)):
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f' % (curr_time, x,
                                                      y, th, ph))
            return 1, 0, x, y, th, ph, None, True, curr_time
        # The invariant
        elif not ((y >= 1.8 and x <= 2.8) or (y <= 0.8 and x <= 2.8)):
            # Compute the x value and print it.
            if not loc1_FT:
                x = loc1_ode_x.compute(vals, curr_time-prev_time)
                y = loc1_ode_y.compute(vals, curr_time-prev_time)
                th = loc1_ode_th.compute(vals, curr_time-prev_time)
                ph = loc1_ode_ph.compute(vals, curr_time-prev_time)
                loc1_FT = True
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f' % (curr_time, x,
                                                      y, th, ph))
            dx, dy = 0, 0
            # if abs(x-3.2) > loc1_ode_x.vtol:
            #     dx = loc1_ode_x.delta(vals, quanta=(3.2-x),
            #                           other_odes=[loc1_ode_y, loc1_ode_th,
            #                                       loc1_ode_ph])
            # else:
            #     x = 3.2
            #     dx = 0

            if abs(x-2.8) > loc1_ode_x.vtol:
                dx = loc1_ode_x.delta(vals, quanta=(2.8-x),
                                      other_odes=[loc1_ode_y, loc1_ode_th,
                                                  loc1_ode_ph])
            else:
                x = 2.8
                dx = 0

            if abs(y-0.8) > loc1_ode_x.vtol:
                dy = loc1_ode_y.delta(vals, quanta=(abs(0.8-y)),
                                      other_odes=[loc1_ode_x, loc1_ode_th,
                                                  loc1_ode_ph])
            else:
                y = 0.8
                dy = 0

            if abs(y-1.8) > loc1_ode_x.vtol:
                # Purposely relaxing this value, else Python does not
                # recurse correctly!
                dy = min(loc1_ode_y.delta(vals, quanta=(1.8-y),
                                          other_odes=[loc1_ode_x, loc1_ode_th,
                                                      loc1_ode_ph]), dy)
            else:
                y = 1.8
                dy = 0
            return 0, min(dx, dy), x, y, th, ph, False, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location1')

    # Location 2 is end state in this example.
    def location2(x, y, th, ph, loc1_FT, loc2_FT, prev_time):
        global step
        print('total steps: ', step)
        # Done
        sys.exit(1)

    # The dictionary for the switch statement.
    switch_case = {
        0: location1,
        1: location2
    }

    prev_time = env.now
    while(True):
        (cstate, delta, x, y, th, ph,
         loc1_FT, loc2_FT, prev_time) = switch_case[cstate](x, y, th, ph,
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
    # Need this for this example.
    sys.setrecursionlimit(2000)
    env = simpy.Environment()
    env.process(ha(env))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=0.07)


if __name__ == '__main__':
    main()
