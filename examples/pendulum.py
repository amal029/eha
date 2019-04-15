#!/usr/bin/env python3

import simpy
import sympy as S
import math
import sys
from src.ode import ODE

step = 0


# Normalizing the angle
def normalize_theta(theta):
    # This always gives a positive answer in anti clock-wise direction.
    return (theta % (2*(math.pi)))


def ha(env, cstate=0):
    """This is the ha itself. This is very similar to the 'C' code that we
    generate from the haskell model, except that the whole thing is
    event drive.

    """
    delta = None               # None to cause failure

    # Some constants
    K1 = 10
    K2 = -40

    # The continous variables used in this ha
    th = math.pi/2                       # The initial value

    loc1_ode_th = ODE(env, S.sympify('diff(th(t))'),
                      S.sympify(K1),
                      ttol=10**-2, iterations=1000)
    loc2_ode_th = ODE(env, S.sympify('diff(th(t))'),
                      S.sympify(K2),
                      ttol=10**-2, iterations=1000)
    loc1_FT = False
    loc2_FT = False

    # XXX: DEBUG
    # print(loc1_ode, loc2_ode)

    # The computations in location1
    # Returning state, delta, value, loc1_FT, loc2_FT
    def location1(th, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('th(t)'): th}

        # TODO: Try to remove this later on
        # Currently, cos(th(t)) <= V, does not work
        gvals = {S.sympify('th'): S.sympify(th)}

        # The edge guard takes preference
        g1 = S.sympify('cos(th) <= -0.9')
        thq = None
        # XXX: This should all be generated by the compiler
        if g1.is_Relational:
            if ((len(g1.args[0].args) == 1) and (g1.args[1].is_Number) and
                (g1.args[0].func in ODE.TRANSCEDENTAL_FUNCS)):
                soln = min(S.solve(g1.args[0]-g1.args[1]))
                # DEBUG
                print(g1, ':', soln, 'normalize_theta: ', normalize_theta(th))
                thq = soln - normalize_theta(th)
                sys.exit(1)
            else:
                raise RuntimeError('Cannot create taylor from: ', g1)
            pass
        else:
            raise RuntimeError('Guards can only be relations: ', g1)

        if (g1.subs(gvals)):
            print('%7.4f: %7.4f' % (curr_time, th))
            return 1, 0, th, None, True, curr_time

        # The location invariant
        elif (True):
            # Compute the x value and print it.
            if not loc1_FT:
                th = loc1_ode_th.compute(vals, curr_time-prev_time)
                loc1_FT = True
            print('%7.4f: %7.4f' % (curr_time, th))

            # TODO: FIXTHIS
            dth = 0, 0
            # Here 3.2 should be obtained by solving the guard condition
            if abs(th-3.2) > loc1_ode_th.vtol:
                dth = loc1_ode_th.delta(vals, quanta=(3.2-th),
                                        other_odes=[])
            else:
                th = 3.2
                dth = 0

            return 0, dth, th, False, None, curr_time
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location1')

    # Location 2 is end state in this example.
    def location2(th, loc1_FT, loc2_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('th(t)'): th}

        # The edge guard takes preference
        # FIXME:
        if (math.cos()):
            print('%7.4f: %7.4f' % (curr_time, th))
            return 0, 0, th, True, None, curr_time

        # The location invariant
        elif (True):
            # Compute the th value and print it.
            if not loc2_FT:
                th = loc1_ode_th.compute(vals, curr_time-prev_time)
                loc2_FT = True
            print('%7.4f: %7.4f' % (curr_time, th))

            # TODO: FIXTHIS
            dth = 0, 0
            # Here 3.2 should be obtained by solving the guard condition
            if abs(th-3.2) > loc2_ode_th.vtol:
                dth = loc2_ode_th.delta(vals, quanta=(3.2-th),
                                        other_odes=[])
            else:
                th = 3.2
                dth = 0

            return 1, dth, th, None, False, curr_time
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
        (cstate, delta, th,
         loc1_FT, loc2_FT, prev_time) = switch_case[cstate](th, loc1_FT,
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
    # sys.setrecursionlimit(2000)
    env = simpy.Environment()
    env.process(ha(env))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=5)


if __name__ == '__main__':
    main()
