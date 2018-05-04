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
    # The constants
    EPI_TVP = 1.4506
    EPI_TV1M = 60.0
    EPI_TV2M = 1150.0

    EPI_TWP = 200.0

    EPI_TW1M = 60.0
    EPI_TW2M = 15.0

    EPI_TS1 = 2.7342
    EPI_TS2 = 16.0
    EPI_TFI = 0.11
    EPI_TO2 = 6
    EPI_TSO1 = 30.0181
    EPI_TSO2 = 0.9957

    EPI_TSI = 1.8875
    EPI_TWINF = 0.07
    EPI_THV = 0.3
    # EPI_THVM = 0.006
    # EPI_THVINF = 0.006
    # EPI_THW = 0.13
    # EPI_THWINF = 0.006
    # EPI_THSO = 0.13
    # EPI_THSI = 0.13
    # EPI_THO = 0.006

    EPI_KWM = 65.0
    EPI_KS = 2.0994
    EPI_KSO = 2.0458

    EPI_UWM = 0.03
    EPI_US = 0.9087
    EPI_UU = 1.55
    EPI_USO = 0.65

    jfi1 = 0.0
    jsi1 = 0.0

    jfi2 = 0.0
    jsi2 = 0.0

    jfi3 = 0.0

    stim = 1.0

    # The continous variables used in this ha
    u = 0.0
    v = 1.0
    s = 0.0
    w = 1.0
    tau = 0.0

    # This we are setting, but dReach searches in their case.
    EPI_TO1 = 0.007

    ut = S.sympify('u(t)')
    vt = S.sympify('v(t)')
    wt = S.sympify('w(t)')
    st = S.sympify('s(t)')

    loc0_ode_tau = ODE(env, S.sympify('diff(tau(t))'), S.sympify('1.0'),
                       ttol=10**-3, iterations=100)
    loc0_ode_u = ODE(env, S.sympify('diff(u(t))'),
                     S.sympify((stim - jfi1) - ((ut/EPI_TO1) + jsi1)),
                     ttol=10**-3, iterations=100)
    loc0_ode_w = ODE(env, S.sympify('diff(w(t))'),
                     S.sympify(((1.0 - (ut / EPI_TWINF) - wt) /
                                (EPI_TW1M + (EPI_TW2M - EPI_TW1M) *
                                 (1.0 / (1 + S.exp(-2*EPI_KWM
                                                   * (ut - EPI_UWM))))))),
                     ttol=10**-3, iterations=100)
    loc0_ode_v = ODE(env, S.sympify('diff(v(t))'),
                     S.sympify(((1.0 - vt)/EPI_TV1M)),
                     ttol=10**-3, iterations=100)
    loc0_ode_s = ODE(env, S.sympify('diff(s(t))'),
                     S.sympify((((1/(1+S.exp(-2 * EPI_KS *
                                             (ut - EPI_US)))) - st)/EPI_TS1)),
                     ttol=10**-3, iterations=100)

    loc0_FT = False

    loc1_ode_tau = ODE(env, S.sympify('diff(tau(t))'), S.sympify('1.0'),
                       ttol=10**-3, iterations=100)
    loc1_ode_u = ODE(env, S.sympify('diff(u(t))'),
                     S.sympify((stim - jfi2) - ((ut/EPI_TO2) + jsi2)),
                     ttol=10**-3, iterations=100)
    loc1_ode_w = ODE(env, S.sympify('diff(w(t))'),
                     S.sympify(((0.94-wt)/(EPI_TW1M + (EPI_TW2M - EPI_TW1M) *
                                           (1.0 /
                                            (1+S.exp(
                                                -2*EPI_KWM*(ut -
                                                            EPI_UWM))))))),
                     ttol=10**-3, iterations=100)
    loc1_ode_v = ODE(env, S.sympify('diff(v(t))'),
                     S.sympify((-vt/EPI_TV2M)),
                     ttol=10**-3, iterations=100)
    loc1_ode_s = ODE(env, S.sympify('diff(s(t))'),
                     S.sympify((((1/(1+S.exp(-2 * EPI_KS *
                                             (ut - EPI_US)))) - st)/EPI_TS1)),
                     ttol=10**-3, iterations=100)

    loc1_FT = False

    loc2_ode_tau = ODE(env, S.sympify('diff(tau(t))'), S.sympify('1.0'),
                       ttol=10**-3, iterations=100)
    loc2_ode_u = ODE(env, S.sympify('diff(u(t))'),
                     S.sympify((stim - jfi3) -
                               ((1.0/(EPI_TSO1+((EPI_TSO2 - EPI_TSO1) *
                                                (1/(1+S.exp(-2 * EPI_KSO *
                                                            (ut -
                                                             EPI_USO)))))))
                                + (0 - (wt * st)/EPI_TSI))),
                     ttol=10**-3, iterations=100, simplify_poly=True)
    loc2_ode_w = ODE(env, S.sympify('diff(w(t))'),
                     S.sympify((-wt/EPI_TWP)),
                     ttol=10**-3, iterations=100, simplify_poly=True)
    loc2_ode_v = ODE(env, S.sympify('diff(v(t))'),
                     S.sympify((-vt/EPI_TV2M)),
                     ttol=10**-3, iterations=100)
    loc2_ode_s = ODE(env, S.sympify('diff(s(t))'),
                     S.sympify((((1/(1+S.exp(-2 * EPI_KS *
                                             (ut - EPI_US)))) - st)/EPI_TS2)),
                     ttol=10**-3, iterations=100, simplify_poly=True)

    loc2_FT = False

    loc3_ode_tau = ODE(env, S.sympify('diff(tau(t))'), S.sympify('1.0'),
                       ttol=10**-3, iterations=100)
    loc3_ode_u = ODE(env, S.sympify('diff(u(t))'),
                     S.sympify((stim - (0 - vt * (ut - EPI_THV) *
                                        (EPI_UU - ut)/EPI_TFI)) -
                               ((1.0 / (EPI_TSO1+((EPI_TSO2 - EPI_TSO1)
                                                  * (1/(1+S.exp(-2*EPI_KSO *
                                                                (ut -
                                                                 EPI_USO)))))))
                                + (0 - (wt * st)/EPI_TSI))),
                     ttol=10**-3, iterations=100, simplify_poly=True)
    loc3_ode_w = ODE(env, S.sympify('diff(w(t))'),
                     S.sympify((-wt/EPI_TWP)),
                     ttol=10**-3, iterations=100)
    loc3_ode_v = ODE(env, S.sympify('diff(v(t))'),
                     S.sympify((-vt/EPI_TVP)),
                     ttol=10**-3, iterations=100)
    loc3_ode_s = ODE(env, S.sympify('diff(s(t))'),
                     S.sympify((((1/(1+S.exp(-2 * EPI_KS *
                                             (ut - EPI_US)))) - st)/EPI_TS2)),
                     ttol=10**-3, iterations=100, simplify_poly=True)

    loc3_FT = False

    # Location 0
    def location0(u, v, w, s, tau, loc0_FT, loc1_FT, loc2_FT,
                  loc3_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('u(t)'): u,
                S.sympify('v(t)'): v,
                S.sympify('w(t)'): w,
                S.sympify('s(t)'): s,
                S.sympify('tau(t)'): tau}
        # The edge guard takes preference
        if u >= 0.006:
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            return 1, 0, u, v, w, s, tau, None, True, None, None, curr_time
        # The invariant
        elif u <= 0.006:
            if not loc0_FT:
                u = loc0_ode_u.compute(vals,
                                       curr_time-prev_time)
                v = loc0_ode_v.compute(vals,
                                       curr_time-prev_time)
                w = loc0_ode_w.compute(vals,
                                       curr_time-prev_time)
                s = loc0_ode_s.compute(vals,
                                       curr_time-prev_time)
                tau = loc0_ode_tau.compute(vals,
                                           curr_time-prev_time)
                loc0_FT = True
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            if abs(u-0.006) > loc0_ode_u.vtol:
                delta = loc0_ode_u.delta(vals, quanta=(0.006-u),
                                         other_odes=[loc0_ode_s, loc0_ode_v,
                                                     loc0_ode_w, loc0_ode_tau])
            else:
                u = 0.006
                delta = 0
            return (0, delta, u, v, w, s, tau,
                    False, None, None, None, curr_time)
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 0')

    def location1(u, v, w, s, tau, loc0_FT, loc1_FT, loc2_FT,
                  loc3_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('u(t)'): u,
                S.sympify('v(t)'): v,
                S.sympify('w(t)'): w,
                S.sympify('s(t)'): s,
                S.sympify('tau(t)'): tau}
        # The edge guard takes preference
        if u >= 0.013:
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            return 2, 0, u, v, w, s, tau, None, True, None, None, curr_time
        # The invariant
        elif u <= 0.013:
            if not loc1_FT:
                u = loc1_ode_u.compute(vals,
                                       curr_time-prev_time)
                v = loc1_ode_v.compute(vals,
                                       curr_time-prev_time)
                w = loc1_ode_w.compute(vals,
                                       curr_time-prev_time)
                s = loc1_ode_s.compute(vals,
                                       curr_time-prev_time)
                tau = loc1_ode_tau.compute(vals,
                                           curr_time-prev_time)
                loc1_FT = True
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            if abs(u-0.013) > loc1_ode_u.vtol:
                delta = loc1_ode_u.delta(vals, quanta=(0.013-u),
                                         other_odes=[loc1_ode_s, loc1_ode_v,
                                                     loc1_ode_w, loc1_ode_tau])
            else:
                u = 0.013
                delta = 0
            return (1, delta, u, v, w, s, tau,
                    False, None, None, None, curr_time)
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 1')

    def location2(u, v, w, s, tau, loc0_FT, loc1_FT, loc2_FT,
                  loc3_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('u(t)'): u,
                S.sympify('v(t)'): v,
                S.sympify('w(t)'): w,
                S.sympify('s(t)'): s,
                S.sympify('tau(t)'): tau}
        # The edge guard takes preference
        if u >= 0.3:
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            return 3, 0, u, v, w, s, tau, None, True, None, None, curr_time
        # The invariant
        elif u <= 0.3:
            if not loc2_FT:
                u = loc2_ode_u.compute(vals,
                                       curr_time-prev_time)
                v = loc2_ode_v.compute(vals,
                                       curr_time-prev_time)
                w = loc2_ode_w.compute(vals,
                                       curr_time-prev_time)
                s = loc2_ode_s.compute(vals,
                                       curr_time-prev_time)
                tau = loc2_ode_tau.compute(vals,
                                           curr_time-prev_time)
                loc2_FT = True
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            if abs(u-0.3) > loc2_ode_u.vtol:
                delta = loc2_ode_u.delta(vals, quanta=(0.3-u),
                                         other_odes=[loc2_ode_s, loc2_ode_v,
                                                     loc2_ode_w, loc2_ode_tau])
            else:
                u = 0.3
                delta = 0
            return (2, delta, u, v, w, s, tau,
                    False, None, None, None, curr_time)
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 2')

    def location3(u, v, w, s, tau, loc0_FT, loc1_FT, loc2_FT,
                  loc3_FT, prev_time):
        curr_time = env.now
        vals = {S.sympify('u(t)'): u,
                S.sympify('v(t)'): v,
                S.sympify('w(t)'): w,
                S.sympify('s(t)'): s,
                S.sympify('tau(t)'): tau}
        # The edge guard takes preference
        if u >= 2.0:
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            return 3, 0, u, v, w, s, tau, None, True, None, None, curr_time
        # The invariant
        elif u <= 2.0:
            if not loc3_FT:
                u = loc3_ode_u.compute(vals,
                                       curr_time-prev_time)
                v = loc3_ode_v.compute(vals,
                                       curr_time-prev_time)
                w = loc3_ode_w.compute(vals,
                                       curr_time-prev_time)
                s = loc3_ode_s.compute(vals,
                                       curr_time-prev_time)
                tau = loc3_ode_tau.compute(vals,
                                           curr_time-prev_time)
                loc3_FT = True
            print('%7.4f: %7.4f %7.4f %7.4f %7.4f %7.4f' % (curr_time, u, v,
                                                            w, s, tau))
            if abs(u-2.0) > loc3_ode_u.vtol:
                delta = loc3_ode_u.delta(vals, quanta=(2.0-u),
                                         other_odes=[loc3_ode_s, loc3_ode_v,
                                                     loc3_ode_w, loc3_ode_tau])
            else:
                u = 2.0
                delta = 0
            return (3, delta, u, v, w, s, tau,
                    False, None, None, None, curr_time)
        else:
            raise RuntimeError('Reached unreachable branch'
                               ' in location 3')

    # The dictionary for the switch statement.
    switch_case = {
        0: location0,
        1: location1,
        2: location2,
        3: location3
    }

    prev_time = env.now
    while(True):
        (cstate, delta, u, v, w, s, tau,
         loc0_FT, loc1_FT, loc2_FT, loc3_FT,
         prev_time) = switch_case[cstate](u, v, w, s, tau,
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
    env.run(until=1.6)
    print('total steps: ', step)


if __name__ == '__main__':
    main()
