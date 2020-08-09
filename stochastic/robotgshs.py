#!/usr/bin/env python3
import sympy as S
import numpy as np


class Compute:
    """This is the main adaptive time step computation class

    """
    # Static error bound
    epsilon = 1e-7

    @staticmethod
    def var_compute(left=None, right=None, deps=None, dWt=None, vars=None,
                    T=0):
        pass

    @staticmethod
    def rate_compute(left=None, right=None, deps=None, Uz=None, vars=None,
                     T=0):
        pass

    @staticmethod
    def guard_compute(expr=None, deps=None, vars=None, T=0):
        pass

    @staticmethod
    def subs(expr=None, vars=None):
        pass


# Total simulation time
SIM_TIME = 2

# Defining the dynamics in different modes
# The constants in the HA
v = 4
wv = 0.1
e = 1e-7

# The length of the stochastic path
p = 3
R = 2**p

# Mode move
# We have drift and diffusion dynamics in all cases
dynamics = {'Move': {S.sympify('x(t)'): [v*wv*S.sympify('-sin(th(t))'),
                                         S.sympify('0')],
                     S.sympify('y(t)'): [v*wv*S.sympify('cos(th(t))'),
                                         S.sympify('0')],
                     S.sympify('th(t)'): [S.sympify('wv').subs('wv', wv),
                                          S.sympify('0')],
                     S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Inner': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                          S.sympify('v').subs('v', v)],
                      S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                          S.sympify('v').subs('v', v)],
                      S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                      S.sympify('z(t)'): [S.sympify('x(t)'), S.sympify('0')]},
            'Outter': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                           S.sympify('v').subs('v', v)],
                       S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                           S.sympify('v').subs('v', v)],
                       S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                       S.sympify('z(t)'): [S.sympify('x(t)'), S.sympify('0')]},
            'Changetheta': {S.sympify('x(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('y(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('th(t)'): [S.sympify('v').subs('v', v),
                                                 S.sympify('0')],
                            S.sympify('z(t)'): [S.sympify('th(t)**3'),
                                                S.sympify('0')]}}


# This is the main GSHA
def main(x, y, th, z, t):
    """The generalised stochastic hybrid automaton
    """
    vars = {S.sympify('x(t)'): x,
            S.sympify('y(t)'): y,
            S.sympify('th(t)'): th,
            S.sympify('z(t)'): z}

    def __compute(location=None, guards=None, Uz=None):
        global th, z, x, y, t
        # Compute the dynamics in state Move
        DM = dynamics[location]
        # XXX: Should we give the random path from here?
        # Create dWt
        dWx = np.random.randn(R)
        dx = Compute.var_compute(left=S.sympify('x(t)'),
                                 right=DM[S.sympify('x(t)')], deps=DM,
                                 dWt=dWx, vars=vars, T=t)
        dWy = np.random.randn(R)
        dy = Compute.var_compute(left=S.sympify('y(t)'),
                                 right=DM[S.sympify('y(t)')], deps=DM,
                                 dWt=dWy, vars=vars, T=t)
        dWth = np.random.randn(R)
        dth = Compute.var_compute(left=S.sympify('th(t)'),
                                  right=DM[S.sympify('th(t)')],
                                  deps=DM, dWt=dWth, vars=vars,
                                  T=t)

        # XXX: This is for computing the spontaneous jump if any
        dz = Compute.rate_compute(left=S.sympify('z(t)'),
                                  right=DM[S.sympify('z(t)')], deps=DM,
                                  vars=vars, T=t, Uz=None)

        # XXX: Accounting for the guards
        dgs = [Compute.guard_compute(expr=i, deps=DM, vars=vars, T=t)
               for i in guards]

        # XXX: dz might be np.inf if there is no spontaneous output
        # This is the step size we will take
        # XXX: T == Δ == δ*R (assumption)
        T = min(dx, dy, dz, dth, *dgs)

        # Now compute the actual values of x, y, th, z using Euler
        # maruyama for step size T First substitute the value of
        # vars with the current values in the argument and then
        # eval.
        x += (Compute.subs(DM[S.sympify('x(t)')][0], vars) * T
              + (np.sqrt(T/R) * Compute.subs([S.sympify('x(t)')][1])
                 * np.sum(dWx)))

        y += (Compute.subs(DM[S.sympify('y(t)')][0], vars) * T
              + (np.sqrt(T/R) * Compute.subs([S.sympify('y(t)')][1])
                 * np.sum(dWy)))

        th += (Compute.subs(DM[S.sympify('th(t)')][0], vars) * T
               + (np.sqrt(T/R) * Compute.subs([S.sympify('th(t)')][1])
                  * np.sum(dWth)))

        z += (Compute.subs(DM[S.sympify('z(t)')][0], vars) * T)
        return T

    def Move(first_time):
        """Location Move
        """
        global th, z, x, y, t
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e):
            # Set the outputs
            th, z = 0, 0
            state = 1           # destination location is Inner
            return state, 0, True
        elif (x**2 + y**2 - v**2 >= e):
            th, z = S.pi, 0
            state = 2           # destination is Outter
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - (v**2 - e)
            g2 = S.sympify('x(t)**2 + y(t)**2') - (v**2 + e)
            T = __compute('Move', [g1, g2], None)
            return 0, T, False

    def Inner(first_time):
        """Location Inner
        """
        global th, z, x, y, t
        if first_time:
            # This is the spontaneous jump condition
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if (abs(x**2 + y**2 - v**2) <= e):
            # Set the outputs
            th = np.arctan(y/x)
            z = 0
            state = 0           # destination location is Move
            return state, 0, True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            g2 = S.sympify('z(t)') - Uz
            T = __compute('Inner', [g1, g2], Uz)
            return 1, T, False

    def Outter(first_time):
        global th, z, x, y, t
        if first_time:
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if abs(x**2 + y**2 - v**2) <= e:
            # Set the outputs
            th = np.arctan(y/x)
            z = 0
            state = 0           # destination location is Move
            return state, 0, True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            g2 = S.sympify('z(t)') - Uz
            T = __compute('Outter', [g1, g2], Uz)
            return 2, T, False

    def Changetheta(first_time):
        global th, z, x, y, t
        if first_time:
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e) and abs(z - Uz) <= e:
            # Set the outputs
            z = 0
            state = 1           # destination location is Inner
            return state, 0, True
        elif (x**2 + y**2 - v**2 >= e) and abs(z - Uz) <= e:
            z = 0
            state = 2           # destination is Outter
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('z(t)') - Uz
            T = __compute('Changetheta', [g1], Uz)
            return 3, T, False

    locations = {
        0: Move,
        1: Inner,
        2: Outter,
        3: Changetheta
    }
    strloc = {
        0: 'Move',
        1: 'Inner',
        2: 'Outter',
        3: 'Changetheta'
    }

    # First compute the invariant to decide upon the location I should
    # be in.
    if abs(x**2 + y**2 - v**2) <= e:
        state = 0
    elif x**2 + y**2 - v**2 <= -e:
        state = 1
    elif x**2 + y**2 - v**2 >= e:
        state = 2
    else:
        raise Exception('Unknown state reached')

    # Print the outputs
    print('%.4f: state:%s, x:%s, y:%s, th:%s, z:%s' % (t, strloc[state], x, y,
                                                       th, z))
    while(True):
        # Call the dynamics and run these until some time
        first_time = True
        state, T, first_time = locations[state](first_time)

        t += T
        # Print the outputs
        print('%.4f: state:%s, x:%s, y:%s, th:%s, z:%s' % (t, strloc[state],
                                                           x, y, th, z))

        if t >= SIM_TIME:
            break


if __name__ == '__main__':
    # These are the initial values
    x = 1
    y = 0
    th = 0
    z = 0
    t = 0
    main(x, y, th, z, t)
