#!/usr/bin/env python3
import sympy as S
import numpy
import mpmath as M
import matplotlib.pyplot as plt
from sdesolver import Compute


# Total simulation time
SIM_TIME = 1.0

# Defining the dynamics in different modes
# The constants in the HA
v = 4
wv = 0.1
e = 1e-1

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

    def __compute(x, y, th, z, t, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th,
                S.sympify('z(t)'): z}
        # Compute the dynamics in the state
        DM = dynamics[location]

        # Create dWt
        dWts = {S.sympify('x(t)'): numpy.random.randn(Compute.R),
                S.sympify('y(t)'): numpy.random.randn(Compute.R),
                S.sympify('th(t)'): numpy.random.randn(Compute.R),
                S.sympify('z(t)'): numpy.zeros(Compute.R)}

        Dts = dict()
        # XXX: This is for computing the spontaneous jump if any
        Dz, vals = Compute.rate_compute(left=S.sympify('z(t)'),
                                        right=DM[S.sympify('z(t)')],
                                        deps=DM, vars=vars, T=t, Uz=Uz,
                                        dWts=dWts)
        Dts[Dz] = vals

        # XXX: Accounting for the guards
        # This one does not have the spontaneous jump in it.
        for i in guards:
            Dt, Dval = Compute.guard_compute(expr=i, deps=DM, vars=vars, T=t,
                                             dWts=dWts, Dz=Dz)
            Dts[Dt] = Dval

        # XXX: dz might be np.inf if there is no spontaneous output
        # This is the step size we will take
        # XXX: T == Δ == δ*Compute.R (assumption)
        k = list(Dts.keys())
        T = min(*k) if len(k) > 1 else k[0]
        if T == numpy.inf:
            T, val = Compute.default_compute(DM, dWts, vars, t)
            return T, val.values()
        else:
            return T, Dts[T].values()

    def Move(x, y, th, z, t, first_time):
        """Location Move
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - (v**2 - e) <= 0):
            # Set the outputs
            th, z = 0, 0
            state = 1           # destination location is Inner
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - (v**2 + e) >= 0):
            th, z = S.pi, 0
            state = 2           # destination is Outter
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            T, vars = __compute(x, y, th, z, t, 'Move', [], None)
            return 0, T, vars, False

    def Inner(x, y, th, z, t, first_time):
        """Location Inner
        """
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if first_time else Uz
        # First compute the outgoing edges and take them
        if x**2 + y**2 - v**2 >= -e:
            # Set the outputs
            th = float(M.atan(y/x))
            z = 0
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        # elif (x**2 + y**2 >= (v**2 + e)):
        #     raise Exception
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            T, vars = __compute(x, y, th, z, t, 'Inner', [g1], Uz)
            return 1, T, vars, False

    def Outter(x, y, th, z, t, first_time):
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if first_time else Uz
        # First compute the outgoing edges and take them
        if x**2 + y**2 - v**2 <= e:
            # Set the outputs
            th = float(M.atan(y/x))
            z = 0
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            # g2 = S.sympify('z(t)') - Uz
            T, vars = __compute(x, y, th, z, t, 'Outter', [g1], Uz)
            return 2, T, vars, False

    def Changetheta(x, y, th, z, t, first_time):
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if first_time else Uz
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - (v**2 - e) <= 0) and abs(z - Uz) <= e:
            # Set the outputs
            z = 0
            state = 1           # destination location is Inner
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - (v**2 + e) >= 0) and abs(z - Uz) <= e:
            z = 0
            state = 2           # destination is Outter
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - (v**2 - e)
            g2 = S.sympify('x(t)**2 + y(t)**2') - (v**2 + e)
            T, vars = __compute(x, y, th, z, t, 'Changetheta', [g1, g2], Uz)
            return 3, T, vars, False

    locations = {
        0: Move,
        1: Inner,
        2: Outter,
        3: Changetheta
    }
    strloc = {
        0: 'M',
        1: 'I',
        2: 'O',
        3: 'CT'
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
    first_time = True
    xs = []
    ys = []
    ts = []
    xy2s = []
    while(True):
        # Call the dynamics and run these until some time
        state, T, (x, y, th, z), first_time = locations[state](x, y, th, z, t,
                                                               first_time)
        xs.append(x)
        ys.append(y)
        ts.append(t)
        xy2s.append(x**2+y**2)
        t += T
        # Print the outputs
        print('%f: state:%s, x:%f, y:%f, xy**2:%f, diff:%s, th:%f, z:%f' %
              (t, strloc[state], x, y, (x**2+y**2), (x**2+y**2)-v**2, th, z))

        if t >= SIM_TIME:
            break
    return xs, ys, xy2s, ts


if __name__ == '__main__':
    # the random seed
    numpy.random.seed(0)
    # These are the initial values
    x = 1
    y = 1
    th = float(M.atan(y/x))
    z = 0
    t = 0
    xs, ys, xy2s, ts = main(x, y, th, z, t)
    print('Count:', len(ts))
    plt.style.use('ggplot')
    # plt.subplot(211)
    # plt.plot(xs, ys, marker='^')
    # plt.xlabel('X (units)', fontweight='bold')
    # plt.ylabel('Y (units)', fontweight='bold')
    # plt.subplot(212)
    plt.plot(ts, xy2s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x^2+y^2$ (units)', fontweight='bold')
    plt.savefig('/tmp/robot.pdf', bbox_inches='tight')
    # plt.show()
