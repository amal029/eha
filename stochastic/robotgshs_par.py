#!/usr/bin/env python3
import sympy as S
import numpy
import mpmath as M
import matplotlib.pyplot as plt
from src.sdesolver import Compute


# Total simulation time
SIM_TIME = 1.2

# Defining the dynamics in different modes
# The constants in the HA
v = 4
wv = 0.1
e = 1e-1

# The length of the stochastic path
# p = 3
# R = 2**p

# Mode move
# We have drift and diffusion dynamics in all cases
dynamics = {'Move': {S.sympify('x(t)'): [v*wv*S.sympify('-sin(th(t))'),
                                         S.sympify('0')],
                     S.sympify('y(t)'): [v*wv*S.sympify('cos(th(t))'),
                                         S.sympify('0')],
                     S.sympify('th(t)'): [S.sympify('0'),
                                          S.sympify('0')],
                     S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Inner': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                          S.sympify(v)],
                      S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                          S.sympify(v)],
                      S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                      S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Outter': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                           S.sympify(v)],
                       S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                           S.sympify(v)],
                       S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                       S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Changetheta': {S.sympify('x(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('y(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('th(t)'): [S.sympify(2*v),
                                                 S.sympify('0')],
                            S.sympify('z(t)'): [
                                S.sympify('(x(t)**2+y(t)**2)*th(t)**3'),
                                S.sympify('0')]},
            'ThetaNochange': {S.sympify('x(t)'): [S.sympify('0'),
                                                  S.sympify('0')],
                              S.sympify('y(t)'): [S.sympify('0'),
                                                  S.sympify('0')],
                              S.sympify('th(t)'): [S.sympify(wv),
                                                   S.sympify('0')],
                              S.sympify('z(t)'): [S.sympify('0'),
                                                  S.sympify('0')]}}


# This is the main GSHA
class GSHS:
    """The generalised stochastic hybrid automaton
    """

    @staticmethod
    def __compute(x, y, th, z, t, dWts, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th,
                S.sympify('z(t)'): z}
        # Compute the dynamics in the state
        DM = dynamics[location]

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
        # XXX: T == Δ == δ*R (assumption)
        k = list(Dts.keys())
        T = min(*k) if len(k) > 1 else k[0]
        if T == numpy.inf:
            T, val = Compute.default_compute(DM, dWts, vars, t)
            return T, val.values()
        else:
            return T, Dts[T].values()

    @staticmethod
    def Move(x, y, th, z, t, dWts, first_time):
        """Location Move
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e):
            # Set the outputs
            # th = 0
            state = 1           # destination location is Inner
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - v**2 >= e):
            # th = S.pi
            state = 2           # destination is Outter
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            T, vars = GSHS.__compute(x, y, th, z, t, dWts, 'Move', [], None)
            return 0, T, vars, False

    @staticmethod
    def Inner(x, y, th, z, t, dWts, first_time):
        """Location Inner
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 >= -e) and (x**2 + y**2 - v**2 <= e):
            # Set the outputs
            # th = float(M.atan(y/x))
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - v**2 >= e):
            state = 2
            # th = S.pi
            return state, 0, (x, y, th, z), True
        else:
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            T, vars = GSHS.__compute(x, y, th, z, t, dWts, 'Inner', [g1], None)
            return 1, T, vars, False

    @staticmethod
    def Outter(x, y, th, z, t, dWts, first_time):
        """Location Outter
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= e) and (x**2 + y**2 - v**2 >= -e):
            # Set the outputs
            # th = float(M.atan(y/x))
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif x**2 + y**2 - v**2 <= -e:
            state = 1           # Destination Inner
            # th = 0
            return state, 0, (x, y, th, z), True
            pass
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            # g2 = S.sympify('z(t)') - Uz
            T, vars = GSHS.__compute(x, y, th, z, t, dWts, 'Outter', [g1],
                                     None)
            return 2, T, vars, False

    @staticmethod
    def Changetheta(x, y, th, z, t, dWts, first_time):
        """Changetheta location
        """
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if first_time else Uz
        if (x**2 + y**2 - v**2 <= e) and (x**2 + y**2 - v**2 >= -e):
            z = 0
            th = float(M.atan(y/x))
            state = 1           # Dest: ThetaNochange
            return state, 0, (x, y, th, z), True
        elif abs(z - Uz) <= e:
            z = 0
            th = th - Uz
            state = 0           # Dest Changetheta
            return state, 0, (x, y, th, z), True
        else:
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            T, vars = GSHS.__compute(x, y, th, z, t, dWts,
                                     'Changetheta', [g1], Uz)
            return 0, T, vars, False

    @staticmethod
    def ThetaNochange(x, y, th, z, t, dWts, first_time):
        if (x**2 + y**2 - v**2 <= -e) or (x**2 + y**2 - v**2 >= e):
            z = 0
            th = float(M.atan(y/x))
            state = 0           # Dest Changetheta
            return state, 0, (x, y, th, z), True
        else:
            T, vars = GSHS.__compute(x, y, th, z, t, dWts,
                                     'ThetaNochange', [], None)
            return 1, T, vars, False


def main(x, y, th, z, t):

    XY_loc = {
        0: GSHS.Move,
        1: GSHS.Inner,
        2: GSHS.Outter
    }
    XY_strloc = {
        0: 'Move',
        1: 'Inner',
        2: 'Outter',
    }

    THETA_loc = {
        0: GSHS.Changetheta,
        1: GSHS.ThetaNochange
    }

    THETA_strloc = {
        0: 'Changetheta',
        1: 'ThetaNochange'
    }

    # First compute the invariant to decide upon the location I should
    # be in.
    if abs(x**2 + y**2 - v**2) <= e:
        state1 = 0               # Move
    elif x**2 + y**2 - v**2 <= -e:
        state1 = 1               # Inner
    elif x**2 + y**2 - v**2 >= e:
        state1 = 2               # Outter
    else:
        raise Exception('Unknown state reached')

    state2 = 0                  # Always start THETA from change state

    # Print the outputs
    print('%.4f: Locs:(%s, %s), x:%s, y:%s, th:%s, z:%s'
          % (t, XY_strloc[state1], THETA_strloc[state2], x, y, th, z))

    FT1 = True
    FT2 = True
    xs = []
    ys = []
    ts = []
    xy2s = []
    while(True):
        # Create dWt
        dWts = {S.sympify('x(t)'): numpy.random.randn(Compute.R),
                S.sympify('y(t)'): numpy.random.randn(Compute.R),
                S.sympify('th(t)'): numpy.random.randn(Compute.R),
                S.sympify('z(t)'): numpy.zeros(Compute.R)}

        vars = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th,
                S.sympify('z(t)'): z}

        # Call the dynamics and run these until some time
        nstate1, T1, (x1, y1, th1, z1), FT1 = XY_loc[state1](x, y,
                                                             th, z, t, dWts,
                                                             FT1)
        nstate2, T2, (x2, y2, th2, z2), FT2 = THETA_loc[state2](x, y,
                                                                th, z, t, dWts,
                                                                FT2)
        if not FT1 and not FT2:
            # Intra-Intra
            if T2 <= T1:
                T = T2
                [x, y] = [Compute.EM(vars[i],
                                     dynamics[XY_strloc[state1]][i][0],
                                     dynamics[XY_strloc[state1]][i][1],
                                     T, T/Compute.R, dWts[i], vars, t)
                          for i in [S.sympify('x(t)'), S.sympify('y(t)')]]
                th, z = th2, z2
            elif T1 <= T2:
                T = T1
                [th, z] = [Compute.EM(vars[i],
                                      dynamics[THETA_strloc[state2]][i][0],
                                      dynamics[THETA_strloc[state2]][i][1],
                                      T, T/Compute.R, dWts[i], vars, t)
                           for i in [S.sympify('th(t)'), S.sympify('z(t)')]]
                x, y = x1, y1
        elif not FT1 and FT2:
            # Intra-Inter
            x, y, th, z = x, y, th2, z2
        elif FT1 and not FT2:
            # Inter-Intra
            x, y, th, z = x1, y1, th, z
        elif FT1 and FT2:
            # Inter-Inter
            x, y, th, z = x1, y1, th2, z2

        # Set the new states
        state1 = nstate1
        state2 = nstate2
        # Append for plotting
        xs.append(x)
        ys.append(y)
        ts.append(t)
        xy2s.append(x**2+y**2)
        t += min(T1, T2)

        # Print the outputs
        print('%.4f: Locs:(%s, %s), x:%s, y:%s, th:%s, z:%s'
              % (t, XY_strloc[state1], THETA_strloc[state2], x, y, th, z))

        if t >= SIM_TIME:
            break
    return xs, ys, xy2s, ts


if __name__ == '__main__':
    # the random seed
    Compute.ROOT_FUNC = 'mpmath'
    Compute.DEFAULT_STEP = 1e-1
    numpy.random.seed(0)
    # These are the initial values
    x = 1
    y = 1
    th = S.atan(y/x)
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
    plt.show()
