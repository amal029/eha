#!/usr/bin/env python3

import sympy as S
import numpy
import matplotlib.pyplot as plt
from src.sdesolver import Compute


# Total simulation time
SIM_TIME = 2000.0

# error
e = 1e-1

# The length of the stochastic path
# p = 3
# R = 2**p

# Constants
kp = 1
ku = 0.001
kd = 0.01
kb = 0.01

dynamics = {'X0': {S.sympify('x(t)'): [S.sympify(kp), S.sympify(0)],
                   S.sympify('z(t)'): [kb*S.sympify('x(t)'), S.sympify(0)]},
            'X1': {S.sympify('x(t)'): [S.sympify(0), S.sympify(0)],
                   S.sympify('z(t)'): [S.sympify(ku), S.sympify(0)]},
            'D': {S.sympify('x(t)'): [-kd*S.sympify('x(t)'), S.sympify(0)],
                  S.sympify('z(t)'): [S.sympify(0), S.sympify(0)]}}


class GSHS:
    def __compute(x, z, t, dWts, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x(t)'): x, S.sympify('z(t)'): z}
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
    def X0(x, z, t, dWts, FT):
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if FT else Uz
        if (x >= 1) and abs(z - Uz) <= e:
            # Destination X1
            x = x-1
            state = 1
            z = 0
            return state, 0, (x, z), True
        else:
            # g = S.sympify('x(t)')*-kd + kp - 1
            T, vars = GSHS.__compute(x, z, t, dWts, 'X0', [], Uz)
            return 0, T, vars, False

    @staticmethod
    def X1(x, z, t, dWts, FT):
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if FT else Uz
        if abs(z - Uz) <= e:
            # Destination X0
            state = 0
            z = 0
            return state, 0, (x, z), True
        else:
            # g = S.sympify('x(t)')*-kd + kp - 1
            T, vars = GSHS.__compute(x, z, t, dWts, 'X1', [], Uz)
            return 1, T, vars, False

    @staticmethod
    def D(x, z, t, dWts, FT):
        T, vars = GSHS.__compute(x, z, t, dWts, 'D', [], None)
        return 0, T, vars, False


def main(x, z, t):
    X_loc = {
        0: GSHS.X0,
        1: GSHS.X1
    }
    X_strloc = {
        0: 'X0',
        1: 'X1',
    }
    D_loc = {0: GSHS.D}
    D_strloc = {0: 'D'}

    state = 0
    state2 = 0

    # Print the outputs
    print('%.4f: Locs:(%s, %s), x:%s, z:%s' %
          (t, X_strloc[state], D_strloc[state2], x, z))

    FT1 = True
    FT2 = True

    xs = []
    ts = []

    while(True):
        # Create dWt
        dWts = {S.sympify('x(t)'): numpy.zeros(Compute.R),
                S.sympify('z(t)'): numpy.zeros(Compute.R)}

        vars = {S.sympify('x(t)'): x,
                S.sympify('z(t)'): z}

        # Call the dynamics and run these until some time
        state, T1, (x1, z1), FT1 = X_loc[state](x, z, t, dWts, FT1)
        state2, T2, (x2, _), FT2 = D_loc[state2](x, z, t, dWts, FT2)

        # print(x1, x2)
        # print('T1, T2', T1, T2)

        # Update depending upon which one is smaller
        if not FT1 and not FT2:
            if T2 <= T1:
                T = T2
                [x1, z] = [Compute.EM(vars[i],
                                      dynamics[X_strloc[state]][i][0],
                                      dynamics[X_strloc[state]][i][1],
                                      T, T/Compute.R, dWts[i], vars, t)
                           for i in [S.sympify('x(t)'), S.sympify('z(t)')]]
            elif T1 <= T2:
                T = T1
                [x2] = [Compute.EM(vars[i],
                                   dynamics[D_strloc[state2]][i][0],
                                   dynamics[D_strloc[state2]][i][1],
                                   T, T/Compute.R, dWts[i], vars, t)
                        for i in [S.sympify('x(t)')]]
                # print('new x2:', x2)
                z = z1
            # Combination like Esterel with `+'
            x = x1 + x2 - vars[S.sympify('x(t)')]
        elif FT1 and not FT2:
            z = z1
            x = x1          # Because T1 = 0
        else:
            raise Exception('HA2 cannot take a transition')

        xs.append(x)
        ts.append(t)
        t += min(T1, T2)

        print('%.4f: Locs:(%s, %s), x:%s, z:%s' %
              (t, X_strloc[state], D_strloc[state2], x, z))

        if t >= SIM_TIME:
            return xs, ts


def set_plt_params():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12


if __name__ == '__main__':
    # Compute.DEFAULT_STEP = 1
    numpy.random.seed(0)
    x = 0
    z = 0
    t = 0
    xs, ts = main(x, z, t)
    print('count:', len(ts))
    set_plt_params()
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x$ (units)', fontweight='bold')
    plt.savefig('/tmp/gene.pdf', bbox_inches='tight')
    plt.show()
