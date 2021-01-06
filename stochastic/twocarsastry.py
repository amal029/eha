#!/usr/bin/env python3

import sympy as S
import numpy as np
import matplotlib.pyplot as plt
from src.sdesolver import Compute


# Total simulation time
SIM_TIME = 20

# error
e = 1e-1

# Constants
vx1 = 10
v2 = 5
a1 = 4

# Some distances
d0 = 10
d1 = 7
d2 = 4
d3 = 1

dynamics = {'C': {S.sympify('x1(t)'): [S.sympify(vx1), S.sympify(1)],
                  S.sympify('v1(t)'): [S.sympify(0), S.sympify(0)],
                  S.sympify('x2(t)'): [S.sympify(v2), S.sympify(0)]},
            'K': {S.sympify('x1(t)'): [S.sympify(v2), S.sympify(1)],
                  S.sympify('v1(t)'): [S.sympify(0), S.sympify(0)],
                  S.sympify('x2(t)'): [S.sympify(v2), S.sympify(0)]},
            'B': {S.sympify('x1(t)'): [S.sympify('v1(t)'), S.sympify(0)],
                  S.sympify('v1(t)'): [S.sympify(-a1), S.sympify(0)],
                  S.sympify('x2(t)'): [S.sympify(v2), S.sympify(0)]}}


class GSHS:
    def __compute(x1, x2, v1, t, dWts, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x1(t)'): x1,
                S.sympify('x2(t)'): x2,
                S.sympify('v1(t)'): v1}
        # Compute the dynamics in the state
        DM = dynamics[location]

        Dts = dict()

        # XXX: Accounting for the guards
        # This one does not have the spontaneous jump in it.
        for i in guards:
            Dt, Dval = Compute.guard_compute(expr=i, deps=DM, vars=vars, T=t,
                                             dWts=dWts, Dz=np.inf)
            Dts[Dt] = Dval

        # XXX: dz might be np.inf if there is no spontaneous output
        # This is the step size we will take
        # XXX: T == Δ == δ*R (assumption)
        k = list(Dts.keys())
        T = min(*k) if len(k) > 1 else k[0] if len(k) == 1 else np.inf
        if T == np.inf:
            T, val = Compute.default_compute(DM, dWts, vars, t)
            return T, val.values()
        else:
            return T, Dts[T].values()

    @staticmethod
    def C(x1, x2, v1, t, dWts):
        if abs(x2 - x1 - d2) <= e:
            state = 2           # Destination K
            return state, 0, (x1, x2, v1)
        else:
            g = S.sympify('x2(t) - x1(t)') - d2
            T, vars = GSHS.__compute(x1, x2, v1, t, dWts, 'C', [g])
            return 1, T, vars

    @staticmethod
    def K(x1, x2, v1, t, dWts):
        if abs(x2 - x1 - d1) <= e:
            state = 1           # state C
            return state, 0, (x1, x2, v1)
        elif abs(x2 - x1 - d3) <= e:
            # Destination B
            state = 3
            return state, 0, (x1, x2, v1)
        else:
            g1 = S.sympify('x2(t) - x1(t)') - d1
            g2 = S.sympify('x2(t) - x1(t)') - d3
            T, vars = GSHS.__compute(x1, x2, v1, t, dWts, 'K', [g1, g2])
            return 2, T, vars

    @staticmethod
    def B(x1, x2, v1, t, dWts):
        if abs(x2 - x1 - d0) <= e:
            state = 1           # destination C
            return state, 0, (x1, x2, v1)
        else:
            g = S.sympify('x2(t) - x1(t)') - d0
            T, vars = GSHS.__compute(x1, x2, v1, t, dWts, 'B', [g])
            return 3, T, vars


def main(x1, x2, v1, t):
    X_loc = {
        1: GSHS.C,
        2: GSHS.K,
        3: GSHS.B
    }
    X_strloc = {
        1: 'C',
        2: 'K',
        3: 'B'
    }

    if x2 - x1 > d2:
        state = 1
    elif x2 - x1 > d3 and x2 - x1 < d1:
        state = 2
    else:
        state = 3

    # Print the outputs
    print('%.4f: Loc:%s, (x1, x2, v1):(%s, %s, %s)' % (t, X_strloc[state],
                                                       x1, x2, v1))

    xs = [(x1, x2, v1)]
    ts = [t]
    while(True):
        # Create dWt
        dWts = {S.sympify('x1(t)'): np.random.randn(Compute.R),
                S.sympify('x2(t)'): np.zeros(Compute.R),
                S.sympify('v1(t)'): np.zeros(Compute.R)}

        # Call the dynamics and run these until some time
        state, T, (x1, x2, v1) = X_loc[state](x1, x2, v1, t, dWts)

        xs.append((x1, x2, v1))
        t += T
        ts.append(t)

        print('%.4f: Loc:%s, (x1, x2, v1):(%s, %s, %s)' % (t, X_strloc[state],
                                                           x1, x2, v1))
        if t >= SIM_TIME:
            return xs, ts


def set_plt_params():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    # plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12


if __name__ == '__main__':
    Compute.ROOT_FUNC = 'mpmath'
    Compute.DEFAULT_STEP = 1
    # np.random.seed(10010)
    np.random.seed(4)
    x1 = 1
    x2 = 6
    v1 = 0.5
    t = 0
    xs, ts = main(x1, x2, v1, t)
    print('count:', len(ts))
    set_plt_params()
    plt.style.use('ggplot')
    x1s = [x[0] for x in xs]
    x2s = [x[1] for x in xs]
    v1s = [x[2] for x in xs]
    ax = plt.subplot(111)
    ax.plot(ts, x2s, label=r'$x1(t)$')
    ax.plot(ts, x1s, label=r'$x2(t)$',)
    ax.legend()
    plt.xlabel('Time (sec)')
    plt.ylabel(r'$Position$ (units)')
    plt.savefig('/tmp/twocarsastry.pdf', bbox_inches='tight')
    plt.show()
