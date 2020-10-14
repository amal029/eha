#!/usr/bin/env python3

import sympy as S
import numpy as np
import matplotlib.pyplot as plt
from src.sdesolver import Compute


# Total simulation time
SIM_TIME = 20

# error
e = 1e-1

dynamics = {'S1': {S.sympify('x(t)'): [S.sympify(-0.01), S.sympify(1)]},
            'S2': {S.sympify('x(t)'): [S.sympify(0.01), S.sympify(-1)]},
            'S3': {S.sympify('x(t)'): [S.sympify(0), S.sympify(0)]}}


class GSHS:
    def __compute(x, t, dWts, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x(t)'): x}
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
    def S1(x, t, dWts):
        if abs(S.cos(x)) <= e:
            state = 3           # Destination S3
            return state, 0, (x,)
        elif S.cos(x) >= e:
            # Destination S2
            state = 2
            return state, 0, (x,)
        else:
            g = S.sympify('cos(x(t))')
            T, vars = GSHS.__compute(x, t, dWts, 'S1', [g])
            return 1, T, vars

    @staticmethod
    def S2(x, t, dWts):
        if abs(S.cos(x)) <= e:
            state = 3
            return state, 0, (x,)
        elif S.cos(x) <= -e:
            # Destination S1
            state = 1
            return state, 0, (x,)
        else:
            g = S.sympify('cos(x(t))')
            T, vars = GSHS.__compute(x, t, dWts, 'S2', [g])
            return 2, T, vars

    @staticmethod
    def S3(x, t, dWts):
        T, vars = GSHS.__compute(x, t, dWts, 'S3', [])
        return 3, T, vars


def main(x, t):
    X_loc = {
        1: GSHS.S1,
        2: GSHS.S2,
        3: GSHS.S3
    }
    X_strloc = {
        1: 'S1',
        2: 'S2',
        3: 'S3'
    }

    if (x < np.pi/2):
        state = 2
    else:
        state = 1

    # Print the outputs
    print('%.4f: Loc:%s, x:%s' % (t, X_strloc[state], x))

    xs = [x]
    ts = [t]
    while(True):
        # Create dWt
        dWts = {S.sympify('x(t)'): np.random.randn(Compute.R)}

        # Call the dynamics and run these until some time
        state, T, (x,) = X_loc[state](x, t, dWts)

        xs.append(x)
        ts.append(t)
        t += T

        print('%.4f: Loc:%s, x:%s' % (t, X_strloc[state], x))
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
    Compute.ROOT_FUNC = 'mpmath'
    Compute.DEFAULT_STEP = 1
    # np.random.seed(4907)
    x = 0.5
    t = 0
    xs, ts = main(x, t)
    print('count:', len(ts))
    set_plt_params()
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x$ (units)', fontweight='bold')
    plt.savefig('/tmp/driver.pdf', bbox_inches='tight')
    plt.show()
