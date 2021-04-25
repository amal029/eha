#!/usr/bin/env python3

import src.mpc as SMPC
import matplotlib.pyplot as plt
import importlib
from math import ceil
from z3 import If


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


gref = None
uref = None
traj = None
objv = None
mcount = 0


def example():
    global objv
    global uref
    global traj
    global gref

    # XXX: Use 0.04 seconds for planning the trajectory
    d = 0.2
    # The time horizon (second)
    h = 4                       # This needs to be optimised
    # N = 4
    N = ceil(h/d)   # The number of prediction steps in MPC

    # XXX: The final position/velocity
    X1 = 0
    X2 = 0

    # XXX: start position and velocity
    x0 = [-1, 0]

    # XXX: Hybrid plant model, just forward Euler for now
    px1 = (lambda x: x[0] + x[1]*d)

    # XXX: The utility functions
    gf1 = (lambda x: 1 + 5*x + (25/2)*x**2 + (125/6)*x**3)
    g1 = (lambda x: gf1(x)/(12.18+gf1(x)))
    g2 = (lambda x: 12.18/(12.18+gf1(x)))

    # XXX: Evolution of x2
    px2 = (lambda x: x[1] + (If(x[3] == 0, g1(x[1]), g2(x[1])))*d)

    # XXX: The reference points
    rx = [[0, 0]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    rg = [[0]]*N    # The ref for the discrete control input

    # XXX: The bounds for state and control inputs
    xl = [-5.5, -0.5]*N
    xu = [1, 3]*N

    # XXX: The boundary condition at the final point
    BACK = 1
    for i in range(N*2-1, N*2-(BACK*2), -2):
        xl[i] = X1
        xl[i-1] = X2
        xu[i] = X1
        xu[i-1] = X2

    # XXX: The control bounds
    ul = [-1]*N
    uu = [1]*N

    # XXX: The discrete control bounds
    gb = [{0, 1}]               # 0 or 1

    # XXX: Cost for different points
    xw = [1, 1]*N
    uw = [0]*N
    gw = [0.5]*N

    # XXX: Initial values for state and control inputs
    # Get the solver
    s = SMPC.MPC(N, 2, 1, [px1, px2], xl, xu, ul, uu, norm=None,
                 P=1, Pset=gb, DEBUG=False)
    uref, gref, traj, objv = s.solve(x0, rx, ru, xw, uw, wg=gw, refg=rg,
                                     plan=True, opt=False)

    # XXX: This ts needs to be fixed to have instantaneous transitions
    ts = [i*d for i in range(1, N+1)]
    ts.insert(0, 0)
    return ts, traj, uref, gref, d


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    ts, traj, uref, gref, d = example()
    import sys
    osout = sys.stdout
    with open('/tmp/gears.txt', 'w') as f:
        sys.stdout = f
        print('ts:', ts, '\n', 'traj:', traj, '\n uref:', uref, '\ngref:',
              gref)
    sys.stdout = osout
    tr1s = [n for i, n in enumerate(traj)
            if i % 2 == 0]
    tr2s = [n for i, n in enumerate(traj)
            if i % 2 != 0]
    plt.style.use('ggplot')
    plt.plot(ts, tr1s)
    # plt.plot(ts, traj[:len(ts)])
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsx1.pdf', bbox_inches='tight')
    plt.close()

    plt.plot(ts, tr2s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x2(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsx2.pdf', bbox_inches='tight')
    plt.close()

    plt.plot(ts[:len(uref)], uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsuref.pdf', bbox_inches='tight')
    plt.close()

    plt.plot(ts[:len(gref)], gref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$g(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsgref.pdf', bbox_inches='tight')
    plt.close()
