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


def example():
    # XXX: Model a simple linear moving robot at constant velocity with
    # disturbance. Control it using MPC
    # The step-size
    # XXX: Use 0.04 seconds for planning the trajectory
    d = 0.08
    # The time horizon (second)
    h = 1.04
    N = ceil(h/d)   # The number of prediction steps in MPC

    # XXX: The wanted water level in Tank 1
    X1 = 2
    # XXX: Wanted level in tank 2
    X2 = 2

    # XXX: Start levels
    # XXX: Having a set here is not very useful
    x0 = [0.5, 1]

    # XXX: Hybrid plant model, just forward Euler for now
    px1 = (lambda x: x[0] + (If(x[3] == 0, x[2], -x[0]**2))*d)
    px2 = (lambda x: x[1] + (If(x[3] == 1, x[2], -x[1]**2))*d)

    rx = [[X1, X2]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    rg = [[0]]*N       # The ref point for the discrete control input

    # XXX: The bounds for state and control inputs
    xl = [0, 0]*N
    xu = [10, 10]*N

    # XXX: Adding special constraint stating that the last point has to
    # be very close to the final Temp
    BACK = 1
    # XXX: We have two state variables, M = 2
    for i in range(N*2-1, N*2-(BACK*2), -2):
        xl[i] = X1
        xl[i-1] = X2
        xu[i] = X1
        xu[i-1] = X2

    # XXX: The continuous control bounds
    ul = [1]*N
    uu = [16]*N

    # XXX: The discrete control bounds
    gb = [{0, 1}]*N             # Pset

    # XXX: Optimisation weights, equal optimisation
    xw = [0.1, 0.1]*N
    uw = [1]*N
    gw = [1]*N                  # discrete input

    # Get the solver
    s = SMPC.MPC(N, 2, 1, [px1, px2], xl, xu, ul, uu, P=1, Pset=gb, norm=None)
    uref, gref, traj = s.solve(x0, rx, ru, xw, uw, plan=True, refg=rg, wg=gw,
                               opt=False)

    ts = [i*d for i in range(1, N+1)]
    ts.insert(0, 0)
    # print(uref, gref, traj)
    return ts, traj, uref, gref, d


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    ts, traj, uref, gref, d = example()
    import sys
    osout = sys.stdout
    with open('/tmp/watertank.txt', 'w') as f:
        sys.stdout = f
        print('ts:', ts, '\n', 'traj:', traj, '\n uref:', uref, '\ngref:',
              gref)
    sys.stdout = osout
    # x1s = [n[0] for n in xs]
    tr1s = [n for i, n in enumerate(traj)
            if i % 2 == 0]
    # x2s = [n[1] for n in xs]
    tr2s = [n for i, n in enumerate(traj)
            if i % 2 != 0]
    plt.style.use('ggplot')
    # plt.plot(ts, x1s)
    plt.plot(ts, tr1s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankx1.pdf', bbox_inches='tight')
    plt.show()
    # plt.plot(ts, x2s)
    # plt.plot(ts, tr2s[:len(ts)])
    plt.plot(ts, tr2s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x2(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankx2.pdf', bbox_inches='tight')
    plt.show()
    # plt.scatter(ts[1:], us)
    plt.plot(ts[:len(uref)], uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankuref.pdf', bbox_inches='tight')
    plt.show()
    # gs = [j for i in gs for j in i]   # requires flattening
    # plt.plot(ts[1:], gs)
    plt.plot(ts[:len(gref)], gref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$g(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankgref.pdf', bbox_inches='tight')
    plt.show()
