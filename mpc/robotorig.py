#!/usr/bin/env python3

import src.mpc as SMPC
import matplotlib.pyplot as plt
import importlib
from math import ceil
from z3 import If, And, Or


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


def robot():
    # XXX: Follow the trajectory of the robot from the QSS paper. The
    # trajectory is generated from simulink.
    d = 0.2                    # The step-size
    h = 1.0                      # Time horizon
    N = ceil(h/d)  # The number of steps to predict + control

    # The plant model with uncertainty. Argument x is a vector of state
    # + control vectors

    # XXX: Supporting functions
    # cos = (lambda t: 1 - t**2/2 + t**4/24)
    # sin = (lambda t: t - t**3/6 + t**5/120)
    # tan = (lambda t: t + t**3/3 + 2*t**5/15)

    cos = (lambda t: 1 - t**2/2)
    sin = (lambda t: t - t**3/6)
    tan = (lambda t: t + t**3/3)
    # XXX: The length of the robotic link
    ll = 1

    # XXX: The guard
    g = (lambda x, y: Or(And(y >= 1.8, x <= 2.8), And(y <= -0.8, x <= 2.8)))

    # XXX: The evolution of x
    px = (lambda x: x[0] + If(g(x[0], x[1]), 0, cos(x[2])*x[4])*d)

    # XXX: The evolution of y
    py = (lambda x: x[1] + If(g(x[0], x[1]), 0, sin(x[2])*-x[4])*d)

    # XXX: The evolution of th
    pth = (lambda x: x[2] + If(g(x[0], x[1]), 0, tan(x[3])/ll*x[4])*d)

    # XXX: Evolution of ph
    pph = (lambda x: x[3] + If(g(x[0], x[1]), 0, -x[5])*d)

    # XXX: All the dynamics together
    ps = [px, py, pth, pph]

    # XXX: The inital state variables
    x0 = [0, 1, 0, 0]

    # XXX: Reference for control inputs
    rus = [[4, 1]]*N

    # XXX: References (this should be inside the loop) -- sliding window
    rxs = [[0.7, -0.7, 2.0, 0.47]]*N

    # XXX: Bounds for state variables
    xl = [0, -0.8, -2, -2]*N
    xu = [2.8, 1.8, 2, 2]*N

    # XXX: The terminal condition
    # e = 1e-6
    # xl[N*4-4] = 0.7 - e
    # xu[N*4-4] = 0.7 + e
    # xl[N*4-3] = -0.7 - e
    # xu[N*4-3] = -0.7 + e

    # XXX: Bounds for control inputs
    ul = [3, 1]*N
    uu = [4, 1]*N

    # XXX: Weights for optimisation
    wx = [1, 1, 0, 0]*N
    wu = [0, 0]*N

    # XXX: The solver
    s = SMPC.MPC(N, 4, 2, ps, xl, xu, ul, uu, norm=None)
    uref, _, traj = s.solve(x0, rxs, rus, wx, wu, plan=True, opt=False)
    ts = [i*d for i in range(N)]
    ts.insert(0, 0)
    return ts, traj, uref


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    ts, traj, uref = robot()
    import sys
    osout = sys.stdout
    with open('/tmp/robot.txt', 'w') as f:
        sys.stdout = f
        print('ts:', ts, '\n', 'traj:', traj, '\n uref:', uref)
    sys.stdout = osout
    xxs = [traj[i] for i in range(0, len(traj), 4)]
    yys = [traj[i] for i in range(1, len(traj), 4)]
    plt.style.use('ggplot')
    plt.plot(xxs, yys)
    plt.savefig('/tmp/robottraj.pdf', bbox_inches='tight')
    plt.close()
    plt.plot(ts[1:], [uref[i] for i in range(0, len(uref), 2)])
    plt.savefig('/tmp/roboturef1.pdf', bbox_inches='tight')
    plt.close()
    plt.plot(ts[1:], [uref[i] for i in range(1, len(uref), 2)])
    plt.savefig('/tmp/roboturef2.pdf', bbox_inches='tight')
