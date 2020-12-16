#!/usr/bin/env python3

import src.mpc as SMPC
import numpy
from scipy import io
import matplotlib.pyplot as plt
import importlib
# from math import pi, ceil
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
    d = 0.01                    # The step-size
    h = 0.54                      # Time horizon
    N = 1   # The number of steps to predict + control

    # The plant model with uncertainty. Argument x is a vector of state
    # + control vectors

    # XXX: Supporting functions
    cos = (lambda t: 1 - t**2/2 + t**4/24)
    sin = (lambda t: t - t**3/6 + t**5/120)
    tan = (lambda t: t + t**3/3 + 2*t**5/15)

    # cos = (lambda t: 1 - t**2/2)
    # sin = (lambda t: t - t**3/6)
    # tan = (lambda t: t + t**3/3)
    # XXX: The length of the robotic link
    ll = 1

    # XXX: The guard
    g = (lambda x, y: Or(And(y >= 1.8, x <= 2.8), And(y <= -0.8, x <= 2.8)))
    gn = (lambda x, y: (y >= 1.8 and x <= 2.8) or (y <= -0.8 and x <= 2.8))

    # XXX: The evolution of x
    px = (lambda x: x[0] + If(g(x[0], x[1]), 0, cos(x[2])*x[4])*d)
    # XXX: The real one with noise
    pnx = (lambda x: x[0] + numpy.random.rand()*0.000 +
           (0 if(gn(x[0], x[1])) else cos(x[2])*x[4]*d))

    # XXX: The evolution of y
    py = (lambda x: x[1] + If(g(x[0], x[1]), 0, sin(x[2])*-x[4])*d)
    pny = (lambda x: x[1] + numpy.random.rand()*0.0015 +
           (0 if(gn(x[0], x[1])) else sin(x[2])*-x[4]*d))
    # XXX: The evolution of th
    pth = (lambda x: x[2] + If(g(x[0], x[1]), 0, tan(x[3])/ll*x[4])*d)
    pnth = (lambda x: x[2] + numpy.random.rand()*0.00 +
            (0 if(gn(x[0], x[1])) else tan(x[3])/ll*x[4]*d))
    # XXX: Evolution of ph
    pph = (lambda x: x[3] + If(g(x[0], x[1]), 0, -x[5])*d)
    pnph = (lambda x: x[3] + numpy.random.rand()*0.00 +
            (0 if(gn(x[0], x[1])) else -x[5]*d))
    # XXX: All the dynamics together
    ps = [px, py, pth, pph]
    pns = [pnx, pny, pnth, pnph]

    # XXX: The reference trajectory from simulink
    m = io.loadmat('/Users/amal029_old/eha/mpc/robottraj.mat')
    # XXX: The inital state variables
    x0 = [m['xout'][0][0], m['yout'][0][0], m['thout'][0][0],
          m['phout'][0][0]]

    # XXX: Reference for control inputs
    rus = [[4, 1]]*N

    # XXX: References (this should be inside the loop) -- sliding window
    rxs = [[m['xout'][i][0], m['yout'][i][0],
            m['thout'][i][0], m['phout'][i][0]]
           for i in range(1, N+1, 1)]

    # XXX: Bounds for state variables
    xl = [0, -0.8, -2, -2]*N
    xu = [2.8, 1.8, 2, 2]*N

    # XXX: Bounds for control inputs
    ul = [0, -1]*N
    uu = [4, 1]*N

    # XXX: Weights for optimisation
    wx = [0.0, 0.0, 0, 0]*N
    wu = [1, 1]*N

    # XXX: Start simulating
    actions = []
    ts = [0]
    xs = [x0]
    count = 1
    # XXX: The solver
    s = SMPC.MPC(N, 4, 2, ps, xl, xu, ul, uu, norm=1)
    print(x0)
    while(True):
        print('------------l∞ norm cost function-------------')
        u0, _ = s.solve(x0, rxs, rus, wx, wu)
        x0 = [p(x0+u0) for p in pns]
        print('time:', count*d)
        print('us: ', u0)
        print('xs:', x0)
        xs.append(x0)
        actions.append(u0)
        ts.append(count*d)
        count += 1
        if (count > h/d-N+1):
            break
        rxs = [[m['xout'][i][0], m['yout'][i][0],
                m['thout'][i][0], m['phout'][i][0]]
               for i in range(1, N+1, 1)]
    return xs, actions, ts, m


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    xs, actions, ts, m = robot()
    print(ts)
    xxs = [i[0] for i in xs]
    yys = [i[1] for i in xs]
    plt.style.use('ggplot')
    plt.plot(xxs, yys)
    plt.plot(m['xout'][:len(m['xout'])-2],
             m['yout'][:len(m['yout'])-2])
    plt.show()
