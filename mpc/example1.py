#!/usr/bin/env python3

from src.mpc import MPC
from math import ceil
import numpy
import matplotlib.pyplot as plt


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
    d = 0.01
    # The time horizon (second)
    h = 0.5
    N = ceil(h/d)   # The number of steps in MPC
    p = (lambda x: x[0] + (1 + x[0]*0.1*x[1])*d)     # Plant model (Euler)

    # XXX: The noisy plant model, which we don't know about
    pn = (lambda x: x[0] + numpy.random.rand()*0.1 + (1 + x[0]*0.1*x[1])*d)
    # FIXME: If the std-deviation is huge then SMT seems to crap out

    rx = [[5]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    # XXX: The bounds for state and control inputs
    xl = []
    xu = []
    ul = [-5000]
    uu = [5000]
    # XXX: Optimisation weights, equal optimisation
    xw = [1]
    uw = [0]
    # XXX: Initial values for state and control inputs
    x0 = [0.5]

    actions = []
    ts = [0]
    xs = [x0]
    count = 0

    # Get the solver
    s = MPC(N, 1, 1, [p], rx, ru, xw, uw, xl, xu, ul, uu)
    # XXX: Start simulating the movement of the robot
    while(count < h-d):
        print('------------l∞ norm cost function-------------')
        u0 = s.solve(x0)

        # XXX: Apply the action to the plant
        # FIXME: Make this plant model have some randomness
        x0 = [pn(x0 + u0)]
        print(x0, u0)
        # Append to list for plotting
        actions += [u0]
        xs += [x0]
        # Increment time
        count += d
        ts += [count]

    return ts, rx, xs
    # print('us:', actions)
    # Plot the results


if __name__ == '__main__':
    set_plt_params()
    ts, rx, xs = example()
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    plt.plot(ts, rx)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x$ (units)', fontweight='bold')
    plt.show()
