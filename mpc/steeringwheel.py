#!/usr/bin/env python3

import src.mpc as SMPC
import numpy
import matplotlib.pyplot as plt
import importlib
from math import pi, ceil
from z3 import If, And


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
    h = 2
    N = ceil(h/d)   # The number of prediction steps in MPC
    # XXX: Hybrid plant model, just forward Euler for now
    p = (lambda x: x[0] +
         (If(And(x[0] > pi/2, x[0] <= 3*pi/2), -x[1], x[1]))*d)

    # XXX: The noisy plant model, which we don't know about
    pn = (lambda x: x[0] + numpy.random.rand()*0.01 +
          (-x[1] if(x[0] > pi/2 and x[0] <= 3*pi/2) else x[1])*d)
    # FIXME: If the std-deviation is huge then SMT seems to crap out

    # XXX: The below things usually come from the planning phase,
    # Planning can be done, say, using the complete horizon with the
    # plant model and a quadratic offline solver for non-linear mpc.

    rx = [[pi/2]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    # XXX: The bounds for state and control inputs
    xl = [0]*N
    xu = [2*pi]*N
    # XXX: Adding special constraint stating that the last point has to
    # be very close to pi/2
    e = 1e-6                    # error bound
    xl[-1] = pi/2 - e
    xu[-1] = pi/2 + e
    ul = [-2]*N
    uu = [2]*N

    # XXX: Optimisation weights, equal optimisation
    xw = [1]
    uw = [0]

    # XXX: Initial values for state and control inputs
    # Get the solver
    s = SMPC.MPC(N, 1, 1, [p], xl, xu, ul, uu)
    uref, traj = s.solve([pi+0.1], rx, ru, xw, uw, plan=True)

    # XXX: Now start following the trajectory with noise
    x0 = [traj[0]]
    actions = []
    ts = [0]
    xs = [x0]

    # XXX: Equal weights
    xw = [1]
    # XXX: This needs to be zero, because we have a lot of noise
    uw = [0]

    # XXX: Predict only N ahead
    N = 1
    xl = [0]*N
    xu = [2*pi]*N
    ul = [-2]*N
    uu = [2]*N
    s = SMPC.MPC(N, 1, 1, [p], xl, xu, ul, uu, norm=1)
    count = 1
    # XXX: Start simulating the movement of the robot
    while(True):
        print('------------l∞ norm cost function-------------')
        u0 = s.solve(x0, [traj[count:count+N]], [uref[count:count+N]],
                     xw, uw)

        # XXX: Apply the action to the plant, with noise
        x0 = [pn(x0 + u0)]
        print(x0, u0)
        # Append to list for plotting
        actions += [u0]
        xs += [x0]
        ts += [count*d]
        # Increment time
        count += 1
        if(count > h/d-N+1):
            break

    return xs, actions, ts, traj, uref


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    xs, us, ts, traj, uref = example()
    # print(ts, rx, xs)
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    plt.plot(ts, traj[:len(ts)])
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x(t)$ (units)', fontweight='bold')
    plt.show()
    plt.scatter(ts[1:], us)
    plt.scatter(ts[1:], uref[:len(us)])
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.show()
