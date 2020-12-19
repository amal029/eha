#!/usr/bin/env python3

import src.mpc as SMPC
import matplotlib.pyplot as plt
import importlib
from math import ceil
from z3 import If
from scipy.optimize import dual_annealing


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
    # XXX: Model a simple linear moving robot at constant velocity with
    # disturbance. Control it using MPC
    # The step-size
    # XXX: Use 0.04 seconds for planning the trajectory
    d = 0.04
    # The time horizon (second)
    h = 1
    # N = 4
    N = ceil(h/d)   # The number of prediction steps in MPC

    # XXX: The wanted temp
    Temp = 18.5
    # XXX: The diff
    # K = 125

    # XXX: Start temperature
    STemp = 20

    # XXX: Hybrid plant model, just forward Euler for now
    p = (lambda x: x[0] + (If(x[0] >= Temp, -x[1], x[1]))*x[0]*d)

    # XXX: The noisy plant model, which we don't know about
    # pn = (lambda x: x[0] + numpy.random.rand()*0.00 +
    #       ((-x[1] if(x[0] >= Temp) else x[1])*x[0]*d))

    rx = [[Temp]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    # XXX: The bounds for state and control inputs
    xl = [-30]*N
    xu = [30]*N

    # XXX: This is very important, can even be modeled as a cost

    # XXX: Adding special constraint stating that the last point has to
    # be very close to the final reference
    BACK = 5
    for i in range(N-1, N-BACK, -1):
        xl[i] = Temp
        xu[i] = Temp
    # xl[-1] = Temp
    # xu[-1] = Temp

    # XXX: The control bounds
    ul = [0]*N
    uu = [0.6]*N

    # XXX: Optimisation weights, equal optimisation
    xw = [4.0]
    uw = [0.0]

    # XXX: Initial values for state and control inputs
    # Get the solver
    s = SMPC.MPC(N, 1, 1, [p], xl, xu, ul, uu, norm=None)
    uref, gref, traj, objv = s.solve([STemp], rx, ru, xw, uw, plan=True,
                                     opt=False)
    Q = 1

    # XXX: Now start differential_evolution to get the minimum
    def mopt(x):
        global objv
        global uref
        global traj
        global gref
        global mcount
        mcount += 1
        if (mcount % 100 == 0):
            print('iter: ', mcount)
        assert(len(x) == N*Q)
        s = SMPC.MPC(N, 1, 1, [p], xl, xu, x, x, norm=None, TIMEOUT=1000)
        nuref, ngref, ntraj, nobjv = s.solve([STemp], rx, ru, x, x,
                                             plan=True, opt=False, mopt=True)
        if nobjv is not None:
            print('found a better solution!')
            objv = nobjv
            uref = nuref
            traj = ntraj
            gref = ngref
        return objv

    bounds = list(zip(ul, uu))
    _ = dual_annealing(mopt, bounds, x0=uref, maxfun=10000,
                       initial_temp=10000)
    ts = [i*d for i in range(N)]
    ts.insert(0, 0)
    # print(traj, uref)
    # assert(True is False)

    # XXX: Now start following the trajectory with noise
    # x0 = [STemp]
    # actions = []
    # ts = [0]
    # xs = [x0]

    # s = SMPC.MPC(N, 1, 1, [p], xl, xu, ul, uu, norm=None)
    # count = 1
    # # XXX: Start simulating the movement of the robot
    # while(True):
    #     u0, _, _ = s.solve(x0, rx, ru, xw, uw, plan=True)

    #     if(count + N)*d >= h:
    #         # XXX: Apply all N inputs
    #         xu = 0
    #         for i in range(N):
    #             u00 = u0[xu:xu+1]
    #             actions += [u00]
    #             x0 = [pn(x0 + u00)]
    #             print(x0, u00)
    #             xs += [x0]
    #             ts += [(count+i)*d]
    #             xu += 1         # Q = 1
    #         print('the end: ', count)
    #         break
    #     else:
    #         u0 = u0[:1]         # Q = 1

    #     # XXX: Apply the action to the plant, with noise
    #     x0 = [pn(x0 + u0)]
    #     print(x0, u0)
    #     # Append to list for plotting
    #     actions += [u0]
    #     xs += [x0]
    #     ts += [count*d]
    #     # Increment time
    #     count += 1

    return traj, uref, ts


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    xs, us, ts = example()
    import sys
    osout = sys.stdout
    with open('/tmp/gears.txt', 'w') as f:
        sys.stdout = f
        print('ts:', ts, '\n', 'traj:', xs, '\n uref:', us)
    sys.stdout = osout
    # print(ts, rx, xs)
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    # plt.plot(ts, traj[:len(ts)])
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsx.pdf', bbox_inches='tight')
    plt.show()
    plt.plot(ts[1:], us)
    # plt.plot(ts[1:], uref[:len(us)])
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsuref.pdf', bbox_inches='tight')
    plt.show()
