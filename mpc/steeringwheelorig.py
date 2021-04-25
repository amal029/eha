#!/usr/bin/env python3

import src.mpc as SMPC
import matplotlib.pyplot as plt
import importlib
from math import pi, ceil
from z3 import If, And
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


objv = None
uref = None
traj = None
gref = None
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
    N = ceil(h/d)   # The number of prediction steps in MPC
    # XXX: Hybrid plant model, just forward Euler for now
    p = (lambda x: x[0] +
         (If(And(x[0] > pi/2, x[0] <= 3*pi/2), -x[1], x[1]))*d)

    # XXX: The noisy plant model, which we don't know about
    # pn = (lambda x: x[0] + numpy.random.rand()*0.01 +
    #       (-x[1] if(x[0] > pi/2 and x[0] <= 3*pi/2) else x[1])*d)
    # FIXME: If the std-deviation is huge then SMT seems to crap out

    # XXX: The below things usually come from the planning phase,
    # Planning can be done, say, using the complete horizon with the
    # plant model and a quadratic offline solver for non-linear mpc.

    rx = [[pi/2]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    # XXX: The bounds for state and control inputs
    # XXX: Path constraint
    xl = [0]*N
    xu = [2*pi]*N

    # XXX: Adding special constraint stating that the last point has to
    # be very close to pi/2

    # XXX: Very important constraint to guarantee convergence with SAT.
    # XXX: Boundary constraint
    BACK = 1
    for i in range(N-1, N-BACK, -1):
        xl[i] = pi/2
        xu[i] = pi/2
    # XXX: Path constraint
    ul = [-2]*N
    uu = [2]*N

    # XXX: Optimisation weights, equal optimisation
    xw = [4]
    uw = [1]

    # XXX: The admissible intial plan
    s = SMPC.MPC(N, 1, 1, [p], xl, xu, ul, uu)
    uref, gref, traj, objv = s.solve([pi], rx, ru, xw, uw, plan=True, opt=True)

    Q = 1     # the number of continous control inputs

    # XXX: Now start differential_evolution to get the minimum
    def mopt(x):
        global objv
        global uref
        global traj
        global gref
        global mcount
        mcount += 1
        if (mcount % 1000 == 0):
            print('iter: ', mcount)
        assert(len(x) == N*Q)
        s = SMPC.MPC(N, 1, 1, [p], xl, xu, x, x)
        nuref, ngref, ntraj, nobjv = s.solve([pi], rx, ru, xw, uw, plan=True,
                                             opt=False, mopt=True)
        if nobjv is not None:
            print('found a better solution!')
            objv = nobjv
            uref = nuref
            traj = ntraj
            gref = ngref
        return objv

    # bounds = list(zip(ul, uu))
    # _ = differential_evolution(mopt, bounds, strategy='rand1bin')
    # _ = dual_annealing(mopt, bounds, x0=uref, maxfun=10000, initial_temp=10000)
    # print(result.x)
    ts = [i*d for i in range(N)]
    ts.insert(0, 0)
    # print(traj, uref)

    return ts, traj, uref


if __name__ == '__main__':
    importlib.reload(SMPC)
    set_plt_params()
    ts, traj, uref = example()
    import sys
    osout = sys.stdout
    with open('/tmp/steeringwheel.txt', 'w') as f:
        sys.stdout = f
        print('ts:', ts, '\n', 'traj:', traj, '\n uref:', uref)
    sys.stdout = osout
    plt.style.use('ggplot')
    # plt.plot(ts, xs)
    plt.plot(ts, traj[:len(ts)])
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/steeringwheelx.pdf', bbox_inches='tight')
    # plt.close()
    plt.show()
    # plt.scatter(ts[1:], us)
    plt.plot(ts[1:], uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/steeringwheeluref.pdf', bbox_inches='tight')
    plt.show()
