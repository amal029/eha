#!/usr/bin/env python3

from gurobipy import Model, GRB, quicksum
from math import ceil, pi
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


def main():
    # XXX: Use 0.04 seconds for planning the trajectory
    d = 0.04
    # The time horizon (second)
    h = 1
    N = ceil(h/d)   # The number of prediction steps in MPC
    # XXX: Number of state vars
    M = 1
    # XXX: Number of control vars
    Q = 1
    m = Model('steeringwheel')

    # XXX: The state variables
    xs = m.addVars(N+1, M, name='x')
    for i in range(N-1):
        m.addRange(xs[i+1, 0], 0, 2*pi)
    # e = 2e-6                    # error bound
    # XXX: Add the final state
    m.addRange(xs[N-1, M-1], pi/2, pi/2)
    # XXX: Add the initial state cosntraints
    m.addConstr(xs[0, 0] == pi+0.1)

    # XXX: The control variables
    us = m.addVars(N, Q, lb=-2, ub=2, name='u')

    # XXX: Path planning constraint
    for i in range(N-1):
        m.addConstr(us[i+1, 0] <= us[i, 0])

    # Now the system evolution constraint

    # XXX: Extra vars
    BIG_M = 10000
    # The binary variables
    zs = m.addVars(N, vtype=GRB.BINARY, name='z')
    # Constraint for the binary variables
    bs = m.addVars(N, 2, vtype=GRB.BINARY, name='bs')
    for i in range(N):
        m.addConstr(BIG_M*bs[i, 0] >= (xs[i, 0] - pi/2))
        m.addConstr(BIG_M*(1 - bs[i, 0]) >= (pi/2 - xs[i, 0]))
        m.addConstr(BIG_M*bs[i, 1] >= (3*pi/2 - xs[i, 0]))
        m.addConstr(BIG_M*(1 - bs[i, 1]) >= (xs[i, 0] - 3*pi/2))
        m.addGenConstrAnd(zs[i], [bs[i, 0], bs[i, 1]])

    # XXX: Add the if-else constraint
    for i in range(N):
        m.addConstr((zs[i] == 1) >> (xs[i+1, 0] == (-us[i, 0]*d+xs[i, 0])),
                    name=('i1_%s' % i))
        m.addConstr((zs[i] == 0) >> (xs[i+1, 0] == (us[i, 0]*d+xs[i, 0])),
                    name=('i2_%s' % i))

    # XXX: DEBUG
    # for i in range(N):
    #     m.addConstr(xs[i+1, 0] == -us[i, 0]*d + xs[i, 0])

    # XXX: Reference and main objective
    rx = m.addVars(N, M, lb=0, ub=2*pi, name='rx')
    # XXX: The last one should be exactly pi/2
    m.addConstr(rx[N-1, M-1] == pi/2)
    ob1 = m.addVars(N, name='ob1')
    ob11 = m.addVars(N, name='ob11')
    for i in range(N):
        m.addConstr(ob1[i] == 1*(rx[i, 0] - xs[i+1, 0]))
        m.addGenConstrAbs(ob11[i], ob1[i])
    ob_1 = quicksum(ob11)

    ru = m.addVars(N, Q, lb=0, ub=0, name='ru')
    ob2 = m.addVars(N, name='ob2')
    ob22 = m.addVars(N, name='ob22')
    for i in range(N):
        m.addConstr(ob2[i] == 0*(ru[i, 0] - us[i, 0]))
        m.addGenConstrAbs(ob22[i], ob2[i])
    ob_2 = quicksum(ob22)
    m.setObjective(quicksum([ob_1, ob_2]), GRB.MINIMIZE)

    # XXX: DEBUG
    m.write('/tmp/steeringwheelmilp.lp')

    # XXX: Set number of threads
    m.setParam('Threads', 1)

    # XXX: Solve the model
    m.optimize()
    if m.status == GRB.OPTIMAL:
        rs = [pi/2]*(N+1)
        ts = [i*d for i in range(N+1)]
        # XXX: Get the trajectory
        xr = [xs[i, 0].getAttr(GRB.Attr.X) for i in range(N+1)]
        ur = [us[i, 0].getAttr(GRB.Attr.X) for i in range(N)]
        return ts, rs, xr, ur


if __name__ == '__main__':
    ts, rx, xs, us = main()
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    plt.plot(ts, rx)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x(t)$ (units)', fontweight='bold')
    plt.show()
    plt.plot(ts[:len(ts)-1], us)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.show()
