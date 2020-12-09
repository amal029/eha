#!/usr/bin/env python3

from z3 import Real, SolverFor, sat


def mpc(N, fs, refx, refu, IS, IC, w, la, consxl, consxr, consul, consur):

    M = len(IS)
    Q = len(IC)

    # XXX: Initialse the solver
    s = SolverFor('QF_NRA')

    # Initialise the variables for each step N
    xs = [Real('x_%s_%s' % (j+1, i))
          for i in range(N+1)
          for j in range(M)]
    # print(xs)
    # Make the control variables
    us = [Real('u_%s_%s' % (j+1, i))
          for i in range(N+1)
          for j in range(Q)]

    # Add the constraint for the initial_control and inital_state
    [s.add(xs[i] == j) for i, j in enumerate(IS)]
    [s.add(us[i] == j) for i, j in enumerate(IC)]

    # Add the evolution of the state variables
    k = 0
    for i in range(0, len(xs)-M, M):
        xss = xs[i:i+M]
        uss = us[k:k+Q]
        for j, f in enumerate(fs):
            s.add(f(xss+uss) == xs[i+j+M])
        k += Q

    # Add the linear constraints for system states
    consxl = consxl*N
    [s.add(xs[i+M] >= consxl[i])
     for i in range(len(consxl))]
    consxr = consxr*N
    [s.add(xs[i+M] <= consxr[i])
     for i in range(len(consxr))]

    # Add the linear constraints for control inputs
    consul = consul*N
    [s.add(us[i+Q] >= consul[i])
     for i in range(len(consul))]
    consur = consur*N
    [s.add(us[i+Q] <= consur[i])
     for i in range(len(consur))]

    # Now add the cost function
    refx = [j for i in refx for j in i]
    w = w*N
    refu = [j for i in refu for j in i]
    la = la*N

    # XXX: Make the minimisation objective
    obj = Real('objective')
    Dxs = [w[i]*(xs[i+M]-refx[i])**2
           for i in range(len(refx))]
    Dus = [la[i]*(us[i+Q]-refu[i])**2
           for i in range(len(refu))]
    s.add(obj == sum(Dxs) + sum(Dus))

    # XXX: The state of the solver
    print(s)
    res = s.check()
    if res == sat:
        print(s.model())
    else:
        print('Model cannot be satisfied')


if __name__ == '__main__':
    # XXX: The evolution function for state variables
    f1 = (lambda x: x[0]+x[1]+x[2])
    f2 = (lambda x: 0*x[0]+x[1]+0.5*x[2])
    N = 3
    fs = [f1, f2]
    refx = [[1, 1], [2, 2], [3, 3]]
    refu = [[1], [1], [1]]
    IS = [0, 0]
    IC = [0]
    w = [1, 0.8]
    la = [1]
    consxl = [0, 0]
    consxr = []
    consul = [1]
    consur = [1]
    mpc(N, fs, refx, refu, IS, IC, w, la, consxl, consxr, consul, consur)
