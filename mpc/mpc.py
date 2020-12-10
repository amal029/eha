#!/usr/bin/env python3

from z3 import Real, Solver, sat, If
import numpy


def MPC(N, M, Q, fs, refx, refu, IS, w, la, consxl, consxr, consul, consur,
        norm=1, DEBUG=False):

    def minimize(s, obj, objv, epsilon=1e-6):
        # XXX: Get a lower bound
        s.push()
        s.add(obj <= objv/2)
        ret = s.check()
        if ret == sat:
            ro = (float(s.model()[obj].numerator_as_long())
                  / float(s.model()[obj].denominator_as_long()))
        s.pop()
        if ret == sat and ro <= epsilon:
            return objv/2, objv
        if ret == sat:
            return minimize(s, obj, objv/2)
        else:
            return objv/2, objv

    def binary_search(lb, ub, s, obj, epsilon=1e-6):
        if (ub - lb <= epsilon):
            s.check()
            return ub, s.model()
        else:
            half = lb + ((ub - lb)/2.0)
            s.push()
            s.add(obj >= lb, obj <= half)
            ret = s.check()
            s.pop()
        if ret == sat:
            return binary_search(lb, half, s, obj, epsilon)
        else:
            return binary_search(half, ub, s, obj, epsilon)

    def mreduce(sV, Ej):
        def max(x, y):
            return If(x > y, x, y)

        if len(Ej) == 0:
            return sV
        else:
            return max(Ej[0], mreduce(sV, Ej[1:]))

    # M = M
    # Q = len(IC)

    # XXX: Initialse the solver
    s = Solver()

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
    # [s.add(us[i] == j) for i, j in enumerate(IC)]

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
    [s.add(us[i] >= consul[i])
     for i in range(len(consul))]
    consur = consur*N
    [s.add(us[i] <= consur[i])
     for i in range(len(consur))]

    # Now add the cost function
    refx = [j for i in refx for j in i]
    w = w*N
    refu = [j for i in refu for j in i]
    la = la*N

    def mabs(x):
        return If(x >= 0, x, -x)
    # XXX: Make the minimisation objective
    obj = Real('objective')
    if norm == 2:
        Dxs = [w[i]*(xs[i+M]-refx[i])**2
               for i in range(len(refx))]
        Dus = [la[i]*(us[i]-refu[i])**2
               for i in range(len(refu))]
        s.add(obj == sum(Dxs) + sum(Dus))

    elif norm == 1:

        Dxs = [w[i]*mabs(xs[i+M]-refx[i])
               for i in range(len(refx))]
        Dus = [la[i]*mabs(us[i]-refu[i])
               for i in range(len(refu))]
        s.add(obj == sum(Dxs) + sum(Dus))

    # XXX: Infinity norm
    elif norm is None:

        Dxs = [w[i]*mabs(xs[i+M]-refx[i])
               for i in range(len(refx))]
        Dus = [la[i]*mabs(us[i]-refu[i])
               for i in range(len(refu))]

        s.add(obj == mreduce(0, Dxs + Dus))

    # XXX: The state of the solver
    # print(s)
    res = s.check()
    if res == sat:
        # XXX: Now minimize the objective function
        # This is current objective value upper bound
        objv = (float(s.model()[obj].numerator_as_long())
                / float(s.model()[obj].denominator_as_long()))
        l, u = minimize(s, obj, objv)
        # XXX: Do the binary search
        res, model = binary_search(l, u, s, obj)
        # XXX: Get the control vector without the zeroth time
        toret = [(float(model[ret].numerator_as_long())
                  / float(model[ret].denominator_as_long()))
                 for ret in us[:Q]]
        if DEBUG:
            import sys
            osout = sys.stdout
            with open('/tmp/DEBUG.txt', 'a') as f:
                sys.stdout = f
                print(s.to_smt2())
                sys.stdout = osout
                print('---------------answer----------------------')
                print(res, '\n', model)
        return toret
    else:
        print('Model cannot be satisfied')


def example():
    # XXX: Model a simple linear moving robot at constant velocity with
    # disturbance. Control it using MPC
    # The step-size
    d = 0.01
    # The time horizon (second)
    h = 0.1
    N = int(h//d)   # The number of steps in MPC
    p = (lambda x: x[0] + (1 + x[0]*0.1*x[1])*d)     # Plant model (Euler)

    # XXX: The noisy plant model, which we don't know about
    pn = (lambda x: x[0] + numpy.random.rand() + (1 + x[0]*0.1*x[1])*d)
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
    xs = [x0]
    count = 0
    # XXX: Start simulating the movement of the robot
    while(count <= h):
        # XXX: This is OK, l₁ norm
        print('------------l₁ norm cost function-------------')
        u0 = MPC(N, 1, 1, [p], rx, ru, x0, xw, uw, xl, xu, ul,
                 uu, norm=1, DEBUG=False)

        # XXX: Apply the action to the plant
        # FIXME: Make this plant model have some randomness
        x0 = [pn(x0 + u0)]
        print(x0, u0)
        # Append to list for plotting
        actions += [u0]
        xs += [x0]
        # Increment time
        count += d

    print('xs:', xs)
    print('us:', actions)


if __name__ == '__main__':
    example()
