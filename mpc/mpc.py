#!/usr/bin/env python3

from z3 import Real, Solver, sat, If


def minimize(s, obj, objv):
    # XXX: Get a lower bound
    s.push()
    s.add(obj <= objv/2)
    ret = s.check()
    s.pop()
    if ret == sat:
        return minimize(s, obj, objv/2)
    else:
        # XXX: What to do here?
        return objv/2, objv


# FIXME: Make this interative
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


def mpc(N, fs, refx, refu, IS, IC, w, la, consxl, consxr, consul, consur,
        norm=2):

    M = len(IS)
    Q = len(IC)

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
    if norm == 2:
        Dxs = [w[i]*(xs[i+M]-refx[i])**2
               for i in range(len(refx))]
        Dus = [la[i]*(us[i+Q]-refu[i])**2
               for i in range(len(refu))]
        s.add(obj == sum(Dxs) + sum(Dus))

    elif norm == 1:
        def abs(x):
            return If(x >= 0, x, -x)

        Dxs = [w[i]*abs(xs[i+M]-refx[i])
               for i in range(len(refx))]
        Dus = [la[i]*abs(us[i+Q]-refu[i])
               for i in range(len(refu))]
        s.add(obj == sum(Dxs) + sum(Dus))

    # XXX: Infinity norm
    elif norm is None:
        def abs(x):
            return If(x >= 0, x, -x)

        Dxs = [w[i]*abs(xs[i+M]-refx[i])
               for i in range(len(refx))]
        Dus = [la[i]*abs(us[i+Q]-refu[i])
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
        # print('-------------solved model------------------')
        # print(s)
        print('---------------answer----------------------')
        print(res, '\n', model)
    else:
        print('Model cannot be satisfied')


if __name__ == '__main__':
    # Made up example for model predictive control
    f1 = (lambda x: x[0]*x[1]+x[2])
    f2 = (lambda x: 0*x[0]+x[1]**2+0.5*x[2])
    N = 3
    fs = [f1, f2]
    # refx = [[0.8, 0.7], [0, 1], [1, 1]]
    refx = [[0.8, 0.7]]*N
    refu = [[0]]*N
    IS = [5, 0.89]
    IC = [0]
    w = [.5, 0.8]
    la = [0]
    consxl = [-10, -10]
    consxr = [10, 10]
    consul = [-1]
    consur = [1]
    # XXX: This is OK, l₁ norm
    print('------------l₁ norm cost function-------------')
    mpc(N, fs, refx, refu, IS, IC, w, la, consxl, consxr, consul, consur,
        norm=1)
    # XXX: l₂ norm takes too long if the system evolution is linear, but why?
    print('------------l₂ norm cost function-------------')
    mpc(N, fs, refx, refu, IS, IC, w, la, consxl, consxr, consul, consur,
        norm=2)
    # XXX: This is the l∞ norm
    print('------------l∞ norm cost function-------------')
    mpc(N, fs, refx, refu, IS, IC, w, la, consxl, consxr, consul, consur,
        norm=None)

    # https://www.alglib.net/translator/man/manual.cpython.html#example_minqp_d_lc1
