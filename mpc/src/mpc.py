#!/usr/bin/env python3

from z3 import Real, Solver, sat, If


class MPC:

    def __init__(self, N, M, Q, fs, refx, refu, w, la, consxl, consxr,
                 consul, consur, norm=1, DEBUG=False):
        self.N = N
        self.M = M
        self.Q = Q
        self.DEBUG = DEBUG

        # XXX: Initialse the solver
        self.s = Solver()

        # Initialise the variables for each step N
        self.xs = [Real('x_%s_%s' % (j+1, i))
                   for i in range(N+1)
                   for j in range(M)]

        # Make the control variables
        self.us = [Real('u_%s_%s' % (j+1, i))
                   for i in range(N+1)
                   for j in range(Q)]

        # Add the evolution of the state variables
        k = 0
        for i in range(0, len(self.xs)-M, M):
            xss = self.xs[i:i+M]
            uss = self.us[k:k+Q]
            for j, f in enumerate(fs):
                self.s.add(f(xss+uss) == self.xs[i+j+M])
            k += Q

        # Add the linear constraints for system states
        consxl = consxl*N
        [self.s.add(self.xs[i+M] >= consxl[i])
         for i in range(len(consxl))]
        consxr = consxr*N
        [self.s.add(self.xs[i+M] <= consxr[i])
         for i in range(len(consxr))]

        # Add the linear constraints for control inputs
        consul = consul*N
        [self.s.add(self.us[i] >= consul[i])
         for i in range(len(consul))]
        consur = consur*N
        [self.s.add(self.us[i] <= consur[i])
         for i in range(len(consur))]

        # Now add the cost function
        refx = [j for i in refx for j in i]
        w = w*N
        refu = [j for i in refu for j in i]
        la = la*N

        def mabs(x):
            return If(x >= 0, x, -x)

        # XXX: Make the minimisation objective
        self.obj = Real('objective')
        if norm == 2:
            Dxs = [w[i]*(self.xs[i+M]-refx[i])**2
                   for i in range(len(refx))]
            Dus = [la[i]*(self.us[i]-refu[i])**2
                   for i in range(len(refu))]
            self.s.add(self.obj == sum(Dxs) + sum(Dus))

        elif norm == 1:
            Dxs = [w[i]*mabs(self.xs[i+M]-refx[i])
                   for i in range(len(refx))]
            Dus = [la[i]*mabs(self.us[i]-refu[i])
                   for i in range(len(refu))]
            self.s.add(self.obj == sum(Dxs) + sum(Dus))

        # XXX: Infinity norm
        elif norm is None:
            Dxs = [w[i]*mabs(self.xs[i+M]-refx[i])
                   for i in range(len(refx))]
            Dus = [la[i]*mabs(self.us[i]-refu[i])
                   for i in range(len(refu))]

            self.s.add(self.obj == MPC.mreduce(0, Dxs + Dus))

    @staticmethod
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
        elif ret == sat:
            return MPC.minimize(s, obj, objv/2)
        else:
            return objv/2, objv

    @staticmethod
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
            return MPC.binary_search(lb, half, s, obj, epsilon)
        else:
            return MPC.binary_search(half, ub, s, obj, epsilon)

    @staticmethod
    def mreduce(sV, Ej):
        def max(x, y):
            return If(x > y, x, y)

        if len(Ej) == 0:
            return sV
        else:
            return max(Ej[0], MPC.mreduce(sV, Ej[1:]))

    def solve(self, IS):

        self.s.push()           # put a point to push
        # Add the constraint for the initial_control and inital_state
        [self.s.add(self.xs[i] == j) for i, j in enumerate(IS)]

        # XXX: The state of the solver
        res = self.s.check()
        if res == sat:
            # XXX: Now minimize the objective function
            # This is current objective value upper bound
            objv = (float(self.s.model()[self.obj].numerator_as_long())
                    / float(self.s.model()[self.obj].denominator_as_long()))
            l, u = MPC.minimize(self.s, self.obj, objv)
            # XXX: Do the binary search
            res, model = MPC.binary_search(l, u, self.s, self.obj)
            # XXX: Get the control vector without the zeroth time
            toret = [(float(model[ret].numerator_as_long())
                      / float(model[ret].denominator_as_long()))
                     for ret in self.us[:self.Q]]
            if self.DEBUG:
                import sys
                osout = sys.stdout
                with open('/tmp/DEBUG.txt', 'a') as f:
                    sys.stdout = f
                    print(self.s.to_smt2())
                    sys.stdout = osout
                    print('---------------answer----------------------')
                    print(res, '\n', model)
            self.s.pop()
            return toret
        else:
            print('Model cannot be satisfied')
