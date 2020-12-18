#!/usr/bin/env python3

from z3 import Real, sat, If, Or, Int, AlgebraicNumRef, Optimize
import z3


class MPC:

    def __init__(self, N, M, Q, fs, consxl, consxr,
                 consul, consur, norm=None, DEBUG=False, P=0, Pset=[]):
        z3.set_param('parallel.enable', True)
        self.N = N
        self.M = M
        self.Q = Q
        self.DEBUG = DEBUG
        self.norm = norm
        self.P = P             # \# of discrete control inputs

        # XXX: Initialse the solver
        self.s = Optimize()
        # self.s.set(timeout=6000)

        # Initialise the variables for each step N
        self.xs = [Real('x_%s_%s' % (j+1, i))
                   for i in range(N+1)
                   for j in range(M)]

        # Make the control variables
        self.us = [Real('u_%s_%s' % (j+1, i))
                   for i in range(N)
                   for j in range(Q)]

        # Make the discrete control variables
        self.gs = [Int('g_%s_%s' % (j+1, i))
                   for i in range(N)
                   for j in range(P)]

        # Add the evolution of the state variables
        k = 0
        p = 0
        for i in range(0, len(self.xs)-M, M):
            xss = self.xs[i:i+M]
            uss = self.us[k:k+Q]
            gss = self.gs[p:p+P]
            for j, f in enumerate(fs):
                self.s.add(f(xss+uss+gss) == self.xs[i+j+M])
            k += Q
            p += P

        # Add the linear constraints for system states
        consxl = consxl*N if len(consxl) == M else consxl
        [self.s.add(self.xs[i+M] >= consxl[i])
         for i in range(len(consxl)) if consxl[i] is not None]
        consxr = consxr*N if len(consxr) == M else consxr
        [self.s.add(self.xs[i+M] <= consxr[i])
         for i in range(len(consxr)) if consxr[i] is not None]

        # Add the linear constraints for control inputs
        consul = consul*N if len(consul) == Q else consul
        [self.s.add(self.us[i] >= consul[i])
         for i in range(len(consul)) if consul[i] is not None]
        consur = consur*N if len(consur) == Q else consur
        [self.s.add(self.us[i] <= consur[i])
         for i in range(len(consur)) if consur[i] is not None]

        # Add the constraints for the discrete control inputs
        Pset = Pset*N if len(Pset) == P else Pset
        for i in range(len(Pset)):
            if isinstance(Pset[i], set):
                gors = [self.gs[i] == j
                        for j in Pset[i]]
                if len(gors) != 0:
                    self.s.add(Or(gors))
            else:
                raise Exception('Discrete variables need to be bounded  \
                using sets')

        # XXX: Make the minimisation objective
        self.obj = Real('objective')

    def add_objective(self, refx, refu, w, la, refg, gl):
        # Now add the cost function
        refx = [j for i in refx for j in i]  # flatten list
        w = w*self.N if len(w) == self.M else w
        refu = [j for i in refu for j in i]
        la = la*self.N if len(la) == self.Q else la
        # XXX: Discrete control inputs
        refg = [j for i in refg for j in i]
        gl = gl*self.N if len(gl) == self.P else gl

        if self.norm == 2:
            Dxs = [w[i]*(self.xs[i+self.M]-refx[i])**2
                   for i in range(len(refx))]
            Dus = [la[i]*(self.us[i]-refu[i])**2
                   for i in range(len(refu))]
            Dgs = [gl[i]*(self.gs[i]-refg[i])**2
                   for i in range(len(refg))]
            self.s.add(self.obj == (sum(Dxs) + sum(Dus) + sum(Dgs)))

        elif self.norm == 1:
            Dxs = [w[i]*MPC.mabs(self.xs[i+self.M]-refx[i])
                   for i in range(len(refx))]
            Dus = [la[i]*MPC.mabs(self.us[i]-refu[i])
                   for i in range(len(refu))]
            Dgs = [gl[i]*MPC.mabs(self.gs[i]-refg[i])
                   for i in range(len(refg))]
            self.s.add(self.obj == (sum(Dxs) + sum(Dus) + sum(Dgs)))

        # XXX: Infinity norm
        elif self.norm is None:
            Dxs = [w[i]*MPC.mabs(self.xs[i+self.M]-refx[i])
                   for i in range(len(refx))]
            Dus = [la[i]*MPC.mabs(self.us[i]-refu[i])
                   for i in range(len(refu))]
            Dgs = [gl[i]*MPC.mabs(self.gs[i]-refg[i])
                   for i in range(len(refg))]

            Dxxs = [MPC.mreduce(0, Dxs[i:i+self.M])
                    for i in range(0, len(Dxs), self.M)]
            Duus = [MPC.mreduce(0, Dus[i:i+self.Q])
                    for i in range(0, len(Dus), self.Q)]
            if self.P != 0:
                Dggs = [MPC.mreduce(0, Dgs[i:i+self.P])
                        for i in range(0, len(Dgs), self.P)]
            else:
                Dggs = []
            self.s.add(self.obj == sum(Dxxs + Duus + Dggs))
            # self.s.add(self.obj ==
            # (MPC.mreduce(0, Dxs) + MPC.mreduce(0, Dus)))
        pass

    @staticmethod
    def mabs(x):
        return If(x >= 0, x, -x)

    @staticmethod
    def mreduce(sV, Ej):
        def max(x, y):
            return If(x > y, x, y)

        if len(Ej) == 0:
            return sV
        else:
            return max(Ej[0], MPC.mreduce(sV, Ej[1:]))

    @staticmethod
    def getnum(val):
        if isinstance(val, AlgebraicNumRef):
            objv = val.approx(10)
        else:
            objv = val
        objv = (objv.numerator_as_long() / objv.denominator_as_long())
        return objv

    def solve(self, IS, refx, refu, wx, wu, plan=False, opt=True,
              refg=[], wg=[]):

        self.s.push()           # put a point to push
        # XXX: Add the objective constraint
        self.add_objective(refx, refu, wx, wu, refg, wg)

        # Add the constraint for the inital_state
        for i, j in enumerate(IS):
            if isinstance(j, set):
                if len(j) >= 2:
                    self.s.add(self.xs[i] >= j.pop())
                    self.s.add(self.xs[i] <= j.pop())
                else:
                    raise Exception('Cannot initialise using IS: ', IS)
            else:
                self.s.add(self.xs[i] == j)

        if self.DEBUG:
            import sys
            osout = sys.stdout
            with open('/tmp/DEBUG.txt', 'a') as f:
                sys.stdout = f
                print(self.s.to_smt2())
                sys.stdout = osout
        # XXX: The state of the solver
        if opt:
            self.s.minimize(self.obj)
        res = self.s.check()
        if res == sat:
            # XXX: Now minimize the objective function
            # This is current objective value upper bound
            objv = MPC.getnum(self.s.model()[self.obj])
            res, model = objv, self.s.model()
            if self.DEBUG:
                print('---------------answer----------------------')
                print(res, '\n', model)

            # XXX: Get the control vector without the zeroth time
            toret = [MPC.getnum(model[ret]) for ret in self.us]
            toretg = [int(str(model[ret])) for ret in self.gs]
            self.s.pop()
            if plan:
                traj = [MPC.getnum(model[i]) for i in self.xs]
            return (toret, toretg, traj) if plan else (toret[:self.Q],
                                                       toretg[:self.P])
        else:
            raise Exception('Model cannot be satisfied')
