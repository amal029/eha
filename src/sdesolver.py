#!/usr/bin/env python3

import numpy as np
import mpmath as mp
import sympy as S
import mpmath as M
from scipy import optimize


class Solver(object):
    def __init__(self, T=None, Tops=None, A=None, B=None, S=None, SB=None,
                 R=4, C=0.1, gstep=1/2**14, montecarlo=False):
        assert R > 1
        assert R % 2 == 0
        self.R = R
        # L is the number of locations
        # _ should always be 2, for start and end
        # N is the number of continuous variables in the system
        (L, _, N) = T.shape
        self.T = T
        self.L = L
        self.N = N
        self.Tops = Tops

        # There should always be a start and end for each variable.
        assert _ == 2

        # Now check that A is of size L x N x N
        assert A.shape == (L, N, N)
        self.A = A

        # Now check that B is of size L x N
        assert B.shape == (L, N)
        self.B = B
        # print(B[0])

        # Now check that S is of size N x N
        assert S.shape == (L, N, N)
        self.S = S

        # Now check that SB is of size N x 1
        assert SB.shape == (L, N)
        self.SB = SB

        # The random path for montecarlo comparison
        self.montecarlo = montecarlo
        self.path = None
        self.dts = None
        self.locs = None

        # The time step to push the system into brownian when you reach
        # a stable point.
        self.gstep = gstep

        # The constant for tolerance
        self.C = C

    @staticmethod
    def _compute_step(fxt, gxt, dq, R, dWt):
        # print('compute step:', dq)

        Winc = sum(dWt)
        gn = gxt * Winc
        # This is the max dq we can take
        # FIXME: IMP
        # This can be negative, or zero then what happens?
        dq2 = abs(gn**2 / (4 * fxt * R))
        # odq = dq
        dq = dq if dq <= dq2 else dq2
        # print('Given dq: %f, chosen dq: %f' % (odq, dq))
        # first coefficient
        a = R * (fxt**2)
        # Second coefficient
        b = ((2 * fxt * dq * R) - (gn**2))
        # Third coefficient
        c = R*(dq**2)

        # Use mpmath to get the roots
        # There can be only one root.
        f = (lambda x: a*(x**2) + (b*x) + c)
        try:
            root1 = mp.findroot(f, 0, tol=1e-14, solver='mnewton')
            # Debug
            # print('root1:', root1)
            root1 = root1 if root1 >= 0 else None
        except ValueError:
            # print(e)
            root1 = None

        # The second polynomial ax² - bx + cx = 0
        b = ((2 * fxt * dq * R) + (gn**2))
        f = (lambda x: a*(x**2) - (b*x) + c)
        try:
            root2 = mp.findroot(f, 0, tol=1e-14, solver='mnewton')
            # print('root2:', root2)
            root2 = root2 if root2 >= 0 else None
        except ValueError:
            # print(e)
            root2 = None

        # Now get Δt and δt
        Dt = root1 if root1 is not None else root2
        Dt = min(Dt, root2) if root2 is not None else Dt
        dt = Dt/R
        # print('Δt: %f, δt: %f' % (Dt, dt))

        # assert False
        return Dt, dt

    def getloc(self, cvs):
        # Step-1 check in what location are the current values in?
        loc = 0
        while loc < self.L:
            left = self.T[loc][0]
            right = self.T[loc][1]
            lop = self.Tops[loc][0]
            rop = self.Tops[loc][1]
            # Now check if cvs are within this range?
            zl = [i[0](*i[1:]) for i in zip(lop, cvs, left)]
            zr = [i[0](*i[1:]) for i in zip(rop, cvs, right)]
            if all(zl) and all(zr):
                break
            loc += 1
        return loc, left, right

    def _get_step(self, x, index, loc, curr_fxt, curr_gxt, dq, dWt, R,
                  epsilon=1e-14):
        """The iterative process that gets the time step that satisfies this
        scalar continuous variable

        """
        dq = dq
        while(True):
            xtemp = x
            xtemph = x
            # xt = x
            # print('with dq:', dq)
            Dt, dt = Solver._compute_step(curr_fxt, curr_gxt, dq=dq, dWt=dWt,
                                          R=R)
            # print('Dt: %s, dt: %s' % (Dt, dt))
            # Now compute x(t) using Euler-Maruyama solution to get x(t)
            # First build the weiner process
            Winc = np.sqrt(dt) * sum(dWt)

            # EM
            # print(xtemp)
            Fxts = (np.dot(self.A[loc], xtemp) + self.B[loc]) * Dt
            Gxts = (np.dot(self.S[loc], xtemp) + self.SB[loc]) * Winc

            # print(Fxts + Gxts)
            xtemp = xtemp + Fxts + Gxts
            # xtemp += np.array([(Dt * curr_fxt) + (curr_gxt * Winc)]*len(x))
            # print(xtemp)

            # XXX: Check this one, does this make sense?
            # Try taking half steps and see what happens.
            # The first step until R/2
            Fxts = np.dot(self.A[loc], xtemph) + self.B[loc]
            Gxts = np.dot(self.S[loc], xtemph) + self.SB[loc]
            part = Gxts * np.sqrt(dt) * sum(dWt[0:R//2])
            xtemph = xtemph + (Dt/2 * Fxts) + part

            # The second step until R
            Fxts = np.dot(self.A[loc], xtemph) + self.B[loc]
            Gxts = np.dot(self.S[loc], xtemph) + self.SB[loc]
            part = Gxts * np.sqrt(dt) * sum(dWt[R//2:R])
            xtemph = xtemph + (Dt/2 * Fxts) + part
            # for dwt in dWt:
            #     Fxts = np.dot(self.A[loc], xtemph) + self.B[loc]
            #     Gxts = np.dot(self.S[loc], xtemph) + self.SB[loc]
            #     part = Gxts * np.sqrt(dt) * dwt
            #     xtemph = xtemph + (dt * Fxts) + part

            dt = float(dt)
            # tol = self.C * np.sqrt(1 + np.log(1/dt))*np.sqrt(dt)
            # err = np.sqrt(np.sum(np.square(xtemph - xtemp)))
            err = (np.sum(np.abs((xtemp - xtemph)/(xtemph + epsilon)))
                   <= self.C)   # XXX: This gives the best results.
            # err = np.all(np.abs(xtemp - xtemph) <=
            #              (self.C * np.abs(x)) + epsilon)
            if err:
                break
            else:
                # print('reducing dq')
                # Can we do better than this?
                dq = dq/2       # Half it and try again
        return dt

    def simulate(self, values, simtime, epsilon=1e-6):
        # Step-1
        curr_time = 0
        vs = [values]
        ts = [curr_time]
        while(curr_time <= simtime):
            cvs = vs[-1].copy()  # The current values
            # Step-1 check in what location are the current values in?
            loc, left, right = self.getloc(cvs)

            # First get the current value of the slope in this location
            Fxts = np.dot(self.A[loc], cvs) + self.B[loc]
            Gxts = np.dot(self.S[loc], cvs) + self.SB[loc]

            # System matrix does not change.
            condf = any([i != 0 for i in Fxts])

            # Browninan does not change.
            condg = any([i != 0 for i in Gxts])

            # Create dWt
            dWt = np.random.randn(self.R)

            # Now compute the steps
            dts = [None]*len(cvs)
            for i, (fxt, gxt) in enumerate(zip(Fxts, Gxts)):
                # print(i, left[i], right[i])
                if abs(left[i]) != np.inf:
                    dq = abs(left[i] - cvs[i])
                    # This is required to simulate if diffusion changes,
                    # all the time.
                    if (dq > epsilon) and condf:
                        dtl = self._get_step(cvs, i, loc, fxt, gxt,
                                             dq, dWt, self.R)
                    elif condg:
                        dtl = self.gstep
                    else:
                        dtl = 0
                else:
                    dtl = np.inf
                if abs(right[i]) != np.inf:
                    dq = abs(right[i] - cvs[i])
                    if dq > epsilon and condf:
                        dtr = self._get_step(cvs, i, loc, fxt, gxt,
                                             dq, dWt, self.R)
                    elif condg:
                        dtr = self.gstep
                    else:
                        dtr = 0
                else:
                    dtr = np.inf
                dts[i] = min(dtl, dtr)

            dts = [i for i in dts if i != 0]  # Remove all 0s

            # If we have reached the stable point and nothing can change
            if dts == []:
                break
            else:
                dt = min(dts)
            # print(Dt, dt)
            # Now compute the steps for each scalar separately.
            # print(cvs)
            cvs += (self.R*dt * Fxts) + Gxts * np.sqrt(dt) * sum(dWt)
            # print(cvs)
            # print(vs)
            vs.append(cvs)
            # print(vs)
            # assert False

            # Increment time
            curr_time += self.R * dt
            ts.append(curr_time)
            if self.montecarlo:
                self.dts = (np.array([dt]*len(dWt)) if self.dts is None
                            else np.append(self.dts, np.array([dt]*len(dWt))))
                self.path = (dWt*np.sqrt(dt) if self.path is None
                             else np.append(self.path, dWt*np.sqrt(dt)))
                self.locs = (np.array([loc]*len(dWt)) if self.locs is None
                             else np.append(self.locs,
                                            np.array([loc]*len(dWt))))
            print(curr_time)
        return vs, ts

    # @njit
    def nsimulate(self, values):
        """This is the naive simulation using Euler-Maruyama on the same random
        path as the quantized state solution.

        """
        curr_time = 0
        vs = [values]
        ts = [curr_time]
        for dt, dwt, loc in zip(self.dts, self.path, self.locs):
            cvs = vs[-1].copy()
            # Get the current value of the slope in this location
            Fxts = np.dot(self.A[loc], cvs) + self.B[loc]
            Gxts = np.dot(self.S[loc], cvs) + self.SB[loc]

            # Now just compute the Euler-Maruyama equation
            cvs = cvs + (Fxts * dt) + (Gxts * dwt)
            vs.append(cvs)

            # Increment the time
            curr_time += dt
            ts.append(curr_time)
        return vs, ts


class Compute:
    """This is the main adaptive time step computation class for generalised
    stochastic hybrid systems.

    """
    # Static error bound
    epsilon = 1e-3
    iter_count = 50
    DEFAULT_STEP = 1
    p = 3
    R = 2**p
    ROOT_FUNC = 'scipy'

    @staticmethod
    def var_compute(deps=None, dWts=None, vars=None,
                    T=0, Dtv=None, dtv=None):
        # print(dWts, left, right, deps, vars, Dtv, dtv)
        # Taking one big step Dtv
        temp1 = {i: Compute.EM(vars[i], deps[i][0], deps[i][1],
                               Dtv, dtv, dWts[i], vars, T)
                 for i in vars}

        # Taking step to Dtv/2
        nvars = {i: Compute.EM(vars[i], deps[i][0], deps[i][1],
                               Dtv/2, dtv, dWts[i][0:Compute.R//2], vars,
                               T)
                 for i in vars}
        # Taking step from Dtv/2 --> Dtv
        nvars = {i: Compute.EM(nvars[i], deps[i][0], deps[i][1],
                               Dtv/2, dtv, dWts[i][Compute.R//2:Compute.R],
                               nvars, T+(Dtv/2))
                 for i in vars}
        errs = list((np.sum(np.abs((temp1[i] - nvars[i]) /
                                   (nvars[i] + Compute.epsilon)))
                     <= Compute.epsilon) for i in nvars)
        return all(errs), temp1, nvars

    @staticmethod
    def build_eq(f, K):
        eq = f - K
        eq = eq.expand().evalf()
        return eq

    @staticmethod
    def build_eq_g(dt, fp, sp, K):
        f = fp + sp
        dtc = f.collect(dt).coeff(dt, 1)
        eq = (f - dtc*dt)**2 - (K - dtc*dt)**2
        eq = eq.expand().evalf()
        # print(eq)
        # raise Exception
        return eq

    @staticmethod
    # XXX: We should convert the radical function into a poylnomial
    def getroot(dt, eq1, eq2, expr):
        if Compute.ROOT_FUNC == 'mpmath':
            leq1 = S.lambdify(dt, eq1, [{'sqrt': M.sqrt}, 'mpmath'])
            try:
                root1 = M.findroot(leq1, 0.0, solver='secant',
                                   tol=Compute.epsilon,
                                   verify=True)
            except ValueError:
                root1 = None
        else:
            leq1 = S.lambdify(dt, eq1, 'scipy')
            root1 = optimize.root(lambda x: leq1(x[0]), 0, method='hybr')
            if root1.success:
                root1 = root1.x[0]
                # print('r1:', root1)
            else:
                root1 = None
        if root1 is not None and M.im(root1) <= Compute.epsilon:
            root1 = M.re(root1) if M.re(root1) > 0 else None
        else:
            root1 = None

        if Compute.ROOT_FUNC == 'mpmath':
            leq2 = S.lambdify(dt, eq2, [{'sqrt': M.sqrt}, 'mpmath'])
            try:
                root2 = M.findroot(leq2, 0, solver='secant',
                                   tol=Compute.epsilon,
                                   verify=True)
            except ValueError:
                root2 = None
        else:
            leq2 = S.lambdify(dt, eq2, 'scipy')
            root2 = optimize.root(lambda x: leq2(x[0]), 0, method='lm')
            if root2.success:
                root2 = root2.x[0]
                # print('r2:', root2)
            else:
                root2 = None
        if root2 is not None and M.im(root2) <= Compute.epsilon:
            root2 = M.re(root2) if M.re(root2) > 0 else None
        else:
            root2 = None
        Dtv = None
        if root1 is not None and root2 is not None:
            Dtv = min(root1, root2)
        elif root1 is not None:
            Dtv = root1
        elif root2 is not None:
            Dtv = root2
        else:
            print('Non positive root detected for %s' % expr)
            print('root1: %s, root2: %s' % (root1, root2))
        # print('Dtv:', Dtv)
        return Dtv

    @staticmethod
    def rate_compute(left=None, right=None, deps=None, Uz=None, vars=None,
                     T=0, dWts=None):
        t = S.var('t')
        Dt = S.var('T')        # This will be the time step
        if not right[1] == 0:
            raise Exception(('Rate %s cannot be a stochastic DE' % left))
        if Uz is None or Uz == np.inf:
            return np.inf, vars

        # Now start computing the actual step
        right = right[0]
        zdt = right             # This the first derivative

        # Now replace vars with current values
        for i, j in vars.items():
            zdt = zdt.replace(i, j)
            # z2dt = z2dt.replace(i, j)
        # Finally replace t if left over with current value T
        zdt = zdt.replace(t, T)*Dt

        # Now doing the two sided root finding
        L = (Uz - vars[left])   # This is the level crossing
        f = zdt

        # FIXME: If the derivative is zero then it will never reach the
        # level. Change this later if needed
        if f == 0:
            return np.inf, vars

        count = 0
        while(True):
            eq1 = Compute.build_eq(f, L)
            # leq1 = S.lambdify(Dt, eq1, 'scipy')
            # This is the second equation
            eq2 = Compute.build_eq(f, -L)
            # leq2 = S.lambdify(Dt, eq2, 'scipy')
            Dtv = Compute.getroot(Dt, eq1, eq2, left.diff(t))
            # XXX: This can be problematic
            if Dtv is None:
                return np.inf, vars
            dtv = Dtv/Compute.R
            # Now check of the error bound is met using standard
            # Euler-Maruyama
            # FIXME: Somehow the z variable is not being computed correctly!
            err, z1s, z2s = Compute.var_compute(deps, dWts, vars, T, Dtv, dtv)

            # XXX: We need to make sure that other variables are also
            # satisfied.
            if err:
                # print('Found rate step z(t):', Dtv)
                return Dtv, z1s
            else:
                count += 1
                if count == Compute.iter_count:
                    raise Exception('Too many iterations Rate compute')
                L = L/2

    @staticmethod
    def EM(init, f, g, Dt, dt, dWts, vars, T):
        f = f.subs(vars).subs(S.var('t'), T)
        g = g.subs(vars).subs(S.var('t'), T)
        res = (init + f*Dt + g*np.sqrt(dt)*np.sum(dWts)).evalf()
        return res

    @staticmethod
    def default_compute(deps, dWts, vars, T):
        Dtv = Compute.DEFAULT_STEP
        while(True):
            dtv = Dtv/Compute.R
            err, z1s, z2s = Compute.var_compute(deps, dWts, vars, T, Dtv, dtv)
            if err:
                return Dtv, z1s
            else:
                Dtv /= 2

    @staticmethod
    def guard_compute(expr=None, deps=None, vars=None, T=0,
                      dWts=None, Dz=None):
        t = S.var('t')
        dt = S.var('dt')
        dWt = {i: S.var('dWt_%s' % str(i.func)) for i in vars}
        kvars = list(vars.keys())
        dvars = S.Matrix(1, len(kvars), [i.diff(t) for i in kvars])

        # XXX: First compute all the partial derivatives that we need.
        jacobian = S.Matrix([expr]).jacobian(kvars)
        # gfirst = [expr.diff(i) for i in kvars]
        # gradient = S.Matrix(len(kvars), 1, gfirst)
        fp = (dvars*jacobian.transpose())[0]

        # Use a hessian matrix for the second order partials
        hessian = S.hessian(expr, kvars)
        sp = 0.5*((dvars*hessian*dvars.transpose())[0])
        # print('fp:', fp)
        # print('sp:', sp)

        # FIXME: This is where we should separate things.
        ddeps = {i.diff(t): deps[i][0]*dt+deps[i][1]*dWt[i] for i in deps}

        # XXX: Now replace the derivates with their equivalents
        fp = fp.subs(ddeps)
        sp = sp.subs(ddeps)
        # print('fp:', fp)
        # print('sp:', sp)

        # XXX: Now replace the vars with the current values
        fp = fp.subs(vars)
        sp = sp.subs(vars)
        # print('fp:', fp)
        # print('sp:', sp)

        # XXX: Substitute any remaining t with T
        fp = fp.subs(t, T)
        sp = sp.subs(t, T)

        # XXX: Now expand the equations
        fp = fp.expand()
        sp = sp.expand()
        # print('fpe:', fp)
        # print('spe:', sp)

        # XXX: Now apply Ito's lemma
        # dWt*dt = dt**2 = 0
        # dWt**2 = dt

        # FIXME: Check this things robustness later on
        fp = fp.subs(dt**2, 0)
        for i in dWt:
            # Now the dodgy one
            fp = fp.subs(dWt[i]*dt, 0)
        for i in dWt:
            fp = fp.subs(dWt[i]**2, dt)
        # print(fp)

        sp = sp.subs(dt**2, 0)
        for i in dWt:
            # Now the dodgy one
            sp = sp.subs(dWt[i]*dt, 0)
        for i in dWt:
            sp = sp.subs(dWt[i]**2, dt)
        # print(sp)

        # Finally, substitute dWts, independently
        ddWts = {str('dWt_%s' % i.func): np.sum(dWts[i])*S.sqrt(dt/Compute.R)
                 for i in dWts}
        fp = fp.subs(ddWts)
        sp = sp.subs(ddWts)
        # print('fp:', fp)
        # print('sp:', sp)

        # XXX: Now get the value of the guart at time T
        gv = expr.subs(vars)
        gv = gv.subs(t, T)
        # print('gv:', gv)

        # XXX: Now we can start solving for the root
        L = -gv

        # XXX: If I use second order here, but then I use a first order
        # approximation when actually doing things then stuff goes
        # wrong.
        # sp = 0

        f = fp + sp

        # XXX: If the derivative is zero then it will never reach the
        # level.
        if f == 0:
            return np.inf, vars

        # XXX: Now the real computation of the time step
        count = 0
        while(True):
            eq1 = Compute.build_eq_g(dt, fp, sp, L)
            eq2 = Compute.build_eq_g(dt, fp, sp, -L)
            Dtv = Compute.getroot(dt, eq1, eq2, expr)

            if Dtv is None:
                print('choosing Dz!', Dz)
            Dtv = min(Dtv, Dz) if Dtv is not None else Dz

            # XXX: Dz might be numpy.inf, because we do not have a z in
            # this HA and we also do not get a real positive root
            if Dtv == np.inf:
                return Dtv, vars

            dtv = Dtv/Compute.R

            # Now check of the error bound is met using standard
            # Euler-Maruyama
            err, v1s, v2s = Compute.var_compute(deps, dWts, vars, T, Dtv, dtv)

            # XXX: We need to make sure that other variables are also
            # satisfied.
            if err:
                # print('Found rate step:', Dtv, v1s)
                return Dtv, v1s
            else:
                count += 1
                if count == Compute.iter_count:
                    raise Exception('Too many iterations %s' % expr)
                L = L/2
