#!/usr/bin/env python3

import numpy as np
import mpmath as mp


class Solver(object):
    def __init__(self, T=None, Tops=None, A=None, B=None, S=None, SB=None,
                 R=4, C=0.1, montecarlo=False):
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
        assert S.shape == (N, N)
        self.S = S

        # Now check that SB is of size N x 1
        assert SB.shape == (N, )
        self.SB = SB

        # The random path for montecarlo comparison
        self.montecarlo = montecarlo
        self.path = None
        self.dts = None

        # The constant for tolerance
        self.C = C

    @staticmethod
    def _compute_step(fxt, gxt, dq, R, dWt):
        # print('compute step:', dq)

        Winc = sum(dWt)
        gn = gxt * Winc
        # This is the max dq we can take
        # FIXME: IMP
        # This can be negative, then what happens?
        dq2 = gn**2 / (4 * fxt * R)
        dq2 = abs(dq2)
        # odq = dq
        dq = dq if dq <= dq2 else dq2
        # print('Given dq: %f, chosen dq: %f' % (odq, dq))
        # First coefficient
        a = R * (fxt**2)
        # Second coefficient
        b = ((2 * fxt * dq * R) - (gn**2))
        # Third coefficient
        c = R*(dq**2)

        # Use mpmath to get the roots
        # There can be only one root.
        f = (lambda x: a*(x**2) + (b*x) + c)
        try:
            root1 = mp.findroot(f, 0)
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
            root2 = mp.findroot(f, 0)
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

    def _get_step(self, x, index, loc, curr_fxt, curr_gxt, dq, dWt, R):
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
            Gxts = (np.dot(self.S, xtemp) + self.SB) * Winc
            # print(Fxts + Gxts)
            xtemp = xtemp + Fxts + Gxts
            # xtemp += np.array([(Dt * curr_fxt) + (curr_gxt * Winc)]*len(x))
            # print(xtemp)

            # XXX: Check this one, does this make sense?
            # Try taking half steps and see what happens.
            # The first step until R/2
            Fxts = np.dot(self.A[loc], xtemph) + self.B[loc]
            Gxts = np.dot(self.S, xtemph) + self.SB
            part = Gxts * np.sqrt(dt) * sum(dWt[0:R//2])
            xtemph = xtemph + (Dt/2 * Fxts) + part

            # The second step until R
            Fxts = np.dot(self.A[loc], xtemph) + self.B[loc]
            Gxts = np.dot(self.S, xtemph) + self.SB
            part = Gxts * np.sqrt(dt) * sum(dWt[R//2:R])
            xtemph = xtemph + (Dt/2 * Fxts) + part

            dt = float(dt)
            tol = self.C * np.sqrt(1 + np.log(1/dt))*np.sqrt(dt)

            # XXX: Here we break it, if error is met,
            # else we half the dq
            err = np.sqrt(np.sum(np.square(xtemph - xtemp)))
            if err <= tol:
                break
            else:
                # Can we do better than this?
                dq = dq/2       # Half it and try again
        return dt

    def simulate(self, values, simtime, epsilon=1e-6):
        # Step-1
        curr_time = 0
        vs = [values]
        ts = [curr_time]
        while(True):
            cvs = vs[len(vs)-1].copy()  # The current values
            # Step-1 check in what location are the current values in?
            # loc = 0
            # while loc < self.L:
            #     left = self.T[loc][0]
            #     right = self.T[loc][1]
            #     lop = self.Tops[loc][0]
            #     rop = self.Tops[loc][1]
            #     # Now check if cvs are within this range?
            #     zl = [i[0](*i[1:]) for i in zip(lop, cvs, left)]
            #     zr = [i[0](*i[1:]) for i in zip(rop, cvs, right)]
            #     if all(zl) and all(zr):
            #         break
            #     loc += 1
            # print('We are in location: %d' % (loc))
            loc, left, right = self.getloc(cvs)

            # First get the current value of the slope in this location
            Fxts = np.dot(self.A[loc], cvs) + self.B[loc]
            Gxts = np.dot(self.S, cvs) + self.SB

            # Create dWt
            dWt = np.random.randn(self.R)

            # Now compute the steps
            dts = [None]*len(cvs)
            for i, (fxt, gxt) in enumerate(zip(Fxts, Gxts)):
                # print(i, left[i], right[i])
                if abs(left[i]) != np.inf:
                    dq = abs(left[i] - cvs[i])
                    if dq != 0:
                        dtl = self._get_step(cvs, i, loc, fxt, gxt,
                                             dq, dWt, self.R)
                    else:
                        dtl = 0
                else:
                    dtl = np.inf
                if abs(right[i]) != np.inf:
                    dq = abs(right[i] - cvs[i])
                    if dq != 0:
                        dtr = self._get_step(cvs, i, loc, fxt, gxt,
                                             dq, dWt, self.R)
                    else:
                        dtr = 0
                else:
                    dtr = np.inf
                dts[i] = min(dtl, dtr)

            # print(dts)
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
            print(curr_time)
            if curr_time >= simtime:
                break
        return vs, ts

    # @njit
    def nsimulate(self, values):
        """This is the naive simulation using Euler-Maruyama on the same random
        path as the quantized state solution.

        """
        curr_time = 0
        vs = [values]
        ts = [curr_time]
        for dt, dwt in zip(self.dts, self.path):
            cvs = vs[-1].copy()
            # First get the location
            loc, _, _ = self.getloc(cvs)
            # Get the current value of the slope in this location
            Fxts = np.dot(self.A[loc], cvs) + self.B[loc]
            Gxts = np.dot(self.S, cvs) + self.SB

            # Now just compute the Euler-Maruyama equation
            cvs = cvs + (Fxts * dt) + (Gxts * dwt)
            vs.append(cvs)

            # Increment the time
            curr_time += dt
            ts.append(curr_time)
        return vs, ts
