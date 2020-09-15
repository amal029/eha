#!/usr/bin/env python3

import sympy as S
import numpy
import mpmath as M
import matplotlib.pyplot as plt
from scipy import optimize


class Compute:
    """This is the main adaptive time step computation class

    """
    # Static error bound
    epsilon = 1e-3
    iter_count = 50
    DEFAULT_STEP = 1

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
                               Dtv/2, dtv, dWts[i][0:R//2], vars,
                               T)
                 for i in vars}
        # Taking step from Dtv/2 --> Dtv
        nvars = {i: Compute.EM(nvars[i], deps[i][0], deps[i][1],
                               Dtv/2, dtv, dWts[i][R//2:R], nvars,
                               T+(Dtv/2))
                 for i in vars}
        errs = list((numpy.sum(numpy.abs((temp1[i] - nvars[i]) /
                                         (nvars[i] + Compute.epsilon)))
                     <= Compute.epsilon) for i in nvars)
        return all(errs), temp1, nvars

    @staticmethod
    def build_eq(f, K):
        eq = f - K
        eq = eq.expand().evalf()
        return eq

    @staticmethod
    def getroot(leq1, leq2, expr):
        # root1 = M.findroot(leq1, 0, solver='secant', tol=Compute.epsilon,
        #                    verify=True)
        root1 = optimize.root(lambda x: leq1(x[0]), 0, method='hybr')
        if root1.success:
            root1 = root1.x[0]
        else:
            root1 = None
            # raise Exception('Could not find a root')
        if root1 is not None and M.im(root1) <= Compute.epsilon:
            root1 = M.re(root1) if M.re(root1) >= 0 else None
        else:
            root1 = None
        root2 = optimize.root(lambda x: leq2(x[0]), 0, method='lm')
        if root2.success:
            root2 = root2.x[0]
        else:
            root2 = None
            # raise Exception('Could not find a root')
        if root2 is not None and M.im(root2) <= Compute.epsilon:
            root2 = M.re(root2) if M.re(root2) >= 0 else None
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
        return Dtv

    @staticmethod
    def rate_compute(left=None, right=None, deps=None, Uz=None, vars=None,
                     T=0, dWts=None):
        t = S.var('t')
        Dt = S.var('T')        # This will be the time step
        if not right[1] == 0:
            raise Exception(('Rate %s cannot be a stochastic DE' % left))
        if Uz is None or Uz == numpy.inf:
            return numpy.inf, vars

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
            return numpy.inf, vars

        count = 0
        while(True):
            eq1 = Compute.build_eq(f, L)
            leq1 = S.lambdify(Dt, eq1, 'scipy')
            # This is the second equation
            eq2 = Compute.build_eq(f, -L)
            leq2 = S.lambdify(Dt, eq2, 'scipy')
            Dtv = Compute.getroot(leq1, leq2, left.diff(t))
            dtv = Dtv/R
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
        res = (init + f*Dt + g*numpy.sqrt(dt)*numpy.sum(dWts)).evalf()
        return res

    @staticmethod
    def default_compute(deps, dWts, vars, T):
        Dtv = Compute.DEFAULT_STEP
        while(True):
            dtv = Dtv/R
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
        ddWts = {str('dWt_%s' % i.func): numpy.sum(dWts[i])*S.sqrt(dt/R)
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
            return numpy.inf, vars

        # XXX: Now the real computation of the time step
        count = 0
        while(True):
            eq1 = Compute.build_eq(f, L)
            leq1 = S.lambdify(dt, eq1, 'scipy')
            eq2 = Compute.build_eq(f, -L)
            leq2 = S.lambdify(dt, eq2, 'scipy')
            Dtv = Compute.getroot(leq1, leq2, expr)

            if Dtv is None:
                print('choosing Dz!', Dz)
            Dtv = min(Dtv, Dz) if Dtv is not None else Dz

            # XXX: Dz might be numpy.inf, because we do not have a z in
            # this HA and we also do not get a real positive root
            if Dtv == numpy.inf:
                return Dtv, vars

            dtv = Dtv/R

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


# Total simulation time
SIM_TIME = 2000.0

# error
e = 1e-1

# The length of the stochastic path
p = 3
R = 2**p

# Constants
kp = 1
ku = 0.001
kd = 0.01
kb = 0.01

dynamics = {'X0': {S.sympify('x(t)'): [kp-kd*S.sympify('x(t)'), S.sympify(0)],
                   S.sympify('z(t)'): [kb*S.sympify('x(t)'), S.sympify(0)]},
            'X1': {S.sympify('x(t)'): [-kd*S.sympify('x(t)'), S.sympify(0)],
                   S.sympify('z(t)'): [S.sympify(ku), S.sympify(0)]}}


class GSHS:
    def __compute(x, z, t, dWts, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x(t)'): x, S.sympify('z(t)'): z}
        # Compute the dynamics in the state
        DM = dynamics[location]

        Dts = dict()
        # XXX: This is for computing the spontaneous jump if any
        Dz, vals = Compute.rate_compute(left=S.sympify('z(t)'),
                                        right=DM[S.sympify('z(t)')],
                                        deps=DM, vars=vars, T=t, Uz=Uz,
                                        dWts=dWts)
        Dts[Dz] = vals

        # XXX: Accounting for the guards
        # This one does not have the spontaneous jump in it.
        for i in guards:
            Dt, Dval = Compute.guard_compute(expr=i, deps=DM, vars=vars, T=t,
                                             dWts=dWts, Dz=Dz)
            Dts[Dt] = Dval

        # XXX: dz might be np.inf if there is no spontaneous output
        # This is the step size we will take
        # XXX: T == Δ == δ*R (assumption)
        k = list(Dts.keys())
        T = min(*k) if len(k) > 1 else k[0]
        if T == numpy.inf:
            T, val = Compute.default_compute(DM, dWts, vars, t)
            return T, val.values()
        else:
            return T, Dts[T].values()

    @staticmethod
    def X0(x, z, t, dWts, FT):
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if FT else Uz
        if (x >= 1) and abs(z - Uz) <= e:
            # Destination X1
            x = x-1
            state = 1
            z = 0
            return state, 0, (x, z), True
        else:
            # g = S.sympify('x(t)')*-kd + kp - 1
            T, vars = GSHS.__compute(x, z, t, dWts, 'X0', [], Uz)
            return 0, T, vars, False

    @staticmethod
    def X1(x, z, t, dWts, FT):
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if FT else Uz
        if abs(z - Uz) <= e:
            # Destination X1
            state = 0
            z = 0
            return state, 0, (x, z), True
        else:
            # g = S.sympify('x(t)')*-kd + kp - 1
            T, vars = GSHS.__compute(x, z, t, dWts, 'X1', [], Uz)
            return 1, T, vars, False


def main(x, z, t):
    X_loc = {
        0: GSHS.X0,
        1: GSHS.X1
    }
    X_strloc = {
        0: 'X0',
        1: 'X1',
    }

    state = 0

    # Print the outputs
    print('%.4f: Locs:%s, x:%s, z:%s' % (t, X_strloc[state], x, z))

    FT1 = True
    xs = []
    ts = []
    while(True):
        # Create dWt
        dWts = {S.sympify('x(t)'): numpy.zeros(R),
                S.sympify('z(t)'): numpy.zeros(R)}

        # Call the dynamics and run these until some time
        state, T, (x, z), FT1 = X_loc[state](x, z, t, dWts, FT1)

        xs.append(x)
        ts.append(t)
        t += T

        print('%.4f: Locs:%s, x:%s, z:%s' % (t, X_strloc[state], x, z))
        if t >= SIM_TIME:
            return xs, ts


if __name__ == '__main__':
    numpy.random.seed(0)
    x = 0
    z = 0
    t = 0
    xs, ts = main(x, z, t)
    print('count:', len(ts))
    plt.style.use('ggplot')
    plt.plot(ts, xs)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x$ (units)', fontweight='bold')
    # plt.savefig('/tmp/robot.pdf', bbox_inches='tight')
    plt.show()
