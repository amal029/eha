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
    DEFAULT_STEP = 1e-3

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
SIM_TIME = 1.2

# Defining the dynamics in different modes
# The constants in the HA
v = 4
wv = 0.1
e = 1e-1

# The length of the stochastic path
p = 3
R = 2**p

# Mode move
# We have drift and diffusion dynamics in all cases
dynamics = {'Move': {S.sympify('x(t)'): [v*wv*S.sympify('-sin(th(t))'),
                                         S.sympify('0')],
                     S.sympify('y(t)'): [v*wv*S.sympify('cos(th(t))'),
                                         S.sympify('0')],
                     S.sympify('th(t)'): [S.sympify('0'),
                                          S.sympify('0')],
                     S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Inner': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                          S.sympify('v').subs('v', v)],
                      S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                          S.sympify('v').subs('v', v)],
                      S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                      S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Outter': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                           S.sympify('v').subs('v', v)],
                       S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                           S.sympify('v').subs('v', v)],
                       S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                       S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Changetheta': {S.sympify('x(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('y(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('th(t)'): [S.sympify('2*v').subs('v', v),
                                                 S.sympify('0')],
                            S.sympify('z(t)'): [
                                S.sympify('(x(t)**2+y(t)**2)*th(t)**3'),
                                S.sympify('0')]},
            'ThetaNochange': {S.sympify('x(t)'): [S.sympify('0'),
                                                  S.sympify('0')],
                              S.sympify('y(t)'): [S.sympify('0'),
                                                  S.sympify('0')],
                              S.sympify('th(t)'): [S.sympify('wv').subs('wv',
                                                                        wv),
                                                   S.sympify('0')],
                              S.sympify('z(t)'): [S.sympify('0'),
                                                  S.sympify('0')]}}


# This is the main GSHA
class GSHS:
    """The generalised stochastic hybrid automaton
    """

    @staticmethod
    def __compute(x, y, th, z, t, dWts, location=None, guards=None, Uz=None):
        # The current values at some time T
        vars = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th,
                S.sympify('z(t)'): z}
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
    def Move(x, y, th, z, t, dWts, first_time):
        """Location Move
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e):
            # Set the outputs
            th = 0
            state = 1           # destination location is Inner
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - v**2 >= e):
            th = S.pi
            state = 2           # destination is Outter
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            T, vars = GSHS.__compute(x, y, th, z, t, dWts, 'Move', [], None)
            return 0, T, vars, False

    @staticmethod
    def Inner(x, y, th, z, t, dWts, first_time):
        """Location Inner
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 >= -e) and (x**2 + y**2 - v**2 <= e):
            # Set the outputs
            th = float(M.atan(y/x))
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - v**2 >= e):
            state = 2
            th = S.pi
            return state, 0, (x, y, th, z), True
        else:
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            T, vars = GSHS.__compute(x, y, th, z, t, dWts, 'Inner', [g1], None)
            return 1, T, vars, False

    @staticmethod
    def Outter(x, y, th, z, t, dWts, first_time):
        """Location Outter
        """
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= e) and (x**2 + y**2 - v**2 >= -e):
            # Set the outputs
            th = float(M.atan(y/x))
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif x**2 + y**2 - v**2 <= -e:
            state = 1           # Destination Inner
            th = 0
            return state, 0, (x, y, th, z), True
            pass
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            # g2 = S.sympify('z(t)') - Uz
            T, vars = GSHS.__compute(x, y, th, z, t, dWts, 'Outter', [g1],
                                     None)
            return 2, T, vars, False

    @staticmethod
    def Changetheta(x, y, th, z, t, dWts, first_time):
        """Changetheta location
        """
        global Uz
        Uz = -numpy.log(numpy.random.rand()) if first_time else Uz
        if (x**2 + y**2 - v**2 <= e) and (x**2 + y**2 - v**2 >= -e):
            z = 0
            th = float(M.atan(y/x))
            state = 1           # Dest: ThetaNochange
            return state, 0, (x, y, th, z), True
        elif abs(z - Uz) <= e:
            z = 0
            th = th - Uz
            state = 0           # Dest Changetheta
            return state, 0, (x, y, th, z), True
        else:
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            T, vars = GSHS.__compute(x, y, th, z, t, dWts,
                                     'Changetheta', [g1], Uz)
            return 0, T, vars, False

    @staticmethod
    def ThetaNochange(x, y, th, z, t, dWts, first_time):
        if (x**2 + y**2 - v**2 <= -e) or (x**2 + y**2 - v**2 >= e):
            z = 0
            state = 0           # Dest Changetheta
            return state, 0, (x, y, th, z), True
        else:
            T, vars = GSHS.__compute(x, y, th, z, t, dWts,
                                     'ThetaNochange', [], None)
            return 1, T, vars, False


def main(x, y, th, z, t):

    XY_loc = {
        0: GSHS.Move,
        1: GSHS.Inner,
        2: GSHS.Outter
    }
    XY_strloc = {
        0: 'Move',
        1: 'Inner',
        2: 'Outter',
    }

    THETA_loc = {
        0: GSHS.Changetheta,
        1: GSHS.ThetaNochange
    }

    THETA_strloc = {
        0: 'Changetheta',
        1: 'ThetaNochange'
    }

    # First compute the invariant to decide upon the location I should
    # be in.
    if abs(x**2 + y**2 - v**2) <= e:
        state1 = 0               # Move
    elif x**2 + y**2 - v**2 <= -e:
        state1 = 1               # Inner
    elif x**2 + y**2 - v**2 >= e:
        state1 = 2               # Outter
    else:
        raise Exception('Unknown state reached')

    state2 = 0                  # Always start THETA from change state

    # Print the outputs
    print('%.4f: Locs:(%s, %s), x:%s, y:%s, th:%s, z:%s'
          % (t, XY_strloc[state1], THETA_strloc[state2], x, y, th, z))

    FT1 = True
    FT2 = True
    xs = []
    ys = []
    ts = []
    xy2s = []
    while(True):
        # Create dWt
        dWts = {S.sympify('x(t)'): numpy.random.randn(R),
                S.sympify('y(t)'): numpy.random.randn(R),
                S.sympify('th(t)'): numpy.random.randn(R),
                S.sympify('z(t)'): numpy.zeros(R)}

        vars = {S.sympify('x(t)'): x,
                S.sympify('y(t)'): y,
                S.sympify('th(t)'): th,
                S.sympify('z(t)'): z}

        # Call the dynamics and run these until some time
        nstate1, T1, (x1, y1, th1, z1), FT1 = XY_loc[state1](x, y,
                                                             th, z, t, dWts,
                                                             FT1)
        nstate2, T2, (x2, y2, th2, z2), FT2 = THETA_loc[state2](x, y,
                                                                th, z, t, dWts,
                                                                FT2)
        # Compute the values of x, y, th, z at T
        # FIXME: HOW DO WE HANDLE SHARED VARIABLE UPDATES!
        if T2 <= T1:
            T = T2
            [x, y] = [Compute.EM(vars[i], dynamics[XY_strloc[state1]][i][0],
                                 dynamics[XY_strloc[state1]][i][1],
                                 T, T/R, dWts[i], vars, t)
                      for i in [S.sympify('x(t)'), S.sympify('y(t)')]]
            th = th2
            z = z2
        elif T1 <= T2:
            T = T1
            # XXX: This is a hack!
            if nstate1 != 'Move':
                [th, z] = [Compute.EM(vars[i],
                                      dynamics[THETA_strloc[state2]][i][0],
                                      dynamics[THETA_strloc[state2]][i][1],
                                      T, T/R, dWts[i], vars, t)
                           for i in [S.sympify('th(t)'), S.sympify('z(t)')]]
            x = x1
            y = y1
        # Set the new states
        state1 = nstate1
        state2 = nstate2
        # Append for plotting
        xs.append(x)
        ys.append(y)
        ts.append(t)
        xy2s.append(x**2+y**2)
        t += T

        # Print the outputs
        print('%.4f: Locs:(%s, %s), x:%s, y:%s, th:%s, z:%s'
              % (t, XY_strloc[state1], THETA_strloc[state2], x, y, th, z))

        if t >= SIM_TIME:
            break
    return xs, ys, xy2s, ts


if __name__ == '__main__':
    # the random seed
    numpy.random.seed(0)
    # These are the initial values
    x = 1
    y = 1
    th = S.atan(y/x)
    z = 0
    t = 0
    xs, ys, xy2s, ts = main(x, y, th, z, t)
    print('Count:', len(ts))
    plt.style.use('ggplot')
    # plt.subplot(211)
    # plt.plot(xs, ys, marker='^')
    # plt.xlabel('X (units)', fontweight='bold')
    # plt.ylabel('Y (units)', fontweight='bold')
    # plt.subplot(212)
    plt.plot(ts, xy2s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x^2+y^2$ (units)', fontweight='bold')
    plt.savefig('/tmp/robot.pdf', bbox_inches='tight')
    # plt.show()
