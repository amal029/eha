#!/usr/bin/env python3
import sympy as S
import numpy as np
import mpmath as M


class Compute:
    """This is the main adaptive time step computation class

    """
    # Static error bound
    epsilon = 1e-3
    iter_count = 50

    @staticmethod
    def var_compute(deps=None, dWts=None, vars=None,
                    T=0, Dtv=None, dtv=None):
        # print(dWts, left, right, deps, vars, Dtv, dtv)
        # Taking one big step Dtv
        temp1 = {i: Compute.EM(vars[i], deps[i][0], deps[i][1],
                               Dtv, dtv, dWts[i], vars, T, i)
                 for i in vars}

        # Taking step to Dtv/2
        nvars = {i: Compute.EM(vars[i], deps[i][0], deps[i][1],
                               Dtv/2, dtv, dWts[i][0:R//2], vars,
                               T, i)
                 for i in vars}
        # Taking step from Dtv/2 --> Dtv
        nvars = {i: Compute.EM(nvars[i], deps[i][0], deps[i][1],
                               Dtv/2, dtv, dWts[i][R//2:R], nvars,
                               T+(Dtv/2), i)
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
    def rate_compute(left=None, right=None, deps=None, Uz=None, vars=None,
                     T=0, dWts=None):
        # XXX: We will only do 2ⁿᵈ order Taylor polynomials everywhere.
        # XXX: First compute the second derivative of the right
        t = S.var('t')
        Dt = S.var('T')        # This will be the time step
        if not right[1] == 0:
            raise Exception(('Rate %s cannot be a stochastic DE' % left))
        if Uz is None or Uz == np.inf:
            return np.inf

        # Now start computing the actual step
        right = right[0]
        zdt = right             # This the first derivative

        # Now replace vars with current values
        for i, j in vars.items():
            zdt = zdt.replace(i, j)
            # z2dt = z2dt.replace(i, j)
        # Finally replace t if left over with current value T
        zdt = zdt.replace(t, T)*Dt
        # XXX: The below is zero because of Ito's lemma
        z2dt = 0  # z2dt.replace(t, T)*(Dt**2)/2

        # Now doing the two sided root finding
        L = (Uz - vars[left])   # This is the level crossing
        f = zdt + z2dt          # This is the part without the Level
        count = 0
        while(True):
            eq1 = Compute.build_eq(f, L)
            leq1 = S.lambdify(Dt, eq1)
            root1 = M.findroot(leq1, 0, solver='secant', tol=Compute.epsilon,
                               verify=False)
            if M.im(root1) <= Compute.epsilon:
                root1 = M.re(root1) if M.re(root1) >= 0 else None
            else:
                root1 = None

            # This is the second equation
            eq2 = Compute.build_eq(f, -L)
            leq2 = S.lambdify(Dt, eq2)
            root2 = M.findroot(leq2, 0, solver='secant', tol=Compute.epsilon,
                               verify=False)
            if M.im(root2) <= Compute.epsilon:
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
                raise Exception('Complex root detected for %s' % left.diff(t))
            dtv = Dtv/R           # This is the small dt

            # Now check of the error bound is met using standard
            # Euler-Maruyama
            err, z1s, z2s = Compute.var_compute(deps, dWts, vars, T, Dtv, dtv)

            # XXX: We need to make sure that other variables are also
            # satisfied.
            if err:
                print('Found rate step z(t):', Dtv)
                return Dtv, z1s
            else:
                count += 1
                if count == Compute.iter_count:
                    raise Exception('Too many iterations Rate compute')
                L = L/2

    @staticmethod
    def EM(init, f, g, Dt, dt, dWts, vars, T, v):
        for i in vars:
            f = f.subs(i, vars[i])
            g = g.subs(i, vars[i])
        res = init + f*Dt + g*np.sqrt(dt)*np.sum(dWts)
        return res.subs(S.var('t'), T).evalf()

    @staticmethod
    def guard_compute(expr=None, deps=None, vars=None, T=0,
                      dWts=None, Dz=None):
        t = S.var('t')
        dt = S.var('dt')
        dWt = {i: S.var('dWt_%s' % str(i.func)) for i in vars}
        kvars = list(vars.keys())
        dvars = S.Matrix(1, len(kvars), [i.diff(t) for i in kvars])

        # XXX: First compute all the partial derivatives that we need.
        gfirst = [expr.diff(i) for i in kvars]
        gradient = S.Matrix(len(kvars), 1, gfirst)
        fp = (dvars*gradient)[0]

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
        ddWts = {str('dWt_%s' % i.func): np.sum(dWts[i])*S.sqrt(dt/R)
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
        f = fp + sp
        count = 0
        while(True):
            eq = Compute.build_eq(f, L)
            # print(eq)
            roots = S.solve(eq, dt)
            root = min(list(filter(lambda x: x >= 0, roots)))
            # print(root)
            if M.im(root) <= Compute.epsilon:
                root = M.re(root) if M.re(root) >= 0 else None
            else:
                raise Exception('Cannot find a real +ve root for guard: %s'
                                % expr)
            Dtv = Dz if Dz <= root else root
            dtv = Dtv/R

            # Now check of the error bound is met using standard
            # Euler-Maruyama
            err, v1s, v2s = Compute.var_compute(deps, dWts, vars, T, Dtv, dtv)

            # XXX: We need to make sure that other variables are also
            # satisfied.
            if err:
                print('Found rate step:', Dtv)
                return Dtv, v1s
            else:
                count += 1
                if count == Compute.iter_count:
                    raise Exception('Too many iterations %s' % expr)
                L = L/2


# Total simulation time
SIM_TIME = 2

# Defining the dynamics in different modes
# The constants in the HA
v = 4
wv = 0.1
e = 1e-7

# The length of the stochastic path
p = 3
R = 2**p

# Mode move
# We have drift and diffusion dynamics in all cases
dynamics = {'Move': {S.sympify('x(t)'): [v*wv*S.sympify('-sin(th(t))'),
                                         S.sympify('0')],
                     S.sympify('y(t)'): [v*wv*S.sympify('cos(th(t))'),
                                         S.sympify('0')],
                     S.sympify('th(t)'): [S.sympify('wv').subs('wv', wv),
                                          S.sympify('0')],
                     S.sympify('z(t)'): [S.sympify('0'), S.sympify('0')]},
            'Inner': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                          S.sympify('v').subs('v', v)],
                      S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                          S.sympify('v').subs('v', v)],
                      S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                      S.sympify('z(t)'): [S.sympify('x(t)'), S.sympify('0')]},
            'Outter': {S.sympify('x(t)'): [v*S.sympify('cos(th(t))'),
                                           S.sympify('v').subs('v', v)],
                       S.sympify('y(t)'): [v*S.sympify('sin(th(t))'),
                                           S.sympify('v').subs('v', v)],
                       S.sympify('th(t)'): [S.sympify('0'), S.sympify('0')],
                       S.sympify('z(t)'): [S.sympify('x(t)'), S.sympify('0')]},
            'Changetheta': {S.sympify('x(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('y(t)'): [S.sympify('0'),
                                                S.sympify('0')],
                            S.sympify('th(t)'): [S.sympify('v').subs('v', v),
                                                 S.sympify('0')],
                            S.sympify('z(t)'): [S.sympify('th(t)**3'),
                                                S.sympify('0')]}}


# This is the main GSHA
def main(x, y, th, z, t):
    """The generalised stochastic hybrid automaton
    """
    # The current values at some time T
    vars = {S.sympify('x(t)'): x,
            S.sympify('y(t)'): y,
            S.sympify('th(t)'): th,
            S.sympify('z(t)'): z}

    def __compute(location=None, guards=None, Uz=None):
        global th, z, x, y, t
        # Compute the dynamics in the state
        DM = dynamics[location]

        # Create dWt
        dWts = {S.sympify('x(t)'): np.random.randn(R),
                S.sympify('y(t)'): np.random.randn(R),
                S.sympify('th(t)'): np.random.randn(R),
                S.sympify('z(t)'): np.zeros(R)}

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
        return T, Dts[T].values()

    def Move(first_time):
        """Location Move
        """
        global th, z, x, y, t
        # First compute the outgoing edges and take them
        if (abs(x**2 + y**2 - v**2) <= -e):
            # Set the outputs
            th, z = 0, 0
            state = 1           # destination location is Inner
            return state, 0, (x, y, th, z), True
        elif (abs(x**2 + y**2 - v**2) >= e):
            th, z = S.pi, 0
            state = 2           # destination is Outter
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - (v**2 - e)
            g2 = S.sympify('x(t)**2 + y(t)**2') - (v**2 + e)
            T, vars = __compute('Move', [g1, g2], None)
            return 0, T, vars, False

    def Inner(first_time):
        """Location Inner
        """
        global th, z, x, y, t
        if first_time:
            # This is the spontaneous jump condition
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if (abs(x**2 + y**2 - v**2) <= e):
            # Set the outputs
            th = np.arctan(y/x)
            z = 0
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            # g2 = S.sympify('z(t)') - Uz
            T, vars = __compute('Inner', [g1], Uz)
            return 1, T, vars, False

    def Outter(first_time):
        global th, z, x, y, t
        if first_time:
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if abs(x**2 + y**2 - v**2) <= e:
            # Set the outputs
            th = np.arctan(y/x)
            z = 0
            state = 0           # destination location is Move
            return state, 0, (x, y, th, z), True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            # g2 = S.sympify('z(t)') - Uz
            T, vars = __compute('Outter', [g1], Uz)
            return 2, T, vars, False

    def Changetheta(first_time):
        global th, z, x, y, t
        if first_time:
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e) and abs(z - Uz) <= e:
            # Set the outputs
            z = 0
            state = 1           # destination location is Inner
            return state, 0, (x, y, th, z), True
        elif (x**2 + y**2 - v**2 >= e) and abs(z - Uz) <= e:
            z = 0
            state = 2           # destination is Outter
            return state, 0, (x, y, th, z), True
        else:
            # XXX: Accounting for the guards
            # g1 = S.sympify('z(t)') - Uz
            T, vars = __compute('Changetheta', [], Uz)
            return 3, T, vars, False

    locations = {
        0: Move,
        1: Inner,
        2: Outter,
        3: Changetheta
    }
    strloc = {
        0: 'Move',
        1: 'Inner',
        2: 'Outter',
        3: 'Changetheta'
    }

    # First compute the invariant to decide upon the location I should
    # be in.
    if abs(x**2 + y**2 - v**2) <= e:
        state = 0
    elif x**2 + y**2 - v**2 <= -e:
        state = 1
    elif x**2 + y**2 - v**2 >= e:
        state = 2
    else:
        raise Exception('Unknown state reached')

    # Print the outputs
    print('%.4f: state:%s, x:%s, y:%s, th:%s, z:%s' % (t, strloc[state], x, y,
                                                       th, z))
    while(True):
        # Call the dynamics and run these until some time
        first_time = True
        state, T, (x, y, th, z), first_time = locations[state](first_time)

        t += T
        # Print the outputs
        print('%f: state:%s, x:%s, y:%s, th:%s, z:%s' % (t, strloc[state],
                                                         x, y, th, z))

        if t >= SIM_TIME:
            break


if __name__ == '__main__':
    # the random seed
    # np.random.seed(1000)
    # These are the initial values
    x = 1
    y = 0
    th = 0
    z = 0
    t = 0
    main(x, y, th, z, t)
