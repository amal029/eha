#!/usr/bin/env python3
import sympy as S
import numpy as np
import mpmath as M


class Compute:
    """This is the main adaptive time step computation class

    """
    # Static error bound
    epsilon = 1e-3

    @staticmethod
    def var_compute(left=None, right=None, deps=None, dWts=None, vars=None,
                    T=0):
        pass

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
        z2dt = right.diff(t)  # This is the second derivative
        # Now substitute the resultants in second derivatives for all vars
        # First build the derivative keys
        toreplace = {i.diff(t): j for (i, j) in deps.items()}
        # The stochastic keys
        dWts = {i.diff(t): dWts[i] for i in dWts}
        # print(zdt, z2dt, toreplace, dWts, R)
        # Now substitute the derivatives in z2dt
        for i in toreplace:
            z2dt = z2dt.subs(i,
                             (toreplace[i][0] +
                              toreplace[i][1]*sum(dWts[i])*S.sqrt(Dt/R)/Dt))
        # Now replace vars with current values
        for i, j in vars.items():
            zdt = zdt.replace(i, j)
            z2dt = z2dt.replace(i, j)
        # Finally replace t if left over with current value T
        zdt = zdt.replace(t, T)*Dt
        z2dt = z2dt.replace(t, T)*Dt**2/2

        # Now doing the two sided root finding
        L = (Uz - vars[left])   # This is the level crossing
        f = zdt + z2dt          # This is the part without the Level

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

            # Taking one big step Dtv
            temp1 = {i: Compute.EM(vars[i], deps[i][0], deps[i][1],
                                   Dtv, dtv, dWts[i.diff(t)], vars, T, i)
                     for i in vars}

            # Taking step to Dtv/2
            nvars = {i: Compute.EM(vars[i], deps[i][0], deps[i][1],
                                   Dtv/2, dtv, dWts[i.diff(t)][0:R//2], vars,
                                   T, i)
                     for i in vars}
            # Taking step from Dtv/2 --> Dtv
            nvars = {i: Compute.EM(nvars[i], deps[i][0], deps[i][1],
                                   Dtv/2, dtv, dWts[i.diff(t)][R//2:R], nvars,
                                   T+(Dtv/2), i)
                     for i in vars}
            z1 = temp1[left]
            z2 = nvars[left]
            err = (np.sum(np.abs((z1 - z2)/(z2 + Compute.epsilon)))
                   <= Compute.epsilon)   # XXX: This gives the best results.
            if err:
                print('Found rate step:', z1, z2, Dtv)
                return Dtv, dtv
            else:
                L = L/2

    @staticmethod
    def EM(init, f, g, Dt, dt, dWts, vars, T, v):
        for i in vars:
            f = f.subs(i, vars[i])
            g = g.subs(i, vars[i])
        res = init + f*Dt + g*np.sqrt(dt)*np.sum(dWts)
        return res.subs(S.var('t'), T).evalf()

    @staticmethod
    def guard_compute(expr=None, deps=None, vars=None, T=0):
        pass


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
        # Compute the dynamics in state Move
        DM = dynamics[location]
        # XXX: Should we give the random path from here?
        # Create dWt
        dWts = {S.sympify('x(t)'): np.random.randn(R),
                S.sympify('y(t)'): np.random.randn(R),
                S.sympify('th(t)'): np.random.randn(R),
                S.sympify('z(t)'): np.zeros(R)}
        dx = Compute.var_compute(left=S.sympify('x(t)'),
                                 right=DM[S.sympify('x(t)')], deps=DM,
                                 dWts=dWts, vars=vars, T=t)
        dy = Compute.var_compute(left=S.sympify('y(t)'),
                                 right=DM[S.sympify('y(t)')], deps=DM,
                                 dWts=dWts, vars=vars, T=t)
        dth = Compute.var_compute(left=S.sympify('th(t)'),
                                  right=DM[S.sympify('th(t)')],
                                  deps=DM, dWts=dWts, vars=vars,
                                  T=t)

        # XXX: This is for computing the spontaneous jump if any
        dz = Compute.rate_compute(left=S.sympify('z(t)'),
                                  right=DM[S.sympify('z(t)')], deps=DM,
                                  vars=vars, T=t, Uz=Uz,
                                  dWts=dWts)

        # XXX: Accounting for the guards
        dgs = [Compute.guard_compute(expr=i, deps=DM, vars=vars, T=t)
               for i in guards]

        # XXX: dz might be np.inf if there is no spontaneous output
        # This is the step size we will take
        # XXX: T == Δ == δ*R (assumption)
        T = min(dx, dy, dz, dth, *dgs)

        # Now compute the actual values of x, y, th, z using Euler
        # maruyama for step size T First substitute the value of
        # vars with the current values in the argument and then
        # eval.
        x += (Compute.subs(DM[S.sympify('x(t)')][0], vars) * T
              + (np.sqrt(T/R) * Compute.subs([S.sympify('x(t)')][1])
                 * np.sum(dWts[S.sympify('x(t)')])))

        y += (Compute.subs(DM[S.sympify('y(t)')][0], vars) * T
              + (np.sqrt(T/R) * Compute.subs([S.sympify('y(t)')][1])
                 * np.sum(dWts[S.sympify('y(t)')])))

        th += (Compute.subs(DM[S.sympify('th(t)')][0], vars) * T
               + (np.sqrt(T/R) * Compute.subs([S.sympify('th(t)')][1])
                  * np.sum(dWts[S.sympify('th(t)')])))

        z += (Compute.subs(DM[S.sympify('z(t)')][0], vars) * T)
        return T

    def Move(first_time):
        """Location Move
        """
        global th, z, x, y, t
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e):
            # Set the outputs
            th, z = 0, 0
            state = 1           # destination location is Inner
            return state, 0, True
        elif (x**2 + y**2 - v**2 >= e):
            th, z = S.pi, 0
            state = 2           # destination is Outter
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - (v**2 - e)
            g2 = S.sympify('x(t)**2 + y(t)**2') - (v**2 + e)
            T = __compute('Move', [g1, g2], None)
            return 0, T, False

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
            return state, 0, True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            g2 = S.sympify('z(t)') - Uz
            T = __compute('Inner', [g1, g2], Uz)
            return 1, T, False

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
            return state, 0, True
        elif abs(z - Uz) <= e:
            z = 0
            state = 3           # destination is Changetheta
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('x(t)**2 + y(t)**2') - v**2
            g2 = S.sympify('z(t)') - Uz
            T = __compute('Outter', [g1, g2], Uz)
            return 2, T, False

    def Changetheta(first_time):
        global th, z, x, y, t
        if first_time:
            Uz = -np.log(np.random.rand())
        # First compute the outgoing edges and take them
        if (x**2 + y**2 - v**2 <= -e) and abs(z - Uz) <= e:
            # Set the outputs
            z = 0
            state = 1           # destination location is Inner
            return state, 0, True
        elif (x**2 + y**2 - v**2 >= e) and abs(z - Uz) <= e:
            z = 0
            state = 2           # destination is Outter
            return state, 0, True
        else:
            # XXX: Accounting for the guards
            g1 = S.sympify('z(t)') - Uz
            T = __compute('Changetheta', [g1], Uz)
            return 3, T, False

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
        state, T, first_time = locations[state](first_time)

        t += T
        # Print the outputs
        print('%.4f: state:%s, x:%s, y:%s, th:%s, z:%s' % (t, strloc[state],
                                                           x, y, th, z))

        if t >= SIM_TIME:
            break


if __name__ == '__main__':
    # These are the initial values
    x = 1
    y = 0
    th = 0
    z = 0
    t = 0
    main(x, y, th, z, t)
