import sympy as S
import mpmath as poly
import numpy as N


class NoRealRoots(Exception):
    pass


class ODE:
    """This class represents the ODEs and associated computation of time.

    """

    MAX_QUANTA = 10**-3
    NUM_TERMS = 3               # This should be adjustable

    TRIG_FUNCS = [S.sin, S.cos, S.tan, S.cot, S.sec, S.csc]
    INV_TRIG_FUNCS = [S.asin, S.acos, S.atan, S.acot, S.asec, S.acsc, S.atan2]
    HYPERBOLIC_FUNCS = [S.sinh, S.cosh, S.tanh, S.coth, S.sech, S.csch]
    INV_HYPERBOLIC_FUNCS = [S.asinh, S.acosh, S.atanh, S.acoth, S.asech,
                            S.acsch]
    EXP_LOG = [S.exp, S.ln]
    TRANSCEDENTAL_FUNCS = (TRIG_FUNCS + INV_TRIG_FUNCS + HYPERBOLIC_FUNCS +
                           INV_HYPERBOLIC_FUNCS + EXP_LOG)

    def __init__(self, env, lvalue, rvalue, qorder=1, torder=1,
                 iterations=20, vtol=0, ttol=10**-2, taylor_expand=5,
                 trans_funcs=[], simplify_poly=False):
        """The quantized state order and taylor series order by default is 1.
        The maximum number of back-stepping iterations is 20 be default.
        The tolerance by default is 10^-2. taylor_expand gives the
        number to terms that we expand transcendental function too,
        default 5. Simplify the polynomial before finding roots, can
        take a very long time. Usually simplification is needed if
        polynomial has both a numerator and a denominator.

        """
        self.env = env
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.qorder = qorder
        self.torder = torder
        self.iterations = iterations
        self.vtol = vtol
        self.ttol = ttol
        ODE.NUM_TERMS = taylor_expand
        ODE.TRANSCEDENTAL_FUNCS += trans_funcs
        ODE.simplify_poly = simplify_poly

    @staticmethod
    # XXX:
    def replace(expr, s, v):
        if expr == s:
            return (S.sympify(v))
        elif expr.args == ():
            return expr
        else:
            return expr.func(*[ODE.replace(a, s, v)
                               for a in expr.args])

    # XXX: Post order traversal
    @staticmethod
    def taylor_expand(expr, around=0):
        if expr.args is ():
            return expr
        args = [ODE.taylor_expand(a) for a in expr.args]
        if expr.func in ODE.TRANSCEDENTAL_FUNCS:
            if len(args) != 1:
                raise RuntimeError('Cannot create a taylor series '
                                   'approximation of: ', expr)
            else:
                # XXX: Build the polynomial for arg
                coeffs = poly.taylor(expr.func, around, ODE.NUM_TERMS)
                # print(coeffs)
                coeffs = [(S.Mul(float(a), S.Mul(*[args[0]
                                                   for i in range(c)])))
                          for c, a in enumerate(coeffs)][::-1]
                # print(coeffs)
                return S.Add(*coeffs)
        else:
            return expr.func(*args)

    def compute(self, init, time):
        # Now init is a dictionary of all the required initial values.
        slope = self.rvalue
        for k in init:
            slope = ODE.replace(slope, k, init[k])
        slope = slope.subs('t', time)
        return init[self.lvalue.args[0]] + float(slope)*time

    def _delta1(self, init):
        return init[self.lvalue.args[0]]

    def _delta2(self, init):
        # slope = ODE.replace(self.rvalue, self.lvalue.args[0], init)
        slope = self.rvalue
        for k in init:
            slope = ODE.replace(slope, k, init[k])
        t = S.Symbol('t')
        return (S.Add(init[self.lvalue.args[0]],
                      (S.Mul(slope, (t - self.env.now)))))

    def _taylor1(self, init, q, q2, quanta, count):
        def is_den(x):
            return (type(x) == S.Pow and x.args[1] == -1)

        def compute_delta(part_poly, d, dl, quanta):
            # XXX: Positive quantum, so polynomial - quanta = 0
            polynomial1 = S.Add(part_poly, -quanta)
            # XXX: Assumption that the time line is called "t"
            # print(polynomial1)
            if not ODE.simplify_poly:
                polynomial1 = (polynomial1.expand().subs('t', d))
            else:
                polynomial1 = S.simplify(polynomial1.expand().subs('t', d))
            ppoly = polynomial1
            # XXX: Taking care of numerator and denominators after
            # expansion.
            if type(polynomial1) == S.Mul:
                if not is_den(polynomial1.args[0]):
                    polynomial1 = polynomial1.args[0]  # Get just the numerator
                else:
                    polynomial1 = polynomial1.args[1]
            # print('polynomial:', polynomial1)
            # If "δ" vanishes after expansion then just return None
            if (type(polynomial1) is S.Float):
                return None
            polynomial1 = S.Poly(polynomial1)
            try:
                nsoln = N.roots(polynomial1.all_coeffs())
                nsoln = nsoln[N.isreal(nsoln)]
                nsoln = nsoln[N.where(nsoln >= 0)]
                # soln = poly.polyroots([poly.mpf(a) for
                #                        a in polynomial1.all_coeffs()])
                # print('1:', nsoln, soln)
            except S.PolynomialError as e:
                print('When trying to solve: ', ppoly)
                raise e
            # dl += [float(a) for a in soln
            #        if type(a) is poly.mpf and float(a) >= 0]
            dl += list(nsoln)
            # The second polynomial
            # XXX: Negative quantum, so polynomial + quanta = 0
            polynomial2 = S.Add(part_poly, quanta)
            # XXX: Assumption that the time line is called "t"
            if not ODE.simplify_poly:
                polynomial2 = (polynomial2.expand().subs('t', d))
            else:
                polynomial2 = S.simplify(polynomial2.expand().subs('t', d))
            ppoly = polynomial2
            # print(ppoly.args[0], ppoly.args[1])
            if type(polynomial2) == S.Mul:
                if not is_den(polynomial2.args[0]):
                    polynomial2 = polynomial2.args[0]  # Get just the numerator
                else:
                    polynomial2 = polynomial2.args[1]
            polynomial2 = S.poly(polynomial2)
            try:
                nsoln = N.roots(polynomial2.all_coeffs())
                nsoln = nsoln[N.isreal(nsoln)]
                nsoln = nsoln[N.where(nsoln >= 0)]
                # soln = poly.polyroots([poly.mpf(a) for
                #                        a in polynomial2.all_coeffs()])
                # print('2:', nsoln, soln)
            except S.PolynomialError as e:
                print('When trying to solve: ', ppoly)
                raise e
            # dl += [float(a) for a in soln
            #        if type(a) is poly.mpf and float(a) >= 0]
            dl += list(nsoln)
            return dl

        def get_d(q):
            d = S.Symbol('d', positive=True, real=True)
            # XXX: My rvalue can depend upon a whole vector os q's
            # TODO: Convert it into a taylor series
            # print(self.rvalue, q)

            # XXX: Making a taylor polynomial if it is transcendental
            # function
            slope = ODE.taylor_expand(self.rvalue)
            # print('slope: ', slope)
            # print(q)
            for k in q:
                slope = ODE.replace(slope, k, q[k]).evalf()
            # print(slope)
            # XXX: IMP CHANGE! Here I am chaning QSS to compare with a
            # constant level not qith "Q". Note that q is the slope
            # itself.
            part_poly = S.Mul(d, slope)
            # print('ppoly: ', part_poly.subs('t', 'd').expand().evalf())
            # XXX: compute_delta saolves for the roots of the polynomial
            dl = compute_delta(part_poly, d, [], quanta)
            if dl is None:
                return None     # The constant slope case
            elif dl == []:
                raise NoRealRoots('No real positive root for: ',
                                  S.Eq(part_poly.subs('t', d).expand(),
                                       quanta), '{:.2e}'.format(quanta))
            d = min(dl)
            return d

        # print('getting δ1')
        d1 = get_d(q)           # Get the future time event from QSS-1
        # print('getting δ2')
        d2 = get_d(q2)          # Get the future time event from modified QSS-2
        if d1 is None:
            return S.oo, quanta  # This is returning infinity, wrong HA
        if d2 is None:
            # d1s = '{:.2e}'.format(d1)
            # quanta = '{:.2e}'.format(quanta)
            # print('chosen Δq: %s δ: %s' % (quanta, d1s))
            return d1, quanta
        elif abs(d1 - d2) <= self.ttol:
            # d1s = '{:.2e}'.format(d1)
            # d2s = '{:.2e}'.format(d2)
            # pquanta = '{:.2e}'.format(quanta)
            # print('chosen Δq: %s δ1: %s δ2: %s' % (pquanta, d1s, d2s))
            # In this case we have satisfied εt so returning first δ
            return d1, quanta
        elif count < self.iterations:
            # If the delta step results in output that is within the
            # user defined error bounds then great. Else, half the
            # quanta and keep on doing this until number of iterations
            # is met. This is reducing the quanta in a geometric
            # progression.

            # XXX: Adaptive Stepsize Runge-Kutta Integration William H.
            # Press, and Saul A. Teukolsky
            newquanta = d1 * pow(abs(self.ttol / (d1 - d2)), 1.0/2)
            quanta = newquanta if newquanta <= quanta else 0.5*quanta
            return self._taylor1(init, q, q2, quanta, (count+1))
        else:
            raise RuntimeError('Could not find delta that satisfies '
                               'the user specified error bound: '
                               'ε: %s δ1: %s δ2: %s Q1: %s Q2: %s '
                               'Δq: %s. Increase interation count'
                               % (self.ttol, d1, d2, q, q2, quanta))

    def _taylor(self, init, q, q2, quanta):
        if self.torder == 1:    # First order taylor only supported
            # The delta step
            return self._taylor1(init, q, q2, quanta, 0)
        elif self.torder > 1:
            raise RuntimeError('Currently only first order taylor supported')

    def get_q(self, init, order):
        # First calculate the q(t) given the qorder
        if order == 1:
            q = self._delta1(init)
        elif order == 2:
            q = self._delta2(init)
        elif order > 2:
            raise RuntimeError('Curretly only upto QSS2 is supported')
        return q

    # XXX: This is the main function, which returns the future time
    # event per level crossing per variable.
    def delta(self, init, other_odes=None, quanta=MAX_QUANTA):
        """This is the main function that returns back the delta step-size.
        Arguments: The initial value of the ode. Returns: The delta
        step-size that is within the user specified error.

        """
        # These two are me XXX: Here we are building the quantized
        # states, i.e., hysterisis for qorder=1 and integration for
        # qorder-2.
        qs = {self.lvalue.args[0]: self.get_q(init, self.qorder)}
        q2s = {self.lvalue.args[0]: self.get_q(init, self.qorder+1)}
        # XXX: Building the quantized states for other odes that we
        # might depend upon, because we can have coupled ODEs.
        if other_odes is not None:
            for ode in other_odes:
                qs[ode.lvalue.args[0]] = ode.get_q(init, ode.qorder)
                q2s[ode.lvalue.args[0]] = ode.get_q(init, ode.qorder+1)
        # XXX: delta is the returned value
        delta, nquanta = self._taylor(init, qs, q2s, quanta)
        # XXX: Handling sudden jumps
        if ((not (type(self.rvalue) is S.Float
                  or type(self.rvalue) is S.Integer) and
             float(nquanta) == quanta and abs(quanta) > ODE.MAX_QUANTA)):
            # print('halved the quanta')
            delta, _ = self._taylor(init, qs, q2s, quanta*0.5)
        return delta

    def __str__(self):
        ode = str(self.lvalue) + ' = ' + str(self.rvalue)
        ret = ' '.join([ode])
        return ret+'\n'
