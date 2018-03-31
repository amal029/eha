import sympy as S
# import numpy.polynomial.polynomial as poly
import mpmath as poly


class NoRealRoots(Exception):
    pass


class ODE:
    """This class represents the ODEs and associated computation of time.

    """
    def __init__(self, env, lvalue, rvalue, qorder=1, torder=1,
                 iterations=20, vtol=10**-4, ttol=10**-2):
        """The quantized state order and taylor series order by default is 1.
        The maximum number of back-stepping iterations is 20 be default.
        The tolerance by default is 10^-2.

        """
        self.env = env
        self.lvalue = lvalue
        self.rvalue = rvalue
        self.qorder = qorder
        self.torder = torder
        self.iterations = iterations
        self.vtol = vtol
        self.ttol = ttol

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

    def compute(self, init, time):
        q = self.get_q(init, self.qorder)
        # XXX:
        slope = ODE.replace(self.rvalue, self.lvalue.args[0], q)
        slope = slope.subs('t', time)
        return init + float(slope)*time

    def _delta2(self, init):
        slope = ODE.replace(self.rvalue, self.lvalue.args[0], init)
        t = S.Symbol('t')
        return (S.Add(init, (S.Mul(slope, (t - self.env.now)))))

    def _taylor1(self, init, q, q2, quanta, count):
        def compute_delta(part_poly, d, dl, quanta, ignore_imag=False):
            polynomial1 = S.Add(part_poly, -quanta)
            # XXX: Assumption that the time line is called "t"
            polynomial1 = polynomial1.expand().subs('t', d)
            # If "δ" vanishes after exapansion then just return None
            if (type(polynomial1) is S.Float):
                return None
            polynomial1 = S.Poly(polynomial1)
            # print(polynomial1)
            soln = poly.polyroots([poly.mpf(a) for
                                   a in polynomial1.all_coeffs()])
            # XXX: This is carried out when this gets called from
            # handle_imaginary_roots, because of numerical root finding,
            # the imaginary component can still exist, but be very
            # small.
            if ignore_imag:
                soln = [poly.re(a) for a in soln]
            dl += [float(a) for a in soln
                   if type(a) is poly.mpf and float(a) >= 0]
            # The second polynomial
            polynomial2 = S.Add(part_poly, quanta)
            # XXX: Assumption that the time line is called "t"
            polynomial2 = S.Poly(polynomial2.subs('t', d))
            # print(polynomial2)
            soln = poly.polyroots([poly.mpf(a) for
                                   a in polynomial2.all_coeffs()])
            # XXX: This is carried out when this gets called from
            # handle_imaginary_roots, because of numerical root finding,
            # the imaginary component can still exist, but be very
            # small.
            if ignore_imag:
                soln = [poly.re(a) for a in soln]
            dl += [float(a) for a in soln
                   if type(a) is poly.mpf and float(a) >= 0]
            return dl

        def get_d(q):
            # XXX: Note that partial derivatives are not supported. E.g.,
            # x'(t) = x(t) + y(t) is not supported for now.
            d = S.Symbol('d', positive=True, real=True)
            slope = ODE.replace(self.rvalue, self.lvalue.args[0], q)
            # XXX: IMP CHANGE!
            # XXX: Here I am chaning QSS to compare with a constant
            # level not qith "Q"
            part_poly = S.Add(S.Add(S.Mul(d, slope), init), (-init))
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
        d1 = get_d(q)
        # print('getting δ2')
        d2 = get_d(q2)
        if d2 is None:
            # d1s = '{:.2e}'.format(d1)
            # quanta = '{:.2e}'.format(quanta)
            # print('chosen Δq: %s δ: %s' % (quanta, d1s))
            return d1
        elif abs(d1 - d2) <= self.ttol:
            d1s = '{:.2e}'.format(d1)
            d2s = '{:.2e}'.format(d2)
            quanta = '{:.2e}'.format(quanta)
            print('chosen Δq: %s δ1: %s δ2: %s' % (quanta, d1s, d2s))
            return d1
        elif count < self.iterations:
            # If the delta step results in output that is within the
            # user defined error bounds then great. Else, half the
            # quanta and keep on doing this until number of iterations
            # is met. This is reducing the quanta in a geometric
            # progression. This is the same as RK-2(3) solver
            # print('old quanta: ', quanta)
            newquanta = 0.8 * pow(self.ttol / abs(d1 - d2), 1.0/2)
            quanta = newquanta if newquanta <= quanta else 0.5*quanta
            # print('new quanta: ', quanta)
            return self._taylor1(init, q, q2, quanta, (count+1))
        else:
            raise RuntimeError('Could not find delta that satisfies '
                               'the user specified error bound: '
                               'ε: %s δ1: %s δ2: %s Q1: %s Q2: %s '
                               'Δq: %s'
                               % (self.ttol, d1, d2, q, q2, quanta))

    def _taylor(self, init, q, q2, quanta):
        if self.torder == 1:
            # The delta step
            return self._taylor1(init, q, q2, quanta, 0)
        elif self.torder > 1:
            raise RuntimeError('Currently only first order taylor supported')

    def get_q(self, init, order):
        # First calculate the q(t) given the qorder
        if order == 1:
            q = init
        elif order == 2:
            q = self._delta2(init)
        elif order > 2:
            raise RuntimeError('Curretly only upto QSS2 is supported')
        return q

    def delta(self, init, quanta=0.1):
        """This is the main function that returns back the delta step-size.
        Arguments: The initial value of the ode. Returns: The delta
        step-size that is within the user specified error.

        """
        q = self.get_q(init, self.qorder)
        q2 = self.get_q(init, self.qorder+1)
        # XXX: This only works for LTI systems! What this says is that
        # linear and quadratic equations will give the same solution for
        # a very large δq value!. XXX: The below is done, because one
        # will see these sudden jumps when the cont value appraoches the
        # level.
        slope = ODE.replace(self.rvalue, self.lvalue.args[0], q)
        if not (abs((quanta - (slope*self.env.now))) > 0.05) and quanta > 0.1:
            print('sub: ', abs((quanta - (slope*self.env.now))))
            quanta = 0.1
        delta = self._taylor(init, q, q2, quanta)
        # print('Computed delta: ', delta)
        return delta

    def __str__(self):
        ode = str(self.lvalue) + ' = ' + str(self.rvalue)
        ret = ' '.join([ode])
        return ret+'\n'
