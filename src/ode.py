import sympy as S


class ODE:
    """This class represents the ODEs and associated computation of time.

    """
    def __init__(self, env, lvalue, rvalue, qorder=1, torder=1,
                 iterations=20, vtol=10**-2, ttol=10**-2):
        """The quantized state order and taylor series order by default is 1.
        The maximum number of back-stepping iterations is 20 be default.
        The error is default 10^-4.

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
        q = self.get_q(init)
        # XXX:
        slope = ODE.replace(self.rvalue, self.lvalue.args[0], q)
        slope = slope.subs('t', time)
        return init + float(slope)*time

    def _delta2(self, init):
        slope = ODE.replace(self.rvalue, self.lvalue.args[0], init)
        t = S.Symbol('t')
        return (S.Add(init, (S.Mul(slope, (t - self.env.now)))))

    def _taylor1(self, init, q, q2, quanta, count):
        def get_d(q):
            # XXX: Note that partial derivatives are not supported. E.g.,
            # x'(t) = x(t) + y(t) is not supported for now.
            d = S.Symbol('d', positive=True, real=True)
            slope = ODE.replace(self.rvalue, self.lvalue.args[0], q)
            # XXX: Assumption that the time line is called "t"
            slope = self.rvalue.replace(self.lvalue.args[0], q).subs('t', d)
            polynomial = S.Eq(S.Mul(d, slope), quanta)
            dl = S.solve(polynomial, d)
            if dl == []:
                raise RuntimeError('No real solution for: ', polynomial)
            d = min(dl)
            return d

        d1 = get_d(q)
        d2 = get_d(q2)
        # print('trying quanta: ', quanta)
        if abs(d1 - d2) <= self.ttol:
            # print(quanta, d1, d2)
            return d1
        elif count < self.iterations:
            # If the delta step results in output that is within the
            # user defined error bounds then great. Else, half the
            # quanta and keep on doing this until number of iterations
            # is met. This is reducing the quanta in a geometric
            # progression. This is the same as RK-2(3) solver
            newquanta = 0.8 * pow(self.ttol / abs(d1 - d2), 1.0/2)
            quanta = newquanta if newquanta <= quanta else 0.5*quanta
            return self._taylor1(init, q, q2, quanta, (count+1))
        else:
            raise RuntimeError('Could not find delta that satisfies '
                               'the user specified error bound: %s %s %s'
                               % (self.ttol, d1, d2))

    def _taylor(self, init, q, q2, quanta):
        if self.torder == 1:
            # The delta step
            return self._taylor1(init, q, q2, quanta, 0)
        elif self.torder > 1:
            raise RuntimeError('Currently only first order taylor supported')

    def get_q(self, init):
        # First calculate the q(t) given the qorder
        if self.qorder == 1:
            q = init
        elif self.qorder == 2:
            q = self._delta2(init)
        elif self.qorder > 2:
            raise RuntimeError('Curretly only upto QSS2 is supported')
        return q

    def delta(self, init, quanta=0.1):
        """This is the main function that returns back the delta step-size.
        Arguments: The initial value of the ode. Returns: The delta
        step-size that is within the user specified error.

        """
        q = self.get_q(init)
        q2 = self._delta2(init)
        # Now do the taylor series
        delta = self._taylor(init, q, q2, quanta)
        # print('Computed delta: ', delta)
        return delta

    def __str__(self):
        ode = str(self.lvalue) + ' = ' + str(self.rvalue)
        ret = ' '.join([ode])
        return ret+'\n'
