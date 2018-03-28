import sympy as S


class ODE:
    """This class represents the ODEs and associated computation of time.

    """
    def __init__(self, env, lvalue, rvalue, qorder=1, torder=1,
                 iterations=20, error=10**-4):
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
        self.error = error

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

    # TODO: LATER
    def _delta2(self, init):
        pass

    def _taylor1(self, init, q, quanta, count):
        def _ode_eval(d, slope):
            delta = float(d)
            expected = init + quanta
            real = (slope.subs('d', d)*delta + init)
            return abs(real-expected)

        # XXX: Note that partial derivatives are not supported. E.g.,
        # x'(t) = x(t) + y(t) is not supported for now.
        d = S.Symbol('d', postive=True)
        slope = ODE.replace(self.rvalue, self.lvalue.args[0], q)
        # raise RuntimeError()
        # XXX: Assumption that the time line is called "t"
        slope = self.rvalue.subs(self.lvalue.args[0], q).subs('t', d)
        polynomial = S.Eq(S.Mul(d, slope), quanta)
        dl = S.solve(polynomial, d)
        d = min(dl)
        if _ode_eval(d, slope) <= self.error:
            return d
        elif count < self.iterations:
            # If the delta step results in output that is within the
            # user defined error bounds then great. Else, half the
            # quanta and keep on doing this until number of iterations
            # is met. This is reducing the quanta in a geometric
            # progression.
            self._taylor1(q, (quanta/2), (count+1))
        else:
            raise RuntimeError('Could not find delta that satisfies '
                               'the user specified error bound: %s'
                               % self.error)

    def _taylor(self, init, q, quanta):
        if self.torder == 1:
            # The delta step
            return self._taylor1(init, q, quanta, 0)
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
        # Now do the taylor series
        return self._taylor(init, q, quanta)

    def __str__(self):
        ode = str(self.lvalue) + ' = ' + str(self.rvalue)
        ret = ' '.join([ode])
        return ret+'\n'
