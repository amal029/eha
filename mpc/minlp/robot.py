#!/usr/bin/env python3
import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt


def set_plt_params():
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12


def example():
    # XXX: Number of time steps
    nt = 5
    tm = np.linspace(0, 1, nt)

    # XXX: Initialize GEKKO
    m = GEKKO(remote=False)
    # XXX: Set the time for the model
    m.time = tm

    # XXX: Now the variables
    m.x = m.Var(value=0, lb=0, ub=2.8, fixed_initial=True, name='x')
    m.y = m.Var(value=1, lb=-0.8, ub=1.8, fixed_initial=True, name='y')
    m.th = m.Var(value=0, lb=-2, ub=2, fixed_initial=True, name='th')
    m.ph = m.Var(value=0, lb=-2, ub=2, fixed_initial=True, name='ph')

    # XXX: The continuous control variable
    m.u1 = m.Var(lb=3, ub=4, fixed_initial=False, name='u1')
    m.u2 = m.Var(lb=1, ub=1, fixed_initial=False, name='u2')

    # XXX: Dynamics of the system
    ll = 1
    g = m.Var(lb=-1, ub=1, integer=True, fixed_initial=False, name='g')
    m.Equation(g == -(m.sign3(m.x-2.8))*(m.sign3(m.y-1.8) - m.sign3(m.y+0.8)))
    m.Equation(m.x.dt() == m.if3(g-1, m.cos(m.th)*m.u1, 0))
    # m.Equation(m.y.dt() == m.if3(g-1, m.sin(m.th)*(-m.u1), 0))
    # m.Equation(m.th.dt() == m.if3(g-1, m.tan(m.ph)/ll*m.u1, 0))
    # m.Equation(m.ph.dt() == m.if3(g-1, -m.u2, 0))

    # XXX: The objective
    m.Obj(
        m.integral((m.x-0.7)**2)
        + m.integral((m.y+0.7)**2)
        # + 0*m.integral((m.th - 2.0)**2)
        # + 0*m.integral((m.ph - 0.47)**2)
    )

    m.options.SOLVER = 1        # APOPT solver
    m.solver_options = ['minlp_maximum_iterations 1000000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 10', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 50', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.0005', \
                        # covergence tolerance
                        'minlp_gap_tol 0.01']
    m.options.IMODE = 5         # simultaneous dynamic collocation
    m.solve(debug=1)

    print(g.value)
    print(m.time, m.x.value, m.y.value, m.th.value,
          m.ph.value, m.u1.value, m.u2.value)

    return (m.time, m.x.value, m.y.value, m.th.value,
            m.ph.value, m.u1.value, m.u2.value)


if __name__ == '__main__':
    set_plt_params()
    ts, xxs, yys, _, _, uref1, uref2 = example()
    plt.style.use('ggplot')
    plt.plot(xxs, yys)
    plt.show()
