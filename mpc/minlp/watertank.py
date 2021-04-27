#!/usr/bin/env python3
import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt


def example():
    # XXX: Number of time steps
    nt = 50
    tm = np.linspace(0, 1, nt)

    # XXX: Initialize GEKKO
    m = GEKKO(remote=False)
    # XXX: Set the time for the model
    m.time = tm

    # XXX: Now declare the state variables
    m.x1 = m.Var(value=0.5, fixed_initial=True, lb=0, ub=4, name='x1')
    m.x2 = m.Var(value=3, fixed_initial=True, lb=0, ub=4, name='x2')

    # XXX: The continuous control variable
    m.u = m.Var(lb=1, ub=16, fixed_initial=False, name='u')
    # XXX: The discrete control variable
    m.g = m.Var(lb=0, ub=1, integer=True, fixed_initial=False, name='g')

    # XXX: Now make the dynamic (Equations)
    m.Equation(m.x1.dt() == (1-m.g)*m.u + m.g*(-(m.x1**2)))
    m.Equation(m.x2.dt() == (1-m.g)*(-(m.x2**2)) + m.g*m.u)

    print(m._path)

    # XXX: The initial boundary condition
    m.fix_initial(m.x1, 0.5)
    m.fix_initial(m.x2, 3)

    # XXX: The final boundary condition
    # m.fix_final(m.x1, 2)
    # m.fix_final(m.x2, 2)

    # FIXME: Add the objective function here later
    m.Obj(
        # m.integral(m.u**2) +
        # m.integral(m.g**2) +
        m.integral((m.x1-2)**2) +
        m.integral((m.x2-2)**2)
    )

    # XXX: Set the solver option to dynamic
    # m.options.COLDSTART = 2
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
    # m.options.DIAGLEVEL = 2
    m.solve(debug=1)

    # print('x1: %s, x2: %s, g: %s, u: %s' % (m.x1.value, m.x2.value,
    #                                         m.g.value, m.u.value))

    return m.time, m.x1.value, m.x2.value, m.u.value, m.g.value


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


if __name__ == '__main__':
    set_plt_params()
    ts, tr1s, tr2s, uref, gref = example()
    plt.style.use('ggplot')

    # x1
    plt.plot(ts, tr1s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankx1minlp.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # x2
    plt.plot(ts, tr2s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x2(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankx2minlp.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # XXX: DEBUG
    plt.plot(ts, tr1s, label='x1')
    plt.plot(ts, tr2s, label='x2')
    plt.legend(loc='best')
    plt.show()

    # u
    plt.plot(ts, uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankuminlp.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()

    # g
    plt.plot(ts, gref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$g(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/watertankgminlp.pdf', bbox_inches='tight')
    # plt.show()
    plt.close()
