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
    nt = 50
    tm = np.linspace(0, 6, nt)

    # XXX: Initialize GEKKO
    m = GEKKO(remote=False)
    # XXX: Set the time for the model
    m.time = tm

    # XXX: The variables
    m.x1 = m.Var(value=-5, fixed_initial=True, lb=-5.5, ub=1)
    m.x2 = m.Var(value=0, fixed_initial=True, lb=-0.5, ub=3)

    # XXX: The final boundary condition
    # m.fix_final(m.x1, 0)
    # m.fix_final(m.x2, 0)

    # XXX: The continuous control variable
    m.u = m.Var(lb=-1, ub=1, fixed_initial=False)

    # XXX: The discrete control variable
    m.g = m.Var(lb=0, ub=1, integer=True, fixed_initial=False)

    # XXX: The dynamics
    m.Equation(m.x1.dt() == m.x2)
    m.Equation(m.x2.dt() == (1-m.g)*(1/(1+m.exp(-5*(m.x2-0.5))))*m.u +
               (m.g)*(1/(1+m.exp(5*(m.x2-0.5))))*m.u)

    # XXX: The objective
    # FIXME: This should be minimize time
    m.Obj(
        4*m.integral(m.x1**2) +
        4*m.integral(m.x2**2)
        + 0.05*m.integral((m.g-1)**2)
    )

    m.options.SOLVER = 1        # APOPT solver
    m.solver_options = ['minlp_maximum_iterations 10000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 10', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 500', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.0005', \
                        # covergence tolerance
                        'minlp_gap_tol 0.01']
    m.options.IMODE = 5         # simultaneous dynamic collocation
    m.solve(debug=1)

    return m.time, m.x1.value, m.x2.value, m.u.value, m.g.value


if __name__ == '__main__':
    set_plt_params()
    example()
    ts, tr1s, tr2s, uref, gref = example()

    plt.style.use('ggplot')
    plt.plot(ts, tr1s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsx1minlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(ts, tr2s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x2(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsx2minlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(tr1s, tr2s)
    plt.xlabel('x1(t)')
    plt.ylabel('x2(t)')
    # plt.legend(loc='best')
    plt.show()
    plt.close()

    plt.plot(ts, uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsurefminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(ts, gref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$g(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/gearsgrefminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
