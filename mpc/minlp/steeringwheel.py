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
    tm = np.linspace(0, 1, nt)

    # XXX: Initialize GEKKO
    m = GEKKO(remote=False)
    # XXX: Set the time for the model
    m.time = tm

    # XXX: The variables
    m.x = m.Var(value=np.pi, fixed_initial=True, lb=0, ub=2*np.pi)

    # XXX: The continuous control variable
    m.u = m.Var(lb=-2, ub=2, fixed_initial=False)

    m.fix_initial(m.x, np.pi)

    m.fix_final(m.x, np.pi/2)

    # XXX: The dynamics (simple sliding mode control)
    m.Equation(m.x.dt() == m.sign3(m.x-np.pi/2)*m.sign3(m.x-3*np.pi/2)*m.u)

    # XXX: The objective
    m.Obj(
        400*m.integral((m.x - np.pi/2)**2)
        + m.integral(m.u**2)
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

    return m.time, m.x.value, m.u.value


if __name__ == '__main__':
    set_plt_params()
    example()
    ts, tr1s, uref = example()
    plt.style.use('ggplot')

    plt.plot(ts, tr1s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/steeringxminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(ts, uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/steeringurefminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
