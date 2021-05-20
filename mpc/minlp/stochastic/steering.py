#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
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


def example(R, delta):
    # XXX: Initialize GEKKO
    m = GEKKO(remote=False)
    time = [i*delta for i in range(R)]

    # XXX: The variables, one for each time step
    m.x = [m.Var(lb=0, ub=2*np.pi, name=('x_%s' % i))
           for i in range(R)]

    # XXX: The continuous control variable
    m.u = [m.Var(lb=-0.5, ub=0.5, fixed_initial=False, name='u_%s' % i)
           for i in range(R-1)]

    # XXX: The brownian path
    rng = default_rng()
    m.n = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)

    # XXX: Initial boundary condition
    m.Equation(m.x[0] == np.pi)
    # XXX: Final boundary condition
    m.Equation(m.x[-1] == np.pi/2)

    # XXX: The dynamics (hybrid stochastic system)
    [m.Equation(m.x[i] == m.x[i-1]
                + m.if3(m.x[i-1]-np.pi/2, m.u[i-1], -m.u[i-1])
                + m.n[i-1])
     for i in range(1, R)]

    # XXX: The objective
    m.Obj(0.1*m.sum([i**2 for i in m.u]) +
          0.5*m.sum([(i-np.pi/2)**2 for i in m.x]))

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
    m.options.IMODE = 2         # steady state control
    m.solve(debug=1)

    return (time,
            [i.value for i in m.x],
            [i.value for i in m.u])


if __name__ == '__main__':
    set_plt_params()
    # XXX: How many steps?
    R = 100
    # How big each step
    delta = 0.01                    # total = R*delta second
    ts, tr1s, uref = example(R, delta)
    plt.style.use('ggplot')

    plt.plot(ts, tr1s)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/steeringxminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(ts[:len(ts)-1], uref)
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$u(t)$ (units)', fontweight='bold')
    plt.savefig('/tmp/steeringurefminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
