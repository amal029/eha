#!/usr/bin/env python3
import numpy as np
from gekko import GEKKO
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D


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
    nt = 30
    tm = np.linspace(0, 2, nt)

    # XXX: Initialize GEKKO
    m = GEKKO(remote=False)
    # XXX: Set the time for the model
    m.time = tm

    # XXX: Discrere control variables
    m.d1 = m.Var(value=0, lb=0, ub=1, integer=True,
                 fixed_initial=False, name='d1')
    m.d2 = m.Var(value=1, lb=0, ub=1, integer=True,
                 fixed_initial=False, name='d2')

    # XXX: state variables
    m.x1 = m.Var(value=0, fixed_initial=True, name='x1')
    m.x2 = m.Var(value=0, fixed_initial=True, name='x2')
    m.x3 = m.Var(value=0, fixed_initial=True, name='x3')

    # Continuous control variable
    m.u = m.Var(value=0.3, lb=-20, ub=20, name='u')

    # Now the switches
    m.Equation(m.x1.dt() == m.d1*(1-m.d2) *
               (1.0979*m.x1 - 0.0105*m.x2 + 0.0167*m.x3 + 0.9801*m.u) +
               (1-m.d1)*m.d2*(1.0979*m.x1 - 0.0105*m.x2 + 0.0167*m.x3 +
                              0.1743*m.u) +
               m.d1*m.d2*(1.0979*m.x1 - 0.0105*m.x2 +
                          0.0167*m.x3 + 0.0952*m.u) + 0
               # (1-m.d1)*(1-m.d2)*0
               )

    m.Equation(m.x2.dt() == m.d1*(1-m.d2) *
               (-0.0105*m.x1 + 1.0481*m.x2 + 0.0825*m.x3 - 0.1987*m.u) +
               (1-m.d1)*m.d2*(-0.0105*m.x1 + 1.0481*m.x2 + 0.0825*m.x3 -
                              0.8601*m.u) +
               m.d1*m.d2*(-0.0105*m.x1 + 1.0481*m.x2 + 0.0825*m.x3 +
                          0.4699*m.u) + 0
               # (1-m.d1)*(1-m.d2)*0
               )

    m.Equation(m.x3.dt() == m.d1*(1-m.d2) *
               (0.0167*m.x1 + 0.0825*m.x2 + 1.1540*m.x3 + 0*m.u) +
               (1-m.d1)*m.d2*(0.0167*m.x1 + 0.0825*m.x2 + 1.1540*m.x3 -
                              0.4794*m.u) +
               m.d1*m.d2*(0.0167*m.x1 + 0.0825*m.x2 +
                          1.1540*m.x3 + 0.8776*m.u)
               # (1-m.d1)*(1-m.d2)*0
               )

    # XXX: Initial boundary condition
    m.fix_initial(m.x1, 0)
    m.fix_initial(m.x2, 0)
    m.fix_initial(m.x3, 0)

    # XXX: final boundary condition
    m.fix_final(m.x2, 1)
    m.fix_final(m.x1, 1)
    m.fix_final(m.x3, 1)

    # XXX: Objective
    m.Obj(m.integral(0.01*m.u**2)
          # + m.integral((m.x1-1)**2)
          # + m.integral((m.x2-1)**2)
          # + m.integral((m.x3-1)**2)
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
                        'minlp_gap_tol 0.0001']
    m.options.IMODE = 5         # simultaneous dynamic collocation
    m.solve(debug=1)

    # XXX: convert binary to number
    def convert(d1, d2):
        toret = []
        for (i, j) in zip(d1, d2):
            if(i == 1 and j == 0):
                toret.append(1)
            elif(i == 0 and j == 1):
                toret.append(2)
            elif(i == 1 and j == 1):
                toret.append(3)
            else:
                toret.append(0)
        return toret

    return (m.time, m.x1.value, m.x2.value, m.x3.value,
            m.u.value, convert(m.d1.value, m.d2.value))


if __name__ == '__main__':
    set_plt_params()
    ts, x1s, x2s, x3s, us, ds = example()
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # print(list(zip(x1s, x2s, x3s)))
    ax.scatter(x1s, x2s, x3s, marker='+')
    ax.set_xlabel('x1(t)')
    ax.set_ylabel('x2(t)')
    ax.set_zlabel('x3(t)')
    ax.view_init(33, -150)
    plt.show()
    plt.close()

    plt.plot(ts, ds, label='d(t)')
    plt.legend(loc='best')
    plt.show()
    plt.close()

    plt.plot(ts, us, label='u(t)')
    plt.legend(loc='best')
    plt.show()
    plt.close()
