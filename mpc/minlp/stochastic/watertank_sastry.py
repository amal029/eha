#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
from gekko import GEKKO
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D

# XXX: Paper: Consistent approximations for the optimal control of
# constrained switched systems—Part 2: An implementable algorithm


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
    print(m._path)
    # XXX: Set the time for the model
    time = [i*delta for i in range(R)]

    # XXX: Discrere control variables
    m.d = [m.Var(lb=0, ub=1, integer=True, name='d_%s' % i)
           for i in range(R-1)]

    # XXX: state variables
    m.x1 = [m.Var(name='x1_%s' % i) for i in range(R)]
    m.x2 = [m.Var(name='x2_%s' % i) for i in range(R)]

    # # XXX: Continous control varible
    m.u = [m.Var(name='u_%s' % i, lb=1, ub=2) for i in range(R-1)]

    rng = default_rng()
    m.n1 = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)
    m.n2 = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)

    # Now the dynamics
    [m.Equation(m.x1[i] == m.x1[i-1] +
                m.d[i-1]*((m.u[i-1] - m.sqrt(m.x1[i-1]))*delta
                          + 0.1*m.n1[i-1])
                + (1-m.d[i-1])*((m.u[i-1] - m.sqrt(m.x1[i-1]))*delta
                                + 0.05*m.n1[i-1]))
     for i in range(1, R)]

    [m.Equation(m.x2[i] == m.x2[i-1] +
                (m.sqrt(m.x1[i-1]) - m.sqrt(m.x2[i-1]))*delta
                + 0.02*m.n2[i-1])
     for i in range(1, R)]

    # XXX: Initial boundary condition
    m.Equation(m.x1[0] == 2)
    m.Equation(m.x2[0] == 2)
    m.Equation(m.d[0] == 1)

    # XXX: final boundary condition
    m.Equation(m.x2[-1] == 3)

    # XXX: Objective
    m.Obj(2*m.sum([(i-3)**2 for i in m.x2]))

    m.options.SOLVER = 1        # APOPT solver
    # m.options.NODES = 5        # number of colocation points
    m.solver_options = ['minlp_maximum_iterations 10000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 100', \
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
    m.options.IMODE = 2         # steady state
    m.options.MAX_TIME = 5
    m.solve(debug=1)

    # XXX: convert binary to number
    return (time,
            [i.value for i in m.x1],
            [i.value for i in m.x2],
            [i.value for i in m.d])


if __name__ == '__main__':
    set_plt_params()
    R = 100
    delta = 0.1
    N = 31
    i = 0
    x1s = []
    x2s = []
    while(i < N):
        try:
            ts, tr1s, tr2s, _ = example(R, delta)
        except Exception:
            pass
        else:
            i += 1
            x1s.append([j for i in tr1s for j in i])
            x2s.append([j for i in tr2s for j in i])

    meanx1 = [0]*R
    meanx2 = [0]*R
    for i in range(R):
        for j in range(N):
            meanx1[i] += x1s[j][i]
            meanx2[i] += x2s[j][i]

        meanx1[i] /= N           # mean at time points
        meanx2[i] /= N           # mean at time points

    # XXX: Now the standard deviation
    sigma1 = [0]*R
    sigma2 = [0]*R
    for i in range(R):
        for j in range(N):
            sigma1[i] += (meanx1[i]-x1s[j][i])**2
            sigma2[i] += (meanx2[i]-x2s[j][i])**2

        sigma1[i] /= N
        sigma1[i] = np.sqrt(sigma1[i])

        sigma2[i] /= N
        sigma2[i] = np.sqrt(sigma2[i])

    # XXX: Now compute the envelope
    tn = 2.576
    x1CI = [tn*i/np.sqrt(N) for i in sigma1]
    x1CIplus = [i + j for i, j in zip(meanx1, x1CI)]
    x1CIminus = [i - j for i, j in zip(meanx1, x1CI)]

    x2CI = [tn*i/np.sqrt(N) for i in sigma1]
    x2CIplus = [i + j for i, j in zip(meanx2, x2CI)]
    x2CIminus = [i - j for i, j in zip(meanx2, x2CI)]

    plt.style.use('ggplot')
    plt.plot(ts, meanx1, label='Mean x1(t)', linestyle='--', marker='+')
    plt.plot(ts, x1CIplus, label='x1(t) CI 99% upper bound', marker='2')
    plt.plot(ts, x1CIminus, label='x1(t) CI 99% lower bound', marker='1')
    plt.plot(ts, meanx2, label='Mean x2(t)', linestyle='--', marker='|')
    plt.plot(ts, x2CIplus, label='x2(t) CI 99% upper bound', marker='4')
    plt.plot(ts, x2CIminus, label='x2(t) CI 99% lower bound', marker='3')
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t), x2(t)$ (lts)', fontweight='bold')
    plt.legend(loc='best')
    plt.savefig('watertanksastrystochastic.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # plt.plot(ts[:len(ts)-1], ds, label='d(t)')
    # plt.legend(loc='best')
    # plt.show()
    # plt.close()
