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

    # XXX: Number of time steps
    time = [i*delta for i in range(R)]

    # XXX: The variables
    m.x1 = [m.Var(value=-5, lb=-5.5, ub=1, name='x1_%s' % i)
            for i in range(R)]
    m.x2 = [m.Var(value=0, lb=-0.5, ub=3, name='x2_%s' % i)
            for i in range(R)]

    # XXX: The continuous control variable
    m.u = [m.Var(lb=-1, ub=1, fixed_initial=False,
                 name='u_%s' % i)
           for i in range(R-1)]

    # XXX: The discrete control variable
    m.g = [m.Var(lb=0, ub=1, integer=True, fixed_initial=False,
                 name='g_%s' % i)
           for i in range(R-1)]

    # XXX: The brownian path
    rng = default_rng()
    m.n1 = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)
    m.n2 = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)

    # XXX: The initial boundary conditions
    m.Equation(m.x1[0] == -5)
    m.Equation(m.x2[0] == 0)

    # XXX: The final boundary condition
    m.Equation(m.x1[-1] == 0)
    m.Equation(m.x2[-1] == 0)

    # XXX: The dynamics
    [m.Equation(m.x1[i] == m.x1[i-1] + m.x2[i-1]*delta + 0.1*m.n1[i-1])
     for i in range(1, R)]
    [m.Equation(m.x2[i] == m.x2[i-1] +
                (1-m.g[i-1])*(1/(1+m.exp(-5*(m.x2[i-1]-0.5))))*m.u[i-1] +
                (m.g[i-1])*(1/(1+m.exp(5*(m.x2[i-1]-0.5))))*m.u[i-1] +
                # XXX: This is the 0.01*dW(t), noise term
                0.01*m.n2[i-1])
     for i in range(1, R)]

    # XXX: The objective
    # FIXME: This should be minimize time
    m.Obj(
        4*m.sum([i**2 for i in m.x1]) +
        4*m.sum([i**2 for i in m.x2]) +
        0.05*m.sum([(i-1)**2 for i in m.g]) +
        0.005*m.sum([i**2 for i in m.u])
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
    m.options.IMODE = 2         # steady state control
    m.options.MAX_TIME = 30
    m.solve(debug=1)

    return (time,
            [i.value for i in m.x1],
            [i.value for i in m.x2],
            [i.value for i in m.u],
            [i.value for i in m.g])


if __name__ == '__main__':
    set_plt_params()
    R = 150
    # How big each step
    delta = 0.04                    # total = R*delta second

    N = 31
    x1s = []
    x2s = []

    i = 0
    while(i < N):
        try:
            ts, tr1s, tr2s, _, _ = example(R, delta)
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
    # tn = 1.96
    tn = 2.576
    x1CI = [tn*i/np.sqrt(N) for i in sigma1]
    x1CIplus = [i + j for i, j in zip(meanx1, x1CI)]
    x1CIminus = [i - j for i, j in zip(meanx1, x1CI)]

    x2CI = [tn*i/np.sqrt(N) for i in sigma1]
    x2CIplus = [i + j for i, j in zip(meanx2, x2CI)]
    x2CIminus = [i - j for i, j in zip(meanx2, x2CI)]

    plt.style.use('ggplot')
    plt.plot(ts, meanx1, label='Mean x1(t)')
    plt.plot(ts, x1CIplus, label='CI 99% upper bound')
    plt.plot(ts, x1CIminus, label='CI 99% lower bound')
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x1(t)$ (units)', fontweight='bold')
    plt.legend(loc='best')
    plt.savefig('gearsstochasticx1minlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(ts, meanx2, label='Mean x2(t)')
    plt.plot(ts, x2CIplus, label='CI 99% upper bound')
    plt.plot(ts, x2CIminus, label='CI 99% lower bound')
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x2(t)$ (units)', fontweight='bold')
    plt.legend(loc='best')
    plt.savefig('gearsstochasticx2minlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(x1CIminus, x2CIminus, marker='1', label='CI 99% lower bound',
             markevery=5)
    plt.plot(meanx1, meanx2, linestyle='--', marker='+',
             label='Mean trajectory', markevery=5)
    plt.plot(x1CIplus, x2CIplus, marker='2', label='CI 99% upper bound',
             markevery=5)
    plt.legend(loc='best')
    plt.xlabel('x1(t)')
    plt.ylabel('x2(t)')
    plt.savefig('gears.pdf', bbox_inches='tight')
    plt.show()
    plt.close()
