#!/usr/bin/env python3

import numpy as np
from numpy.random import default_rng
from gekko import GEKKO
from matplotlib.patches import Rectangle
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
    print(m._path)

    # XXX: Number of time steps
    time = [i*delta for i in range(R)]

    # XXX: The variables
    m.x = [m.Var(name='x_%s' % i, lb=0, ub=10) for i in range(R)]
    m.y = [m.Var(name='y_%s' % i, lb=2, ub=4) for i in range(R)]
    m.a = [m.Var(name='a_%s' % i, lb=0, ub=2*np.pi) for i in range(R)]

    # XXX: The continuous control variable
    m.p = [m.Var(lb=-1, ub=1, fixed_initial=False,
                 name='p_%s' % i)
           for i in range(R-1)]
    # XXX: Angle ω
    m.q = [m.Var(lb=-0.5, ub=0.5, fixed_initial=False,
                 name='q_%s' % i)
           for i in range(R-1)]

    # XXX: The discrete control variable
    m.g = [m.Var(lb=0, ub=1, integer=True, fixed_initial=False,
                 name='g_%s' % i)
           for i in range(R-1)]

    # XXX: The brownian path
    rng = default_rng()
    m.e1m = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)
    m.e2m = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)
    m.e3m = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)

    m.e1t = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)
    m.e2t = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)
    m.e3t = rng.normal(loc=0, scale=np.sqrt(delta), size=R-1)

    # XXX: The initial boundary conditions
    m.Equation(m.x[0] == 2)
    m.Equation(m.y[0] == 3)
    # m.Equation(m.a[0] == np.pi/2)

    # XXX: The final boundary condition
    m.Equation(m.x[-1] == 5)
    m.Equation(m.y[-1] == 2)

    # XXX: The dynamics
    [m.Equation(m.x[i] == m.x[i-1] +
                # XXX: Location "Move"
                (1-m.g[i-1])*(m.p[i-1]*m.cos(m.a[i-1])*delta +
                              0.1*m.e1m[i-1]*m.cos(m.a[i-1])) +
                # XXX: Location "Turn"
                m.g[i-1]*(0.1*m.e1t[i-1]))
     for i in range(1, R)]

    [m.Equation(m.y[i] == m.y[i-1] +
                # XXX: Location "Move"
                (1-m.g[i-1])*(m.p[i-1]*m.sin(m.a[i-1])*delta +
                              0.2*m.e2m[i-1]*m.sin(m.a[i-1])) +
                # XXX: Location "Turn"
                m.g[i-1]*(0.2*m.e2t[i-1]))
     for i in range(1, R)]

    [m.Equation(m.a[i] == m.a[i-1] +
                # XXX: Location "Move"
                (1-m.g[i-1])*(0.1*m.e3m[i-1]) +
                # XXX: Location "Turn"
                m.g[i-1]*(m.q[i-1]*delta + 0.1*m.e3t[i-1]))
        for i in range(1, R)]

    # XXX: Draw obstacle from 3-4 on x and all except 2-4 on y

    # XXX: Objective
    m.Obj(
        40*m.sum([(i-5)**2 for i in m.x])
        + 40*m.sum([(i-2)**2 for i in m.y])
        + m.sum([i**2 for i in m.p])
        + m.sum([i**2 for i in m.q])
        + m.sum([i**2 for i in m.g])
    )

    m.options.SOLVER = 1        # APOPT solver
    m.solver_options = ['minlp_maximum_iterations 10000', \
                        # minlp iterations with integer solution
                        'minlp_max_iter_with_int_sol 500', \
                        # treat minlp as nlp
                        'minlp_as_nlp 0', \
                        # nlp sub-problem max iterations
                        'nlp_maximum_iterations 100', \
                        # 1 = depth first, 2 = breadth first
                        'minlp_branch_method 1', \
                        # maximum deviation from whole number
                        'minlp_integer_tol 0.0005', \
                        # covergence tolerance
                        'minlp_gap_tol 0.01']
    m.options.IMODE = 2         # steady state control
    m.options.MAX_TIME = 5
    # m.options.DIAGLEVEL = 2
    # m.options.COLDSTART = 2
    m.solve(debug=2)

    return (time,
            [i.value for i in m.x],
            [i.value for i in m.y],
            [i.value for i in m.a],
            [i.value for i in m.p],
            [i.value for i in m.q],
            [i.value for i in m.g])


if __name__ == '__main__':
    set_plt_params()
    R = 40
    # How big each step
    delta = 0.2                    # total = R*delta second

    N = 31

    xs = []
    ys = []
    i = 0

    while(i < N):
        try:
            ts, tr1s, tr2s, _, _, _, _ = example(R, delta)
            # i += 1
            # xs.append([j for i in tr1s for j in i])
            # ys.append([j for i in tr2s for j in i])
        except Exception:
            pass
        else:
            i += 1
            xs.append([j for i in tr1s for j in i])
            ys.append([j for i in tr2s for j in i])

    meanx1 = [0]*R
    meanx2 = [0]*R
    for i in range(R):
        for j in range(N):
            meanx1[i] += xs[j][i]
            meanx2[i] += ys[j][i]

        meanx1[i] /= N           # mean at time points
        meanx2[i] /= N           # mean at time points

    # XXX: Now the standard deviation
    sigma1 = [0]*R
    sigma2 = [0]*R
    for i in range(R):
        for j in range(N):
            sigma1[i] += (meanx1[i]-xs[j][i])**2
            sigma2[i] += (meanx2[i]-ys[j][i])**2

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
    plt.plot(ts, meanx1, label='Mean x(t)')
    plt.plot(ts, x1CIplus, label='CI 99% upper bound')
    plt.plot(ts, x1CIminus, label='CI 99% lower bound')
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$x(t)$ (units)', fontweight='bold')
    plt.savefig('motionstochasticxsminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    plt.plot(ts, meanx2, label='Mean x2(t)')
    plt.plot(ts, x2CIplus, label='CI 99% upper bound')
    plt.plot(ts, x2CIminus, label='CI 99% lower bound')
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel(r'$y(t)$ (units)', fontweight='bold')
    plt.savefig('motionstochasticysminlp.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    fig, ax = plt.subplots()
    ax.plot(x1CIminus, x2CIminus, marker='1', label='CI 99% lower bound')
    ax.plot(meanx1, meanx2, linestyle='--', marker='+',
            label='Mean Trajectory')
    ax.plot(x1CIplus, x2CIplus, marker='2', label='CI 99% upper bound')
    ax.add_patch(Rectangle((3, 4), 1, 6))
    ax.add_patch(Rectangle((3, 0), 1, 2))
    plt.xlabel('x(t)')
    plt.ylabel('y(t)')
    plt.legend(loc='best')
    plt.savefig('motionplanning.pdf', bbox_inches='tight')
    plt.show()
    plt.close()

    # plt.plot(ts, aas)
    # plt.xlabel('Time (seconds)', fontweight='bold')
    # plt.ylabel(r'$\omega(t)$ (units)', fontweight='bold')
    # plt.savefig('/tmp/motionstochasticaasminlp.pdf', bbox_inches='tight')
    # plt.show()
    # plt.close()

    # plt.plot(ts[:len(ts)-1], ps)
    # plt.xlabel('Time (seconds)', fontweight='bold')
    # plt.ylabel(r'$p(t)$ (units)', fontweight='bold')
    # plt.savefig('/tmp/motionstochasticurefminlp.pdf', bbox_inches='tight')
    # plt.show()
    # plt.close()

    # plt.plot(ts[:len(ts)-1], qs)
    # plt.xlabel('Time (seconds)', fontweight='bold')
    # plt.ylabel(r'$q(t)$ (units)', fontweight='bold')
    # plt.savefig('/tmp/motionstochasticurefminlp.pdf', bbox_inches='tight')
    # plt.show()
    # plt.close()

    # plt.plot(ts[:len(ts)-1], gref)
    # plt.xlabel('Time (seconds)', fontweight='bold')
    # plt.ylabel(r'$g(t)$ (units)', fontweight='bold')
    # plt.savefig('/tmp/motionstochasticgrefminlp.pdf', bbox_inches='tight')
    # plt.show()
    # plt.close()
