#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import time
import operator as op
from src.sdesolver import Solver

# FIXME: Check why it goes above the zero-crossing sometimes.
if __name__ == '__main__':
    # np.random.seed(0)         # same as simulink
    # Example dx(t) = -*sgn(x(t)-pi/2) + dw(t)

    L = 3
    N = 1
    # This is the bounds matrix θ for different locations
    T = np.array([[(-np.inf), (np.pi/2)], [(np.pi/2), (np.inf)],
                  [(np.pi/2), (np.pi/2)]])
    T = T.reshape((L, 2, N))

    Tops = np.array([[(op.gt), (op.lt)], [(op.gt), (op.lt)],
                     [(op.ge), (op.le)]])
    Tops = Tops.reshape(L, 2, N)

    # This is the system matrix at different locations
    A = np.array([[0], [0], [0]])
    A = A.reshape(L, N, N)

    # This is the B matrix in the system equation
    B = np.array([[1], [-1], [0]])
    B = B.reshape(L, N)

    # This is the brownian motion matrix
    S = np.array([0]*(L*N*N))
    S = S.reshape(L, N, N)

    # This is the SB matrix for brownian motion
    SB = np.array([1]*(L*N))
    SB = SB.reshape(L, N)

    # ivals = [10]

    for c in [1e-7]:      # The tolerance constant
        ivals = [0.5]            # Just one initial value
        M = 1                    # The number of montecarlo runs
        SIM_TIME = 5.0
        toplot = np.array([])
        timetaken = np.array([])
        # name = __file__.split('.')[1].split('/')[1]
        # name = ('/tmp/results/'+name)+'driver'
        # dfile = name+'_'+str(c)+'.csv'
        # dfile2 = name+'_'+str(c)+'time.csv'
        # The arrays to hold the final result
        for p in range(9, 10):
            err = 0
            aerr = 0
            time1 = 0
            time2 = 0
            avgdt = 0
            avgndt = 0
            for i in range(M):
                solver = Solver(T, Tops, A, B, S, SB, R=2**p, C=c,
                                montecarlo=True)
                print('Doing 2̂ᵖ=%d, M=%d, C=%e' % (2**p, i, c))
                st = time.time()
                vs, ts = solver.simulate(ivals, SIM_TIME)
                avgdt += len(ts)
                avgndt += len(solver.dts)
                time1 += (time.time() - st)
                print('simulate done')
                st = time.time()
                nvs2, nts2 = solver.nsimulate(ivals)
                time2 += (time.time() - st)
                print('nsimulate done')
                err += np.sum(np.square(nvs2[-1] - vs[-1]))
                aerr += np.sum(np.abs((nvs2[-1] - vs[-1])/nvs2[-1]))
                print('Total square error: %f, %f' % (err, aerr))

            print('Total time taken by proposed technique:', time1/M)
            print('Total time taken by naive technique:', time2/M)

            avgndt = SIM_TIME/(avgndt/M)
            print('Average dt:', avgndt)

            avgdt = SIM_TIME/(avgdt/M)
            print('Average Dt:', avgdt)

            # mean_error = np.log(np.sqrt(err/M))
            # aerr = aerr/M
            # bound = 0.5 * np.log(avgdt)
            # bound = 0.5 * np.log((1 + np.log(1/avgdt))) + 0.5 * np.log(avgdt)
            # print('Log Error: %f, Log Bound: %f' % (mean_error, bound))
            # print('O(bound):', 0.5*np.log(avgdt))
            # print('Log error <= Bound', mean_error <= bound)

            # Append to the array to plot it later
            toplot = np.append(toplot, [[avgdt, np.sqrt(err/M), (aerr/M),
                                         avgndt]])
            toplot = toplot.reshape(len(toplot)//4, 4)

            timetaken = np.append(timetaken, [[time1/M, time2/M]])
            timetaken = timetaken.reshape(len(timetaken)//2, 2)
        # np.savetxt(dfile, toplot, header='Dt, RMSE, MAPE, dt', fmt='%+10.10f',
        #            delimiter=',')
        # np.savetxt(dfile2, timetaken, header='PT, NT', fmt='%+10.10f',
        #            delimiter=',')

        # print(ts, vs)
        plt.plot(nts2, nvs2)
        # plt.plot(ts, vs, marker='1')
        # plt.yticks(np.arange(0.5, 1.6, step=0.1))
        plt.grid(which='both')
        plt.show()
