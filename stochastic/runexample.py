#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import operator as op
from src.sdesolver import Solver
import time

if __name__ == '__main__':
    L = 9
    N = 2
    T = np.array([[(5, 0), (5, 0)],
                  [(5, 0), (5, np.inf)],
                  [(5, -np.inf), (5, 0)],
                  [(-np.inf, 0), (5, 0)],
                  [(5, 0), (np.inf, 0)],
                  [(-np.inf, -np.inf), (5, 0)],
                  [(-np.inf, 0), (5, np.inf)],
                  [(5, -np.inf), (np.inf, 0)],
                  [(5, 0), (np.inf, np.inf)]])
    T = T.reshape(L, 2, N)
    Tops = [[(op.ge, op.ge), (op.le, op.le)],
            [(op.ge, op.gt), (op.le, op.lt)],
            [(op.ge, op.gt), (op.le, op.lt)],
            [(op.gt, op.ge), (op.lt, op.le)],
            [(op.gt, op.ge), (op.lt, op.le)],
            [(op.gt, op.gt), (op.lt, op.lt)]*(L-5)]
    Tops = np.array([item for sublist in Tops for item in sublist])
    Tops = Tops.reshape(L, 2, N)

    A = np.append(np.array([[0, 0], [0, 0],
                            [0, 0], [1, 1],
                            [0, 0], [1, 1],
                            [0, 1], [0, 0],
                            [0, 1], [0, 0]]),
                  np.array([[0, 1], [1, 1]]*(L-5)))
    A = A.reshape(L, N, N)
    # print(A)

    alpha = 4                   # This is a parameter
    beta = -10
    # Now comes the control input of size B
    fx1 = (lambda x: -alpha*np.sign(x-5))
    fx2 = (lambda x: beta*np.sign(x))
    B = np.array([[fx1(5), fx2(0)],
                  [fx1(5), fx2(1)],
                  [fx1(5), fx2(-1)],
                  [fx1(-5), fx2(0)],
                  [fx1(6), fx2(0)],
                  [fx1(-4), fx2(-1)],
                  [fx1(-4), fx2(1)],
                  [fx1(6), fx2(-1)],
                  [fx1(6), fx2(1)]])
    B = B.reshape(L, N)
    # print(B)

    S = np.array([0]*(L*N*N))
    S = S.reshape(L, N, N)

    sbalpha = 1                   # This is a parameter
    sbbeta = -1
    # Now comes the control input of size B
    fx1 = (lambda x: -sbalpha*np.sign(x-5))
    fx2 = (lambda x: sbbeta*np.sign(x))
    SB = np.array([[fx1(5), fx2(0)],
                  [fx1(5), fx2(1)],
                  [fx1(5), fx2(-1)],
                  [fx1(-5), fx2(0)],
                  [fx1(6), fx2(0)],
                  [fx1(-4), fx2(-1)],
                  [fx1(-4), fx2(1)],
                  [fx1(6), fx2(-1)],
                  [fx1(6), fx2(1)]])
    SB = B.reshape(L, N)

    for c in [1e-4]:      # The tolerance constant
        # ivals = [5, 1]
        ivals = [-0.5, 5.4]
        M = 1                    # The number of montecarlo runs
        SIM_TIME = 1.0
        toplot = np.array([])
        timetaken = np.array([])
        # name = __file__.split('.')[1].split('/')[1]
        # name = '/tmp/results/'+name+'new'
        # dfile = name+'_'+str(c)+'.csv'
        # dfile2 = name+'_'+str(c)+'time.csv'
        # print(dfile, dfile2)
        # The arrays to hold the final result
        for p in range(3, 4):
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
        # np.savetxt(dfile, toplot, header='Dt, RMSE, MAPE, dt',
        # fmt='%+10.10f', delimiter=',')
        # np.savetxt(dfile2, timetaken, header='PT, NT', fmt='%+10.10f',
        #            delimiter=',')

    xs = [i[0] for i in vs]
    ys = [i[1] for i in vs]

    # Plot the output
    # plt.plot(ts[2500:3200], xs[2500:3200])
    # plt.show()
    # plt.plot(ts[2500:3200], ys[2500:3200])
    # plt.show()
    # print(len(ts))
    # plt.plot(xs, ys)
    # plt.show()

    # TODO: Implement the same with same seed with ordinary EM
    # print(solver.path.shape, solver.dts.shape)
    plt.plot(xs, ys, marker='1')
    xs = [i[0] for i in nvs2]
    ys = [i[1] for i in nvs2]
    plt.plot(xs, ys)
    plt.show()
