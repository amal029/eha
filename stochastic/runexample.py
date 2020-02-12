#!/usr/bin/env python3

import numpy as np
# import matplotlib.pyplot as plt
import operator as op
from src.sdesolver import Solver
import time

if __name__ == '__main__':
    # np.random.seed(0)
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

    S = np.array([0, 0, 0, 0])
    S = S.reshape(N, N)

    SB = np.array([1, 1])
    SB = SB.reshape(N, )

    # Initial values
    ivals = [-5, 5]

    M = 10                    # The number of montecarlo runs
    SIM_TIME = 1.0
    toplot = np.array([])
    timetaken = np.array([])
    dfile = (__file__.split('.')[1].split('/')[1])+'.csv'
    dfile2 = (__file__.split('.')[1].split('/')[1])+'time.csv'
    # The arrays to hold the final result
    for p in range(8, 13):
        err = 0
        time1 = 0
        time2 = 0
        avgdt = 0
        for i in range(M):
            solver = Solver(T, Tops, A, B, S, SB, R=2**p, montecarlo=True)
            print('Doing 2̂ᵖ=%d, M=%d' % (2**p, i))
            st = time.time()
            vs, ts = solver.simulate(ivals, SIM_TIME)
            avgdt += len(ts)
            time1 += (time.time() - st)
            print('simulate done')
            st = time.time()
            nvs2, nts2 = solver.nsimulate(ivals)
            time2 += (time.time() - st)
            print('nsimulate done')
            err += np.sum(np.square(nvs2[-1] - vs[-1]))
            print('Total square error: %f' % err)

        print('Total time taken by proposed technique:', time1/M)
        print('Total time taken by naive technique:', time2/M)
        avgdt = SIM_TIME/(avgdt/M)
        print('Average Dt:', avgdt)
        mean_error = np.log(np.sqrt(err/M))
        bound = 0.5 * np.log((1 + np.log(1/avgdt))) + 0.5 * np.log(avgdt)
        print('Log Error: %f, Log Bound: %f' % (mean_error, bound))
        print('Log error <= Bound', mean_error <= bound)
        # Append to the array to plot it later
        toplot = np.append(toplot, [[avgdt, mean_error]])
        toplot = toplot.reshape(len(toplot)//2, 2)

        timetaken = np.append(timetaken, [[time1/M, time2/M]])
        timetaken = timetaken.reshape(len(timetaken)//2, 2)
    np.savetxt(dfile, toplot, header='Dt, Err', fmt='%+10.10f', delimiter=',')
    np.savetxt(dfile2, timetaken, header='PT, NT', fmt='%+10.10f',
               delimiter=',')

    # xs = [i[0] for i in vs]
    # ys = [i[1] for i in vs]

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
    # xs = [i[0] for i in nvs2]
    # ys = [i[1] for i in nvs2]
    # plt.plot(xs, ys)
    # plt.show()
