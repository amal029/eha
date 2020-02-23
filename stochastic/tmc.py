#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import operator as op
from src.sdesolver import Solver
import time

# FIXME: Something gives me a slope of zero in this example.
if __name__ == '__main__':
    L = 9
    N = 2
    T = np.array([[(0, 0), (0, 0)],      # (=0, =0)
                  [(0, 0), (0, np.inf)],  # (=0, >0)
                  [(0, -np.inf), (0, 0)],  # (=0 <0)
                  [(-np.inf, 0), (0, 0)],  # (<0, =0)
                  [(0, 0), (np.inf, 0)],   # (>0, =0)
                  [(-np.inf, -np.inf), (0, 0)],  # (<0 <0)
                  [(-np.inf, 0), (0, np.inf)],   # (<0 >0)
                  [(0, -np.inf), (np.inf, 0)],   # (>0 <0)
                  [(0, 0), (np.inf, np.inf)]])   # (>0 >0)
    T = T.reshape(L, 2, N)
    Tops = [[(op.ge, op.ge), (op.le, op.le)],
            [(op.ge, op.gt), (op.le, op.lt)],
            [(op.ge, op.gt), (op.le, op.lt)],
            [(op.gt, op.ge), (op.lt, op.le)],
            [(op.gt, op.ge), (op.lt, op.le)],
            [(op.gt, op.gt), (op.lt, op.lt)]*(L-5)]
    Tops = np.array([item for sublist in Tops for item in sublist])
    Tops = Tops.reshape(L, 2, N)

    # First y(t) then x(t)
    A = np.array([[0, 0], [1, 0]]*L)
    A = A.reshape(L, N, N)
    # print(A)

    b = 3
    s = 1
    # Now comes the control input of size B
    fy = (lambda x, y: -(b+s)/2*np.sign(x) - (b-s)/2*np.sign(y))
    B = np.array([[fy(0, 0), 0],
                  [fy(0, 1), 0],
                  [fy(0, -1), 0],
                  [fy(-1, 0), 0],
                  [fy(6, 0), 0],
                  [fy(-4, -4), 0],
                  [fy(-4, 4), 0],
                  [fy(6, -1), 0],
                  [fy(6, 4), 0]])
    B = B.reshape(L, N)
    # print(B)

    S = np.array([0]*(L*N*N))
    S = S.reshape(L, N, N)

    # Now comes the control input of size B
    SB = np.append(np.array([0]), [1]*((L-1)*N))
    SB = B.reshape(L, N)

    for c in [1e-4]:      # The tolerance constant
        # ivals = [5, 1]
        ivals = [2, 3]
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
        for p in range(8, 9):
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
    # plt.plot(xs, ys, marker='1')
    # xs = [i[0] for i in nvs2]
    # ys = [i[1] for i in nvs2]
    plt.style.use('ggplot')
    plt.plot(xs, ys)
    plt.show()
