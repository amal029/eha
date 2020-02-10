#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import operator as op
from src.sdesolver import Solver


if __name__ == '__main__':
    np.random.seed(0)
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

    solver = Solver(T, Tops, A, B, S, SB, R=2**10)

    # Initial values
    ivals = [-5, 5]
    vs, ts = solver.simulate(ivals, 2.0)
    xs = [i[0] for i in vs]
    ys = [i[1] for i in vs]

    # Plot the output
    plt.plot(ts[2500:3200], xs[2500:3200])
    plt.show()
    plt.plot(ts[2500:3200], ys[2500:3200])
    plt.show()
    plt.plot(xs, ys)
    plt.show()

    # TODO: Implement the same with same seed with ordinary EM
