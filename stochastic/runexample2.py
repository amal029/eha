#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import operator as op
from src.sdesolver import Solver


if __name__ == '__main__':
    # Example dx(t) = -5*sgn(x(t)) + (x(t) + 1) dw(t), x(0) = -10

    L = 3
    N = 1
    # This is the bounds matrix θ for different locations
    T = np.array([[(-np.inf), (0)], [(0), (np.inf)], [(0), (0)]])
    T = T.reshape((L, 2, N))

    Tops = np.array([[(op.gt), (op.lt)], [(op.gt), (op.lt)],
                     [(op.ge), (op.le)]])
    Tops = Tops.reshape(L, 2, N)

    # This is the system matrix at different locations
    A = np.array([[0], [0], [0]])
    A = A.reshape(L, N, N)

    # This is the B matrix in the system equation
    B = np.array([[5], [-5], [0]])
    B = B.reshape(L, N)

    # This is the brownian motion matrix
    S = np.array([[1]])
    S = S.reshape(N, N)

    # This is the SB matrix for brownian motion
    SB = np.array([[1]])
    SB = SB.reshape(N, )

    solver = Solver(T, Tops, A, B, S, SB, R=2**14, montecarlo=True)
    ivals = [-10]
    vs, ts = solver.simulate(ivals, 1.0)

    # Plot the output
    print(len(ts))
    plt.plot(ts, vs)
    plt.show()

    # Now doing the same with T = 1, R=R
    M = len(ts)-1
    path = solver.path.reshape(len(solver.path)//M, M)
    path = np.sum(path, axis=1)
    print(len(path), path.shape)
    # Now just need to complete doing the SDE solution normally.
