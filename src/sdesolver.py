#!/usr/bin/env python3

import numpy as np
import mpmath as mp


class Solver(object):
    def __init__(self, T=None, A=None, B=None, S=None, SB=None):
        # First check the dimensions of each matrix
        # L is the number of locations
        # _ should always be 2, for start and end
        # N is the number of continuous variables in the system
        (L, _, N) = T.shape
        self.T = T

        # There should always be a start and end for each variable.
        assert _ == 2

        # Now check that A is of size L x N x N
        assert A.shape == (L, N, N)
        self.A = A

        # Now check that B is of size L x N
        assert B.shape == (L, N)
        self.B = B
        print(B[0])

        # Now check that S is of size N x N
        assert S.shape == (N, N)
        self.S = S

        # Now check that SB is of size N x 1
        assert SB.shape == (N, 1)
        self.SB = SB


if __name__ == '__main__':
    # Example dx(t) = -sgn(x(t)) + dw(t), x(0) = 10

    L = 3
    N = 1
    # This is the bounds matrix θ for different locations
    T = np.array([[(-np.inf), (0)], [(0), (np.inf)], [(0), (0)]])
    T = T.reshape((L, 2, N))

    # This is the system matrix at different locations
    A = np.array([[0], [0], [0]])
    A = A.reshape(L, N, N)

    # This is the B matrix in the system equation
    B = np.array([[1], [-1], [0]])
    B = B.reshape(L, N)

    # This is the brownian motion matrix
    S = np.array([[0]])
    S = S.reshape(N, N)

    # This is the SB matrix for brownian motion
    SB = np.array([[1]])
    SB = SB.reshape(N, 1)

    solver = Solver(T, A, B, S, SB)
