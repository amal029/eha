#!/usr/bin/env python3

from src.mpc import MPC
import numpy


def example():
    # XXX: Model a simple linear moving robot at constant velocity with
    # disturbance. Control it using MPC
    # The step-size
    d = 0.01
    # The time horizon (second)
    h = 0.1
    N = int(h//d)   # The number of steps in MPC
    p = (lambda x: x[0] + (1 + x[0]*0.1*x[1])*d)     # Plant model (Euler)

    # XXX: The noisy plant model, which we don't know about
    pn = (lambda x: x[0] + numpy.random.rand() + (1 + x[0]*0.1*x[1])*d)
    # FIXME: If the std-deviation is huge then SMT seems to crap out

    rx = [[5]]*N    # The ref point for system state
    ru = [[0]]*N    # The ref point for control input
    # XXX: The bounds for state and control inputs
    xl = []
    xu = []
    ul = [-5000]
    uu = [5000]
    # XXX: Optimisation weights, equal optimisation
    xw = [1]
    uw = [0]
    # XXX: Initial values for state and control inputs
    x0 = [0.5]

    actions = []
    xs = [x0]
    count = 0

    # Get the solver
    s = MPC(N, 1, 1, [p], rx, ru, xw, uw, xl, xu, ul, uu, norm=None)
    # XXX: Start simulating the movement of the robot
    while(count < h):
        print('------------l∞ norm cost function-------------')
        u0 = s.solve(x0)

        # XXX: Apply the action to the plant
        # FIXME: Make this plant model have some randomness
        x0 = [pn(x0 + u0)]
        print(x0, u0)
        # Append to list for plotting
        actions += [u0]
        xs += [x0]
        # Increment time
        count += d

    print('xs:', xs)
    print('us:', actions)


if __name__ == '__main__':
    example()
