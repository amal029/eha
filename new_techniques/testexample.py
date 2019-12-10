#!/usr/bin/env python3
import sympy as S
import simpy
import matplotlib.pyplot as plt
from src.solver import Solver


step = 0                        # The number of integration steps
STOP_TIME = 1                   # in seconds
data = dict()                   # The final result


def lorenz(env, solver, cstate=0):
    """Example of a ha being solved using the new technique.

    """
    # TODO: Need to fix so that this works.
    # odes = {x.diff(solver.t): (x+5)*solver.t}

    x = S.sympify('x(t)')     # x

    # Initial values
    vals_at_tn = {x: 2}
    global data
    data['x'] = [vals_at_tn[x]]
    data['t'] = [0]

    # Ode expressions
    xdt = -6*x + 6

    def location1(x, vals_at_tn):
        # The odes for all continuous variables in location1
        odes = {x.diff(solver.t): xdt}

        # Get the tokens for x
        dict_tokens = {x: solver.build_tokens(x, odes)}

        # First get the polynomial expression from tokens
        xps = {x: solver.get_polynomial(x, tokens, vals_at_tn)
               for x, tokens in dict_tokens.items()}

        # This is the intra-location transition
        # Take the minimum from amongst the two
        h = max(STOP_TIME/50, 0.2)  # Just like simulink
        h = solver.delta((dict_tokens, vals_at_tn), h, env.now)

        # Now compute the new values for continuous variables
        vals_at_tn = {k: solver.get_vals_at_tn_h(x, vals_at_tn, h, env.now)
                      for k, x in xps.items()}
        data['x'].append(vals_at_tn[x])
        data['t'].append(env.now)
        return 0, h, vals_at_tn

    # The dictionary for the switch statement.
    switch_case = {
        0: location1,
    }

    # The initial values at time 0
    print('%f: %s' % (env.now, vals_at_tn[x]))

    # Now start running the system until all events are done or
    # simulation time is over.
    while(True):
        (cstate, delta, vals_at_tn) = switch_case[cstate](x, vals_at_tn)
        # The new values of the continuous variables
        if delta != 0:
            print('%f: %s' % (env.now, vals_at_tn[x]))
        # This should always be the final statement in this function
        global step
        step += 1
        yield env.timeout(delta)


def main():
    # Initiaise the solver
    solver = Solver(n=1, epsilon=1e-2)

    env = simpy.Environment()
    env.process(lorenz(env, solver))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=STOP_TIME)

    # Plot the output
    plt.plot(data['t'], data['x'])
    plt.show()


if __name__ == '__main__':
    main()
