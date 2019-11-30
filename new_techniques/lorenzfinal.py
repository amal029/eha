#!/usr/bin/env python3
import sympy as S
import simpy
import matplotlib.pyplot as plt
from src.solver import Solver


step = 0                        # The number of integration steps
STOP_TIME = 10                   # in seconds
data = dict()                   # The final result


def lorenz(env, solver, cstate=0):
    """Example of a ha being solved using the new technique.

    """
    # TODO: Need to fix so that this works.
    # odes = {x.diff(solver.t): (x+5)*solver.t}

    # First initialise the continuous variables.
    s = 10                    # σ
    la = 28                   # λ
    b = 8/3                   # β

    x1 = S.sympify('x1(t)')     # x
    x2 = S.sympify('x2(t)')     # y
    x3 = S.sympify('x3(t)')     # z

    # Initial values
    vals_at_tn = {x1: 1, x2: 1, x3: 1}
    global data
    data['x'] = [vals_at_tn[x1]]
    data['y'] = [vals_at_tn[x2]]
    data['z'] = [vals_at_tn[x3]]

    # Ode expressions
    x1dt = s*(x2-x1)
    x2dt = ((la-x3)*x1)-x2
    x3dt = x1*x2 - b*x3

    def location1(x1, x2, x3, vals_at_tn):
        # The odes for all continuous variables in location1
        odes = {x1.diff(solver.t): x1dt,
                x2.diff(solver.t): x2dt,
                x3.diff(solver.t): x3dt}

        # Get the tokens for x
        dict_tokens = {x1: solver.build_tokens(x1, odes),
                       x2: solver.build_tokens(x2, odes),
                       x3: solver.build_tokens(x3, odes)}

        # First get the polynomial expression from tokens
        xps = {x: solver.get_polynomial(x, tokens, vals_at_tn)
               for x, tokens in dict_tokens.items()}

        # This is the intra-location transition
        # Take the minimum from amongst the two
        h = max(STOP_TIME/50, 0.2)  # Just like simulink
        h = solver.delta((dict_tokens, vals_at_tn), h, env.now)

        # Now compute the new values for continuous variables
        vals_at_tn = {k: solver.get_vals_at_tn_h(x, vals_at_tn, h)
                      for k, x in xps.items()}
        data['x'].append(vals_at_tn[x1])
        data['y'].append(vals_at_tn[x2])
        data['z'].append(vals_at_tn[x3])
        return 0, h, vals_at_tn

    # The dictionary for the switch statement.
    switch_case = {
        0: location1,
    }

    # The initial values at time 0
    # print('%f: %s %s %s' % (env.now, vals_at_tn[x1], vals_at_tn[x2],
    #                         vals_at_tn[x3]))

    # Now start running the system until all events are done or
    # simulation time is over.
    while(True):
        (cstate, delta, vals_at_tn) = switch_case[cstate](x1, x2, x3,
                                                          vals_at_tn)
        # The new values of the continuous variables
        # if delta != 0:
        #     print('%f: %s %s %s' % (env.now, vals_at_tn[x1], vals_at_tn[x2],
        #                             vals_at_tn[x3]))
        # This should always be the final statement in this function
        global step
        step += 1
        yield env.timeout(delta)


def main():
    # Initiaise the solver
    solver = Solver(epsilon=1e-1)

    env = simpy.Environment()
    env.process(lorenz(env, solver))
    # Run the simulation until all events in the queue are processed.
    # Make it some number to halt simulation after sometime.
    env.run(until=STOP_TIME)

    # Plot the output
    plt.plot(data['x'], data['z'])
    plt.show()


if __name__ == '__main__':
    main()
