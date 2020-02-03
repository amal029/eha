#!/usr/bin/env python3
import matplotlib.pyplot as plt
import numpy as N


def main(dt=1e-1, R=4, T=1):
    """Example linear stochastic differential equation:
    λX(t) dt + μX(t)dW(t)
    """
    # The number of points in weiner process we need
    wN = int(N.ceil(T/dt))
    # First get the random numbers needed to make the weiner process.
    dWt = N.sqrt(dt) * N.random.randn(wN-1)
    W = N.cumsum(dWt)
    W = N.insert(W, 0, 0)

    ld = 2
    mu = 1

    # Euler-Maruyama solution
    # Xⱼ = Xⱼ₋₁ + Δt ⋆ (λ ⋆ Xⱼ₋₁) + μ ⋆ Xⱼ₋₁ ⋆ (Wⱼ - Wⱼ₋₁)
    # Wⱼ = W[Δt⋆j⋆R]
    # Δt = R⋆dt, this is done to make life easy for ourselves.

    Dt = R * dt
    X = 1                       # Initial value X(0)
    Xm = 1
    vso = [X]
    vsm = [Xm]
    vdt = [X]
    tso = [0]
    Xd = 1
    vs = [Xd]

    for j in range(0, int(wN/R)):
        part_sum = sum(dWt[(j*R):((j+1)*R)])
        # EM
        X = X + (Dt * (ld * X)) + (mu * X * part_sum)
        vso.append(X)
        tso.append(dt*R*j)

        # This is with a large step already, using W(ⱼ - Wⱼ₋₁) = sqrt(Δ
        # t) N(0,1)
        vdt.append(vdt[-1] + (Dt * ld * X) +
                   (mu * X * N.sqrt(Dt) * N.random.rand()))

        # Milstein's method, with partial derivative.
        Xm = (Xm + (Dt * (ld * X)) + (mu * Xm * part_sum)
              + 0.5*mu**2*Xm*((part_sum**2) - Dt))
        vsm.append(Xm)

        # Deterministic
        Xd = Xd + Dt * Xd * ld
        vs.append(Xd)

    plot_dict = dict()
    plot_dict['t'] = tso
    plot_dict['vsm'] = vsm
    plot_dict['vdt'] = vdt

    # This is the real closed form solution
    Xtruet = N.arange(0, T, dt)
    Xtrue = N.exp((ld-0.5*mu**2)*(Xtruet+mu*W))
    plt.plot(Xtruet, Xtrue)
    plt.plot(tso, vso, marker='1')
    plt.plot(plot_dict['t'], plot_dict['vsm'], marker='2')
    plt.plot(plot_dict['t'], plot_dict['vdt'], marker='*')
    plt.plot(tso, vs, marker='3')
    plt.show()


if __name__ == '__main__':
    # N.random.seed(100)            # Remove this later on
    for i in range(10):
        main(dt=(1/2**8), R=4, T=2)
