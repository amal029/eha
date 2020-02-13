#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np


def main(file=None):
    toplot = np.loadtxt(file, delimiter=',', skiprows=1)
    xs = np.log(toplot[:, 0])
    plt.plot(xs, np.log(toplot[:, 1]), marker='*')
    # idealy = np.sqrt(toplot[:, 0])
    idealy = 0.5 * np.log(toplot[:, 0])
    plt.plot(xs, idealy)
    plt.show()


if __name__ == '__main__':
    main('/tmp/results/runexample4inward_5.csv')
    main('/tmp/results/runexample4inward_10.csv')
