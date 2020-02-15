#!/usr/bin/env python3

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress


def main(file=None):
    toplot = np.loadtxt(file, delimiter=',', skiprows=1)
    xs = np.log(np.sort(toplot[:, 0]))
    ys = np.log(toplot[:, 1])
    plt.plot(xs, ys, marker='*')
    # idealy = np.sqrt(toplot[:, 0])
    # idealy = 0.5 * np.log(toplot[:, 0])
    # plt.plot(xs, idealy, marker='1')
    # plt.title(file)
    # plt.show()
    print(file, ':', linregress(xs, ys)[0])


if __name__ == '__main__':
    import os
    for filename in os.listdir('/tmp/results/'):
        main('/tmp/results/'+filename)
