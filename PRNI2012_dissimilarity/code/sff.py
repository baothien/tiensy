import numpy as np
import matplotlib.pyplot as plt

def m(p, c=2.0):
    return c * p * np.log(p)

def bound(p, c=2.0):
    return p * np.exp(-m(p,c)/p)

if __name__ == '__main__':
    
    c = 3.0

    p = np.arange(1, 100)

    plt.ion()
    # plt.figure()
    plt.plot(p, bound(p,c))

