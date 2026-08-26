import numpy as np
import matplotlib.pyplot as plt

from pyacpr import *

def f(x):
    return np.sin(50*x)*(x - 0.5)**4*(x - 0.25)**2*x**x

def g(r, relative_energy=1.0, p=1.0, alpha=2.5, D=2.5, r0=2.5, n=2):
    term_1 = (r*alpha)**2 - (p*alpha)**2
    term_2 = (r*alpha)**2*D/relative_energy*(np.exp(-2.*alpha*(r - r0)) - 2.*np.exp(-alpha*(r - r0)))
    scale = 1./(1. + r*alpha)**n
    return term_1*scale - term_2*scale

def transform(x, L=2.0):
    return L/np.tan(x*np.pi/2.)**2

def h(x):
    return np.tanh((x - 0.5)/0.0001)

def j(x):
    return x*x - 1e-9

def q(x):
    return 0.

def r(x):
    return (x-0.5)/(x + 1e-3)**2

def s(x):
    return (x - 0.5)**2 + 1e-6

num_x_values = 1000
x_values = np.linspace(0.0, 1.0, num_x_values)
for test_function in [f, h, j, q, r, s]:
    
    plt.figure()
    plt.plot(x_values, [test_function(x_) for x_ in x_values])

    epsilon = 1e-5
    interval_limit = 1e-10
    nmax = 2049
    n0 = 2
    a = 0
    b = 1
    config = {
        'epsilon': epsilon,
        'delta': 1e-9,
        'interval_limit': interval_limit,
        'N_max': nmax
    }

    intervals, coefficients = chebyshev_subdivide_py(
        test_function, a, b, epsilon, n0, nmax, interval_limit
    )

    for interval, coeffs in zip(intervals, coefficients):
        domain = np.linspace(interval[0], interval[1], num_x_values)
        evaluated = [chebyshev_approximate_py(coeffs, interval[0], interval[1], x) for x in domain]
        actual = [test_function(x) for x in domain]
        plt.plot(domain, evaluated, linestyle='--')
        plt.gca().text(interval[0] + (interval[1] - interval[0])/3., np.mean(np.abs(evaluated))*1.5, f'N={len(coeffs)-1}')

    roots = find_roots_py(test_function, a, b, config)
    for root in roots:
        print(f'f({root}) = {test_function(root)}')
    print()
    plt.scatter(roots, np.zeros_like(roots), marker='*')
plt.show()
