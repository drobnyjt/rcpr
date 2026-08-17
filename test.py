import numpy as np
import matplotlib.pyplot as plt

from pyacpr import *

def f(x):
    return np.log(x + 1)*np.sin(x**2*25*np.log(1 + x))*(x - 0.1)**2*(x - 0.2)**3*(x - 0.8)**4*(x - 0.9)/(1+x)**(2+3+4-1)*np.exp(x)

x_values = np.linspace(0, 1, 1000)
plt.figure(1)
plt.plot(x_values, f(x_values))


a, e = chebyshev_coefficients_py(f, 0, 1, n_max=128)

f_c = [chebyshev_approximate_py(a, 0, 1, x) for x in x_values]
#plt.plot(x_values, f_c, linestyle='--')

epsilon = 1e-9
interval_limit = 0.01
intervals, coefficients = chebyshev_subdivide_py(
    f, 0, 1, epsilon, 2, 250, interval_limit
)

for interval, coeffs in zip(intervals, coefficients):
    domain = np.linspace(interval[0], interval[1], 1000)
    evaluated = [chebyshev_approximate_py(coeffs, interval[0], interval[1], x) for x in domain]
    actual = [f(x) for x in domain]
    plt.figure(1)
    plt.plot(domain, evaluated, linestyle='--')
    plt.gca().text(interval[0] + (interval[1] - interval[0])/3., np.mean(np.abs(evaluated))*1.5, f'N={len(coeffs)-1}')

    plt.figure(2)
    plt.semilogy(domain, abs(np.array(actual) - np.array(evaluated))/np.array(actual))
    np.testing.assert_allclose(actual, evaluated, atol=2*epsilon)

plt.figure(1)

roots = find_roots_py(f, 0, 1, epsilon=1e-9)
plt.scatter(roots, np.zeros_like(roots), marker='*')

plt.show()