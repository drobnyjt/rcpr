import numpy as np
import matplotlib.pyplot as plt

from pyacpr import *

def w(x, n=40):
    out = 1.
    for i in range(1, n + 1):
        out *= (x - i)
    return out

num_x_values = 10000
a = 0.0
b = 40
x_values = np.linspace(a, b, num_x_values)

plt.plot(x_values,[w(x_) for x_ in x_values])

epsilon = 1e-9
interval_limit = 1e-9
N_max = 512

config = {
    'epsilon': epsilon,
    'interval_limit': interval_limit,
    'N_max': N_max,
    'error_calc': 'Relative'
}

intervals, coefficients = chebyshev_subdivide_py(
    w, a, b, epsilon, 2, N_max, interval_limit
)

for interval, coeffs in zip(intervals, coefficients):
        domain = np.linspace(interval[0], interval[1], num_x_values)
        evaluated = [chebyshev_approximate_py(coeffs, interval[0], interval[1], x) for x in domain]
        plt.plot(domain, evaluated, linestyle='--')
        plt.gca().text(interval[0] + (interval[1] - interval[0])/3., np.mean(np.abs(evaluated))*1.5, f'N={len(coeffs)-1}')

roots = find_roots_py(w, a, b, config)
print(len(roots), roots)
plt.scatter(roots, np.zeros_like(roots), marker='*')
plt.show()