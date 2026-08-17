import numpy as np
import matplotlib.pyplot as plt

from pyacpr import *

def f(x):
    return np.sin(x*500)*(x - 0.1)**2*(x - 0.2)**3*(x - 0.8)**4*(x - 0.9)/(1+x)**(2+3+4-1)*1e5

x_values = np.linspace(0, 1, 1000)
plt.plot(x_values, f(x_values))


a, e = chebyshev_coefficients_py(f, 0, 1, n_max=128)

f_c = [chebyshev_approximate_py(a, 0, 1, x) for x in x_values]
#plt.plot(x_values, f_c, linestyle='--')

intervals, coefficients = chebyshev_subdivide_py(
    f, 0, 1, 1e-6, 2, 100, 1e-3
)

for interval, coeffs in zip(intervals, coefficients):
    domain = np.linspace(interval[0], interval[1], 100)
    evaluated = [chebyshev_approximate_py(coeffs, interval[0], interval[1], x) for x in domain]
    plt.plot(domain, evaluated, linestyle='--')

plt.show()
breakpoint()