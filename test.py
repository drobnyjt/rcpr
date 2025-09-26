from rcpr_py import  find_all_roots
import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x*x*np.sin(x*x*10) + x*x*x - x*x + 0.001

a = -0.1
b = 0.1
x = np.linspace(a, b, 10000)
plt.plot(x, f(x))

print(f(0.0))

roots = find_all_roots(f, a, b, 4, 1e-12, 100, 1e-3, 0.0, 1e-12, 1e9)

plt.scatter(roots, [f(root) for root in roots])

plt.show()