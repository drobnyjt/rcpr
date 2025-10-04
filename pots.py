import numpy as np
import matplotlib.pyplot as plt
from sympy.parsing.mathematica import parse_mathematica
from sympy.parsing.latex import parse_latex
from numba import jit

def mie(r, n, m, sigma=0.1, epsilon=2.5):
    C = n/(n - m)*(n/m)**(m/(n-m))
    return C*epsilon*((sigma/r)**n - (sigma/r)**m)

def roots(Er, p, sigma=0.1, epsilon=2.5):

    E0 = epsilon/Er
    s = p/sigma

    u1 = \
        s**2./4 - np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)/2 - \
        np.emath.sqrt(s**4./2 - (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) - \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9 - \
        ((-2048*E0)/27 + s**6.)/ \
        (4*np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)))/2 \

    u2 = \
        s**2./4 - np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)/2 + \
        np.emath.sqrt(s**4./2 - (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) - \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9 - \
        ((-2048*E0)/27 + s**6.)/ \
        (4*np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)))/2 \

    u3 = \
        s**2./4 + np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)/2 - \
        np.emath.sqrt(s**4./2 - (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) - \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9 + \
        ((-2048*E0)/27 + s**6.)/ \
        (4*np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)))/2 \

    u4 = \
        s**2./4 + np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)/2 + \
        np.emath.sqrt(s**4./2 - (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) - \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9 + \
        ((-2048*E0)/27 + s**6.)/ \
        (4*np.emath.sqrt(s**4./4 + (32*2**(2/3)*E0*(-4 + s**2.))/(9*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3)) + \
        (4*2**(1/3)*(256*E0**2. - 27*E0*s**4. + np.emath.sqrt(E0**2.*(65536*E0**2. + 729*s**8. - 512*E0*(-128 + 96*s**2. + 3*s**4. + 2*s**6.))))**(1/3))/9)))/2 \

    r1 = sigma*np.emath.sqrt(u1)
    r2 = sigma*np.emath.sqrt(u2)
    r3 = sigma*np.emath.sqrt(u3)
    r4 = sigma*np.emath.sqrt(u4)

    return np.real_if_close(r1, tol=1e-6), np.real_if_close(r2, tol=1e-6), np.real_if_close(r3, tol=1e-6), np.real_if_close(r4, tol=1e-6)

num_p = 64
num_Er = 64
Er = np.logspace(-6, 0, num_Er)
p = np.linspace(0.0, 4.0, num_p)


r1 = np.zeros(num_p)
r2 = np.zeros(num_p)
r3 = np.zeros(num_p)
r4 = np.zeros(num_p)

doca = np.zeros((num_p, num_Er))

for j, Er_ in enumerate(Er):
    for i, p_ in enumerate(p):
        roots_list = np.array(roots(Er_, p_))
        r1[i] = roots_list[0]
        r2[i] = roots_list[1]
        r3[i] = roots_list[2]
        r4[i] = roots_list[3]
        try:
            doca[i, j] = np.real(np.max(roots_list[np.logical_and(np.logical_not(np.isnan(roots_list)), np.abs(np.imag(roots_list)) <= 1e-3)]))
        except ValueError:
            doca[i, j] = None
    color = plt.plot(p, np.real(r1))[0].get_color()
    plt.plot(p, np.real(r2), color=color)
    plt.plot(p, np.real(r3), color=color)
    plt.plot(p, np.real(r4), color=color)

plt.figure()
plt.pcolormesh(p, Er, doca.transpose())
plt.gca().set_yscale('log')
plt.colorbar()
plt.show()
breakpoint()
exit()
print(roots(0.01, 1.1))

r = np.linspace(0.0, 1.0, 10000)
epsilon = 1.0
m = 6
n = 8
if n != m:
    plt.semilogy(r, mie(r, n, m, epsilon = 1.0) + epsilon*1.1, label=f'{n},{m}')
plt.legend()
plt.show()