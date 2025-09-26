from rcpr_py import  find_all_roots
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time

PI = np.pi
Q = 1.602E-19
EV = Q
AMU = 1.66E-27
ANGSTROM = 1E-10
MICRON = 1E-6
PM = 1E-12
NM = 1E-9
CM = 1E-2
EPS0 = 8.85E-12
A0 = 0.52918E-10
K = 1.11265E-10
ME = 9.11E-31
SQRTPI = 1.772453850906
SQRT2PI = 2.506628274631
C = 299792000.
BETHE_BLOCH_PREFACTOR = 4.*PI*(Q*Q/(4.*PI*EPS0))*(Q*Q/(4.*PI*EPS0))/ME/C/C
LINDHARD_SCHARFF_PREFACTOR = 1.212*ANGSTROM*ANGSTROM*Q
LINDHARD_REDUCED_ENERGY_PREFACTOR = 4.*PI*EPS0/Q/Q
#[[{{"MORSE"={{D=5.4971E-20, r0=2.782E-10, alpha=1.4198E10}}}}]]
#[[{{"KRC_MORSE"={{D=5.4971E-20, r0=2.782E-10, alpha=1.4198E10, k=7E10, x0=0.75E-10}}}}]]

@jit
def smootherstep(x, k, x0):
    x_transformed = x*k/8.
    x0_transformed = x0*k/8.

    x_transformed = x*k/8.
    x0_transformed = x0*k/8.
    if x_transformed <= x0_transformed - 0.5:
        return 0.
    elif x_transformed >= 0.5 + x0_transformed:
        return 1.
    else:
        x1  = x_transformed - x0_transformed + 0.5
        return x1 * x1 * x1 * (x1 * (6. * x1 - 15.) + 10.)

@jit
def krc_morse(r, D=5.4971e-20, alpha=1.4198e10, r0=2.782e-10, k=7e10, x0=0.75e-10):
    return  smootherstep(r, k, x0)*morse(r, D, alpha, r0) + smootherstep(r, -k, x0)*screened_coulomb(r, a=lindhard_screening_length(1, 28), z1=1, z2=28)

@jit
def morse(r, D, alpha, r0):
    return  D*(np.exp(-2.*alpha*(r - r0)) - 2.*np.exp(-alpha*(r - r0)))

@jit
def krc(x):
    '''
        Krypton-Carbon "Universal" Screening Function

        Args:
            x: normalized distance r/a

        Returns:
            phi(x)
    '''
    return 0.190945*np.exp(-0.278544*x) + 0.473674*np.exp(-0.637174*x) + 0.335381*np.exp(-1.919249*x)

@jit
def lindhard_screening_length(z1, z2):
    '''
        Lindhard screening length for interatomic potentials

        Args:
            z1, z2: atomic numbers

        Returns:
            a: screening length in meters
    '''
    return 0.8553*A0*(np.sqrt(z1) + np.sqrt(z2))**(-2./3.)

@jit
def screened_coulomb(r, a=lindhard_screening_length(1, 1), phi=krc, z1=1, z2=1):
    '''
        Screened coulomb interatomic potential for z1, z2 using screening function phi

        Args:
            r: distance in meters
            a: screeneing length in meters
            phi: screening function
            z1, z2: atomic numbers

        Returns:
            V(r) [J]
    '''
    return (1./4./np.pi/EPS0)*z1*z2*Q**2/r*phi(r/a)

@jit
def distance_of_closest_approach_function(r, V, p, Er):
    value = ((r/ANGSTROM)**2*(1. - V(r)/Er) - (p/ANGSTROM)**2)*((r/ANGSTROM)**2 + 1.0)
    return value

@jit
def G(u, V, p, Er, r0):
    '''
        Scattering integrand using r = r0/(1 - u**2)

        Args:
            V: interaction potential
            p: impact parameter
            Er: relative energy
            r0: distance of closest approach

        Returns:
            Scattering integrand for r=r0/(1 - u**2) and bounds u: [0, 1]
    '''
    return 4*p*u/(r0*np.sqrt(1-V(r0/(1 - u**2))/Er-p**2*(-u**2+1)**2/r0**2))

def distance_of_closest_approach(V, p, Er):
    a = 1.0
    b = 10.0
    N0 = 4
    N_max = 1000
    epsilon = 0.01
    interval_limit = 1e-20
    far_from_zero = 1e24
    complex_threshold = 1e-9
    truncation_threshold = 1e-20

    f = lambda s: distance_of_closest_approach_function(s*ANGSTROM, V, p, Er)
    
    roots = find_all_roots(
        f,
        a,
        b,
        N0,
        epsilon,
        N_max,
        complex_threshold,
        truncation_threshold,
        interval_limit,
        far_from_zero
    )

    if np.size(roots) > 0:
        return np.max(roots)*ANGSTROM
    else:
        return 0.0

def gauss_legendre_5(V, p, Er):
    '''
        5th order Gauss-Legendre quadrature
        I derived this one myself from the Wikipedia page on Gauss-Legendre

        Args:
            V: interaction potential [J]
            p: impact parameter [m]
            Er: relative energy [J]

        Returns:
            theta: scattering angle in the CoM frame [radians]
    '''

    #Find distance of closest approach
    r0 = distance_of_closest_approach(V, p, Er)

    #Quadrature
    x = np.array([0., -0.538469, 0.538469, -0.90618, 0.90618])
    w = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927])

    #transform w, x to [-1, 1] from [0, 1]
    w /= 2.
    x = x/2 + 1/2

    theta = np.pi - np.sum([w_*G(x_, V, p, Er, r0) for w_, x_ in zip(w, x)])
    return theta

def run_simulation(mask=None):
    Za = 1
    Zb = 28
    Ma = 1.008
    Mb = 58.6934
    mu = Mb/(Ma + Mb)
    a = lindhard_screening_length(Za, Zb)
    num_energies = 256
    num_p = 256
    min_p = 6.0
    max_p = 8.0
    reduced_energies = np.logspace(-3, -6, num_energies)
    energies = reduced_energies/(LINDHARD_REDUCED_ENERGY_PREFACTOR*a*mu/Za/Zb)
    relative_energies = energies*Mb/(Ma + Mb)


    p = np.linspace(min_p, max_p, num_p)*1e-10

    theta = np.zeros((num_p, num_energies))
    doca = np.zeros((num_p, num_energies))

    if mask is None:
        for i, Er_ in enumerate(relative_energies):
            print(f'{np.round(i/len(relative_energies)*100., 1)}%')
            for j, p_ in enumerate(p):
                theta[j, i] = gauss_legendre_5(krc_morse, p_, Er_)
                doca[j, i] = distance_of_closest_approach(krc_morse, p_, Er_)
    else:
        for i, Er_ in enumerate(relative_energies):
            print(f'{np.round(i/len(relative_energies)*100., 1)}%')
            for j, p_ in enumerate(p):
                if mask[j, i]:
                    theta[j, i] = gauss_legendre_5(krc_morse, p_, Er_)
                    doca[j, i] = distance_of_closest_approach(krc_morse, p_, Er_)
    
    return theta, doca, relative_energies, p

run_calc = True

if run_calc:
    start = time.time()
    theta, doca, relative_energies, p = run_simulation()
    stop = time.time()
    print(stop - start)
    np.savetxt('theta.txt', theta)
    np.savetxt('doca.txt', doca)
    np.savetxt('Er.txt', relative_energies)
    np.savetxt('p.txt', p)
else:
    theta = np.genfromtxt('theta.txt')
    doca = np.genfromtxt('doca.txt')
    relative_energies = np.genfromtxt('Er.txt')
    p = np.genfromtxt('p.txt')

plt.figure(1)
plt.pcolormesh(p/ANGSTROM, relative_energies/Q, np.sin(theta.transpose()/2.)**2, vmin=0., vmax=1.)
plt.gca().set_yscale('log')
plt.title('Energy Transfer')
plt.colorbar()

plt.figure(2)
plt.pcolormesh(p/ANGSTROM, relative_energies/Q, doca.transpose()/ANGSTROM)
plt.gca().set_yscale('log')
plt.title('DOCA')
plt.colorbar()

plt.figure(3)
plt.pcolormesh(p/ANGSTROM, relative_energies/Q, theta.transpose(), vmin=-2*np.pi, vmax=np.pi)
plt.gca().set_yscale('log')
plt.title('Scattering Angle')
plt.colorbar()

plt.show()
exit()