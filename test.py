from rcpr_py import  find_all_roots
import numpy as np
import matplotlib.pyplot as plt
from numba import jit
import time
from matplotlib.colors import SymLogNorm, LogNorm
from scipy.interpolate import RectBivariateSpline
from scipy.optimize import newton

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
def piecewise(r, r0=2.0e-10, E0=2.5*Q, V0=100*Q):
    if r < r0:
        return V0 - (E0 + V0)/r0 * r
    elif r0 <= r < 2*r0:
        return E0/r0*r - 2*E0
    else:
        return 0.0

@jit
def inverse_quadratic(r, b=0.996055, c=0.793701):
    if r == 0:
        return 1e9
    else:
        return 16.0*(b/(r/ANGSTROM)**2 - c/(r/ANGSTROM))*Q
    
@jit
def lj(r, epsilon=2.5*Q, sigma=2.5*ANGSTROM):
    return 4*epsilon*((sigma/r)**12 - (sigma/r)**6)

@jit(nopython=True)
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

@jit(nopython=True)
def krc_morse(r, D=5.4971e-20, alpha=1.4198e10, r0=2.782e-10, k=7e10, x0=0.75e-10):
    return  smootherstep(r, k, x0)*morse(r, D, alpha, r0) + smootherstep(r, -k, x0)*screened_coulomb(r, a=lindhard_screening_length(1, 28), z1=1, z2=28)

@jit(nopython=True)
def morse(r, D=5.4971e-20, alpha=1.4198e10, r0=2.782e-10):
    return  D*(np.exp(-2.*alpha*(r - r0)) - 2.*np.exp(-alpha*(r - r0)))

@jit(nopython=True)
def krc(x):
    '''
        Krypton-Carbon "Universal" Screening Function

        Args:
            x: normalized distance r/a

        Returns:
            phi(x)
    '''
    return 0.190945*np.exp(-0.278544*x) + 0.473674*np.exp(-0.637174*x) + 0.335381*np.exp(-1.919249*x)

@jit(nopython=True)
def lindhard_screening_length(z1, z2):
    '''
        Lindhard screening length for interatomic potentials

        Args:
            z1, z2: atomic numbers

        Returns:
            a: screening length in meters
    '''
    return 0.8553*A0*(np.sqrt(z1) + np.sqrt(z2))**(-2./3.)

@jit(nopython=True)
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

@jit(nopython=True)
def distance_of_closest_approach_function(r, V, p, Er, n=1):
    return ((r/ANGSTROM)**2*(1. - (V(r)/Q)/(Er/Q)) - (p/ANGSTROM)**2)*((r/ANGSTROM)**n + 1.0)

@jit(nopython=True)
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

def distance_of_closest_approach(V, p, Er, use_newton=False):
    a = 0.0
    b = 10.0
    N0 = 3
    N_max = 1000
    epsilon = 0.001
    interval_limit = 1e-24
    far_from_zero = 1e9
    complex_threshold = 1e-6
    truncation_threshold = 1e-6

    f = lambda s: distance_of_closest_approach_function(s*ANGSTROM, V, p, Er)

    if use_newton:
        root = newton(f, 10.0, x1=7.0, maxiter=10000, tol=1e-4, disp=False)
        return root*ANGSTROM
    else:
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

def gauss_legendre_5(V, p, Er, use_newton=False):
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
    r0 = distance_of_closest_approach(V, p, Er, use_newton)

    #Quadrature
    x = np.array([0., -0.538469, 0.538469, -0.90618, 0.90618])
    w = np.array([0.568889, 0.478629, 0.478629, 0.236927, 0.236927])

    #transform w, x to [-1, 1] from [0, 1]
    w /= 2.
    x = x/2 + 1/2
    try:
        theta = np.pi - np.sum([w_*G(x_, V, p, Er, r0) for w_, x_ in zip(w, x)])
        return r0, theta
    except ZeroDivisionError:
        return r0, np.nan

def run_simulation(
    Za=1,
    Zb=28,
    Ma=1.008,
    Mb=58.6934,
    n=0.0914,
    num_energies=32,
    num_p=32,
    min_E=-4,
    max_E=-2,
    potential=morse,
    use_newton=False
    ):

    mu = Mb/(Ma + Mb)
    a = lindhard_screening_length(Za, Zb)

    min_p = 0.0
    max_p = np.pi*(n)**(-1./3.)
    max_p = 6.0

    reduced_energies = np.logspace(min_E, max_E, num_energies)
    energies = reduced_energies/(LINDHARD_REDUCED_ENERGY_PREFACTOR*a*mu/Za/Zb)
    relative_energies = energies*Mb/(Ma + Mb)

    p = np.linspace(min_p, max_p, num_p)*1e-10

    theta = np.zeros((num_p, num_energies))
    doca = np.zeros((num_p, num_energies))

    for i, Er_ in enumerate(relative_energies):
        print(f'{np.round(i/len(relative_energies)*100., 1)}%')
        for j, p_ in enumerate(p):
            doca[j, i], theta[j, i] = gauss_legendre_5(potential, p_, Er_, use_newton=use_newton)

    return theta, doca, relative_energies, p

run_calc = True
use_newton = False
n = 0.0914
pmax = np.pi*(n)**(-1./3.)
potential = morse
name='Morse (ACPR)'

if run_calc:
    start = time.time()
    theta, doca, relative_energies, p = run_simulation(n=n, potential=potential, use_newton=use_newton)
    stop = time.time()
    print(f'Run took {stop - start}s')
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
plt.pcolormesh(
    p/ANGSTROM,
    relative_energies/Q,
    np.sin(theta.transpose()/2.)**2,
    )
plt.xlabel('p [A]')
plt.ylabel('Er [eV]')
plt.gca().set_yscale('log')
plt.title('Energy Transfer')
plt.colorbar(label='T [eV]')
plt.savefig(f'{name}_delta_E.png')

j = 1
k = 1
plt.figure(2)
plt.pcolormesh(
    (p/ANGSTROM)**j,
    (relative_energies/Q)**k,
    doca.transpose()/ANGSTROM,
    #vmin=0.0,
    #vmax=10.0
    )
plt.xlabel('p [A]')
plt.ylabel('Er [eV]')
plt.gca().set_yscale('log')
plt.title('DOCA')
plt.colorbar(label='DOCA [A]')
plt.savefig(f'{name}_doca.png')

plt.figure(3)
plt.pcolormesh(
    p/ANGSTROM,
    relative_energies/Q,
    np.cos(theta.transpose()),
    vmin=-1.0,
    vmax=1.0
)
plt.xlabel('p [A]')
plt.ylabel('Er [eV]')
plt.gca().set_yscale('log')
plt.title('Scattering Angle')
plt.colorbar(label='cos(theta)')
plt.savefig(f'{name}_theta.png')

theta[np.isnan(theta)] = np.pi

theta_interpolate = RectBivariateSpline(p/ANGSTROM, relative_energies/Q, theta)
energy_loss_interpolate = RectBivariateSpline(p/ANGSTROM, relative_energies/Q, np.sin(theta/2.)**2)

plt.show()