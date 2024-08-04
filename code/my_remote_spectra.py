# Modify SZ_cosmo https://github.com/rcayuso/SZ_cosmo/tree/master
from __future__ import print_function
from __future__ import absolute_import

import numpy as np
from scipy import special
import scipy.integrate as integrate
import my_kszpsz_config as conf
from numpy.linalg import inv
from scipy.interpolate import interp1d, interp2d
import scipy.optimize as optimize

######################################################################################################
################   HUBBLE PARAMETER, DENSITIES, COMOVING DISTANCE , ETC     ###################
######################################################################################################


# Auxiliary scale factor grid
ag = np.logspace(np.log10(conf.adec), 0, conf.transfer_integrand_sampling)

def az(z):
    """Scale factor at a given redshift"""
    return 1.0 / (1.0 + z)


def aeq(Omega_b, Omega_c, h, Omega_r_h2=conf.Omega_r_h2):
    """Scale factor at matter radiation equality"""
    return Omega_r_h2 / (Omega_b + Omega_c) / h**2


def k_sampling(config=conf):
    return np.logspace(config.k_min, config.k_max, config.k_res)

def L_sampling(config=conf):
    return np.arange(config.ksz_estim_signal_lmax)


# Modification to dark energy equation of state (only one model for now)
def fde(a):
    return 1.0 - a


def fde_da(a):
    return -1.0


def H0(h):
    """Hubble parameter today in Mpc**-1"""
    return 100.0 * h / (3.0 * 1.0e5)


def Omega_L(Omega_b, Omega_c, Omega_K, h, Omega_r_h2=conf.Omega_r_h2):
    """Omega_L in terms of Omega_b, Omega_c, and K imposed by consistency"""
    return 1 - Omega_K - (Omega_b + Omega_c) - Omega_r_h2/h**2


def E(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=conf.Omega_r_h2):
    """Reduced Hubble parameter, H/H_0"""
    exp_DE = 3.0*(1.0 + w + wa*fde(a))
    E2 = (Omega_b + Omega_c) / a**3 + Omega_K / a**2 \
        + Omega_L(Omega_b, Omega_c, Omega_K, h) / a**exp_DE \
        + Omega_r_h2/h**2  / a**4
    return np.sqrt(E2)


def H(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Hubble parameter given a & cosmological parameters"""
    Ea = E(a, Omega_b, Omega_c, w, wa, Omega_K, h)
    return Ea * H0(h)


def H_config(a, config):
    """Hubble parameter given a & config settings"""
    return H(a, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def dEda(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=conf.Omega_r_h2) :
    """
    Derivative of the reduced Hubble parameter respect to the scale factor
    """
    exp_DE = 3.0*(1.0 + w + wa*fde(a))
    d = -3.0*(Omega_b + Omega_c) / a**4 - 2.0*Omega_K / a**3 \
        + (-3.0*wa*a*np.log(a) * fde_da(a) - exp_DE) * Omega_L(Omega_b, Omega_c, Omega_K, h) / a**(exp_DE + 1) \
        - 4.0*Omega_r_h2/h**2 / a**5
    derv_E = d / (2.0 * E(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=Omega_r_h2))
    return derv_E


def dHdt(a, Omega_b, Omega_c, w, wa, Omega_K, h) :
    """(non-conformal) time-derivative of hubble parameter"""
    return a*H(a, Omega_b, Omega_c, w, wa, Omega_K, h)*H0(h) \
        * dEda(a, Omega_b, Omega_c, w, wa, Omega_K, h)


def dHdt_config(a, config) :
    """(non-conformal) time-derivative of hubble parameter"""
    return dHdt(a, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def sigma_nez(z, Omega_b, h):
    """Electron density as a function of redshift times the Thompson cross section"""
    mProton_SI = 1.673e-27
    G_SI = 6.674e-11
    H0 = 100 * h * 1e3  # Hubble constant in m s^-1 Mpc^-1
    MegaparsecTometer = 3.086e22
    thompson_SI = 6.6524e-29

    sigma_ne = thompson_SI * (3 * 0.88 / 8. / np.pi / MegaparsecTometer) * \
        (H0**2) * (1. / mProton_SI / G_SI) * Omega_b * (1 + z)**3

    return sigma_ne


def tau_z(z, Omega_b, h):
    """Optical depth at a given redshift"""
    chi = chifromz(z)
    chi_grid = np.linspace(0, chi, 100)
    z_grid = zfromchi(chi_grid)
    ae = az(z_grid)
    sigma_ne = sigma_nez(z_grid, Omega_b, h)

    integrand = ae * sigma_ne

    tau = integrate.simps(integrand, chi_grid)

    return tau


def tau_grid(Chi_grid, Z_grid, Omega_b, h):
    """
    Optical depth as a function of redshift
    Assumes Z_grid starts at z = 0.0!
    """
    ae = az(Z_grid)
    sigma_ne = sigma_nez(Z_grid, Omega_b, h)
    integrand = ae * sigma_ne
    tau_grid = integrate.cumtrapz(integrand, Chi_grid, initial=0.0)

    return tau_grid


def z_re(Omega_b, h, tau):
    """ redshift of recombination (?) """
    zguess = 6.0
    sol = optimize.root(root_tau2z, zguess, args=(Omega_b, h, tau))
    z_re = sol.x
    return z_re


def spherical_jn_pp( l, z ):
    """
    Second derivative of spherical Bessel function.
    """
    if l == 0:
        jn = special.spherical_jn(2, z) - special.spherical_jn(1, z)/z
    else:
        jn = (special.spherical_jn(l-1,z, True)
              - (l+1)/z * special.spherical_jn(l,z, True)
              + (l+1)/(z**2) * special.spherical_jn(l,z))
    return jn



##########################################################################
################   COMOVING DISTANCES AND THEIR DERIVATIVES     ##########
##########################################################################

def Integrand_chi(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integrand of the comoving distance defined below (chia, small c)"""
    int_chi = 1 / ((a**2) * H(a, Omega_b, Omega_c, w, wa, Omega_K, h))
    return int_chi


def chia(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Comoving distance to a (using integrate.quad)"""
    chi = integrate.quad(Integrand_chi, a, 1, args=(
        Omega_b, Omega_c, w, wa, Omega_K, h, ))[0]
    return chi


def Integrand_Chi(y, a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """ Function to integrate to find chi(a) defined below (Chia, capital C) """
    g = -((a**2) * H(a, Omega_b, Omega_c, w, wa, Omega_K, h))**-1
    return g

def Chia(Omega_b, Omega_c, w, wa, Omega_K, h):
    """Comoving distance to a (using integrate.odeint)"""
    Integral = (integrate.odeint(Integrand_Chi, chia(ag[0], Omega_b, Omega_c, w, wa, Omega_K, h),
                                 ag, args=(Omega_b, Omega_c, w, wa, Omega_K, h,))).reshape(len(ag),)

    if ag[-1] == 1.0:
        Integral[-1] = 0.0

    return Integral


def chifromz(z, config=conf):
    """Comoving distance at a redshift z"""
    chi = Chia_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)(az(z))
    return chi


def zfromchi(chi, config=conf):
    """Get redshift z given comoving distance chi"""
    afromchi = interp1d(
        Chia(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h), ag,
        kind="cubic", bounds_error=False, fill_value='extrapolate')

    aguess = afromchi(chi)

    sol = optimize.root(root_chi2a, aguess,
        args=(chi, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h))
    z = 1.0/sol.x - 1.0

    return z



#############################################################################
################                                          ###################
################          GROWTH FUNCTIONS                ###################
################                                          ###################
#############################################################################


def Integrand_GF(s, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Integrand for the growth function (as defined below)
    """
    Integrand = 1 / (s * E(s, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0))**3
    return Integrand


def Integral_GF(a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Integral appearing in the growth factor GF
    """
    Integral = integrate.quad(Integrand_GF, 0, a, args=(
        Omega_b, Omega_c, w, wa, Omega_K, h, ))[0]
    return Integral


def Integrand_GF_ODE(y, a, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Function for solving Integral_GF as an ODE (as defined below)
    """
    f = (a * E(a, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0))**-3
    return f*1.0e13 # 10^13 changes scale so ODE has enough precision


def Integral_GF_ODE(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Integral appearing in the growth function GF solved as an ODE
    """
    Integral = integrate.odeint(Integrand_GF_ODE,
                                1.0e13*Integral_GF(ag[0], Omega_b, Omega_c,
                                            w, wa, Omega_K, h),
                                ag, args=(Omega_b, Omega_c, w, wa, Omega_K, h,))

    return Integral * 1.0e-13 # 10^13 changes scale so ODE has enough precision


def GF(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Growth function using ODE, D_1(ag) / ag
    Dodelson Eq. 7.5, 7.77
    """
    l = 5.0/2.0 * (Omega_b + Omega_c) * \
        E(ag, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0) / ag
    GF = l * Integral_GF_ODE(Omega_b, Omega_c, w, wa,
                             Omega_K, h).reshape(len(ag),)
    return GF


def Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    ~ Potential "growth function" from linear theory plus approximations
    Phi = Phi_prim * T(k) * (9/10 * D_1(a)/a) per Dodelson Eq. 7.5
    Dpsi is ("9/10" * D_1(a)/a)
    Dodelson Eq. 7.5, Eq. 7.32
    """
    y = ag / aeq(Omega_b, Omega_c, h)
    fancy_9_10 = (16.0*np.sqrt(1.0 + y) + 9.0*y**3 + 2.0*y**2 - 8.0*y - 16.0) / (10.0*y**3)
    Dpsi = fancy_9_10 * GF(Omega_b, Omega_c, w, wa, Omega_K, h)
    return Dpsi


def derv_Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Derivative of the growth function with respect to the scale factor
    """
    y = ag / aeq(Omega_b, Omega_c, h)
    Eag = E(ag, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0)
    dEdag = dEda(ag, Omega_b, Omega_c, w, wa, Omega_K, h, Omega_r_h2=0.0)
    fancy_9_10 = (16.0*np.sqrt(1.0 + y) + 9.0*y**3 + 2.0*y**2 - 8.0*y - 16.0) / (10.0*y**3)
    P1 = ((8.0/np.sqrt(1.0 + y) + 27.0*y**2 + 4.0*y - 8.0) / (aeq(Omega_b,
      Omega_c, h)*10.0*y**3)) * GF(Omega_b, Omega_c, w, wa, Omega_K, h)
    P2 = Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h) * (-4/ag + dEdag/Eag)
    P3 = fancy_9_10 * (5.0/2.0) * (Omega_b + Omega_c) / (ag**4 * Eag**2)

    derv_Dpsi = P1 + P2 + P3
    return derv_Dpsi


def Dv(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Velocity growth function on superhorizon scales
    Dodelson 7.15 minus 7.16, v is 5.78
    v_i ~ - Dv d_i psi
    grad*v_i ~ k^2 * Dv * psi
    """
    y = ag / aeq(Omega_b, Omega_c, h)
    Dv = 2.0 * (ag**2) * H(ag, Omega_b, Omega_c, w, wa, Omega_K, h) \
                / ((Omega_b + Omega_c)*H0(h)**2) * y / (4.0 + 3.0*y) \
            * (Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h)
                + ag*derv_Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h))
    return Dv


def T(k, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    'Best fitting' transfer function From Eq. 7.70 in Dodelson
    Assumes: no baryons, nonlinear effects, phi=psi always (9/10 -> 0.86), normal (no?) DE effects
    """
    fac = np.exp(-Omega_b*(1+np.sqrt(2*h)/(Omega_b+Omega_c)))
    keq = aeq(Omega_b, Omega_c, h) * H(aeq(Omega_b, Omega_c, h),
                                       Omega_b, Omega_c, w, wa, Omega_K, h)
    x = k / keq / fac
    x[np.where(x<1.0e-10)] = 1
    T = (np.log(1 + 0.171 * x) / (0.171 * x)) * (1 + 0.284 * x
      + (1.18 * x)**2 + (0.399 * x)**3 + (0.49 * x)**4)**(-0.25)
    return T


def T_config(k, config):
    return T(k, config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


# TODO?
# Replace T*Dpsi (or other growth*transfer instances) with CAMB ones?
# Just to study the impact of BAO, other assumptions implicit in T(k)
# def TDpsi_CAMB(k, Omega_b, Omega_c, w, wa, Omega_K, h) :


def Ppsi(k, As, ns):
    """
    Power spectrum of primordial potential
    """
    P = (2.0/3.0)**2 * 2.0 * np.pi**2 / (k**3) \
        * As * 10**-9 * (k / conf.k0)**(ns - 1)
    return P


##########################################################################
################                                          ################
################ INTERPOLATING AND AUXILIARY FUNCTIONS    ################
################                                          ################
##########################################################################

def Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    Chia_inter = interp1d(ag, Chia(Omega_b, Omega_c, w, wa, Omega_K, h),
                          kind="cubic", bounds_error=False, fill_value='extrapolate')
    return Chia_inter

def Chia_inter_config(config):
    return Chia_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    Dpsi_inter = interp1d(ag, Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h),
                          kind="cubic", bounds_error=False, fill_value='extrapolate')
    return Dpsi_inter

def Dpsi_inter_config(config):
    return Dpsi_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    Returns interpolating function
    of derivative of the growth function with respect to the scale factor
    """
    derv_Dpsi_inter = interp1d(ag, derv_Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h),
                               kind="cubic", bounds_error=False, fill_value='extrapolate')
    return derv_Dpsi_inter

def derv_Dpsi_inter_config(config):
    return derv_Dpsi_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h):
    """Returns interpolating funcntion of velocity growth function"""
    Dv_inter = interp1d(ag, Dv(Omega_b, Omega_c, w, wa, Omega_K, h),
                        kind="cubic", bounds_error=False, fill_value='extrapolate')
    return Dv_inter

def Dv_inter_config(config):
    return Dv_inter(config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def root_chi2a(a, chis, Omega_b, Omega_c, w, wa, Omega_K, h):
    """ Needed to use the root function of scipy's optimize module. """
    return Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a) - chis


def root_tau2z(z, Omega_b, h, tau):
    """Needed to use the root function of scipy's optimize module below."""
    return tau_z(z, Omega_b, h) - tau



#######################################################################
################                                    ###################
################ KERNELS FOR SW, ISW AND DOPPLER    ###################
################                                    ###################
#######################################################################



def G_SW_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ Integral kernel for SW effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_SW_ksz = 3 * (2 * Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h)
                    [0] - 3 / 2) * special.spherical_jn(1, k * (chidec - chie))

    return G_SW_ksz


def G_SW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """pSZ Integral kernel for SW effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_SW_psz = -4 * np.pi * (2 * Dpsi(Omega_b, Omega_c, w, wa, Omega_K, h)
                             [0] - 3 / 2) * special.spherical_jn(2, k * (chidec - chie))

    return G_SW_psz


def G_SW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """CMB Integral kernel for SW effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    G_SW_CMB = 4 * np.pi * ((1j)**l) * (2 * Dpsi(Omega_b, Omega_c, w,
        wa, Omega_K, h)[0] - 3 / 2) * special.spherical_jn(l, k * chidec)

    return G_SW_CMB


def G_Dopp_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ Integral kernel for the Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_Dopp_ksz = k * Dv(Omega_b, Omega_c, w, wa, Omega_K, h)[0] * (special.spherical_jn(0, k * (
        chidec - chie)) - 2 * special.spherical_jn(2, k * (chidec - chie))) - k * Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    return G_Dopp_ksz


def G_localDopp_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ Integral kernel for the local Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_localDopp_ksz = -k * \
        Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    return G_localDopp_ksz


def G_Dopp_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """pSZ Integral kernel for the Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    G_Dopp_psz = (4 * np.pi / 5) * k * Dv(Omega_b, Omega_c, w, wa, Omega_K, h)[0] * (
        3 * special.spherical_jn(3, k * (chidec - chie)) - 2 * special.spherical_jn(1, k * (chidec - chie)))

    return G_Dopp_psz


def G_Dopp_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """CMB Integral kernel for the Doppler effect"""
    chidec = Chia(Omega_b, Omega_c, w, wa, Omega_K, h)[0]
    G_Dopp_CMB = (4 * np.pi / (2.0 * l + 1.0)) * (1j**l) * k * Dv(Omega_b, Omega_c, w, wa, Omega_K, h)[
        0] * (l * special.spherical_jn(l - 1, k * chidec) - (l + 1) * special.spherical_jn(l + 1, k * chidec))

    return G_Dopp_CMB


def G_ISW_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integral kernel for the ksz ISW effect"""
    a = np.logspace(np.log10(conf.adec), np.log10(az(ze)), conf.transfer_integrand_sampling)
    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    Deltachi = Chia - Chie
    Deltachi[-1] = 0
    s2 = k[..., np.newaxis] * Deltachi

    integrand = special.spherical_jn(1, s2) * derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    g_isw_ksz = 6.0*integrate.simps(integrand, a)

    return g_isw_ksz


def G_ISW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integral kernel for the psz ISW effect"""
    a = np.logspace(np.log10(conf.adec), np.log10(az(ze)), 1000)
    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))
    Deltachi = Chia - Chie
    Deltachi[-1] = 0

    s2 = k[..., np.newaxis] * Deltachi
    integrand = special.spherical_jn(
        2, s2) * derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    g_isw_psz = -8 * np.pi * integrate.simps(integrand, a)

    return g_isw_psz


def G_ISW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Integral kernel for the CMB ISW effect"""
    a = np.logspace(np.log10(conf.adec), np.log10(1.0), 1000)
    Chia = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    Deltachi = Chia
    Deltachi[-1] = 0
    s2 = k[..., np.newaxis] * Deltachi

    integrand = special.spherical_jn(
        l, s2) * derv_Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(a)
    g_isw_CMB = 8 * np.pi * \
        (1j**l) * integrate.simps(integrand, a)

    return g_isw_CMB


def G_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Sum of kSZ integral kernels"""

    # Lower sampling for slow ISW term
    ks_isw = np.logspace(conf.k_min, conf.k_max, conf.k_res//10)
    Gs_isw = G_ISW_ksz( ks_isw, ze, Omega_b, Omega_c, w, wa, Omega_K, h )
    G_int_ISW = interp1d(
        ks_isw, Gs_isw, kind="cubic", bounds_error=False, fill_value='extrapolate')

    return G_int_ISW(k) \
        + G_SW_ksz( k, ze, Omega_b, Omega_c, w, wa, Omega_K, h ) \
        + G_Dopp_ksz( k, ze, Omega_b, Omega_c, w, wa, Omega_K, h )


def G_ksz_localDopp(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """kSZ integral kernel including only local peculiar velocity"""
    G_s = G_localDopp_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
    return G_s


def G_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Sum of psz integral kernels"""
    G = G_SW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_Dopp_psz( k, ze, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_ISW_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)

    return G


def G_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h):
    """Sum of CMB integral kernels"""
    G = G_SW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_Dopp_CMB( k, l, Omega_b, Omega_c, w, wa, Omega_K, h )\
        + G_ISW_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h)

    return G



##########################################################
################                       ###################
################ TRANSFER FUNCTIONS    ###################
################                       ###################
##########################################################


def Chi_bin_boundaries(z_min, z_max, N) :
    """
    Get comoving distances (chi) of boundaries of N bins from z_min to z_max,
    equally spaced in comoving distance
    """
    Chi_min = Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(z_min))
    Chi_max = Chia_inter(conf.Omega_b, conf.Omega_c, conf.w, conf.wa, conf.Omega_K, conf.h)(az(z_max))
    Chi_boundaries = np.linspace(Chi_min, Chi_max, N+1)
    return Chi_boundaries


def Chi_bin_centers(z_min, z_max, N) :
    """
    Get comoving distances at center of of bins from Chi_bin_boundaries()
    """
    Chi_boundaries = Chi_bin_boundaries(z_min, z_max, N)
    # Get center of bins in comoving distance, convert to redshift
    Chis = ( Chi_boundaries[:-1] + Chi_boundaries[1:] ) / 2.0
    return Chis


def Z_bin(N) :
    """
    Get redshifts corresponding to bin centers from Chi_bin_centers
    """
    Chis = Chi_bin_centers(conf.z_min, conf.z_max, N)
    return zfromchi(Chis)


def Z_bin_samples(N_bins, Bin_num, N_samples_in_bin):
    """
    Get redshifts of samples in a "bin" between conf.z_min and conf.z_max,
    uniformly distributed in chi, and at bin centers (so excluding boundaries.)

    N = number of bins between z_min and z_max
    B = bin number to get samples in
    N_samples = number of samples in bin
    """
    # Get boundaries of bins
    Chi_boundaries = Chi_bin_boundaries(conf.z_min, conf.z_max, N_bins)
    Z_boundaries = zfromchi(Chi_boundaries)

    # Generate redshift samples inside bin
    Chi_samples = np.linspace(Chi_boundaries[Bin_num], Chi_boundaries[Bin_num + 1], N_samples_in_bin)

    # Translate this to redshift boundaries
    z_samples = zfromchi(Chi_samples)
    return z_samples


def Z_bin_samples_conf(Bin_num, config=conf):
    return Z_bin_samples(config.N_bins, Bin_num, config.n_samples)


def Get_Windowed_Transfer(Transfer, redshifts, redshift_weights, L, *args, **kwargs) :
    """
    Generic function to compute windowed transfer functions; computes:

      Transfer(k, L, z, args, kwargs)

    at each redshift in `redshifts`, then integrates between
    redshifts[0] and redshifts[-1] using `redshift_weights` as weighting.
    Weights don't have to be normalized (they are below).
    Cosmology parameters are those provided in kszpsz_config (for now).

    *args and **kwargs are passed through to transfer.
    For a single redshift, no integration is performed.
    ks are determined by config passed in kwargs, otherwise by kszpsz_config sampling.
    """
    if 'config' in kwargs :
        k = k_sampling(config=kwargs['config'])
    else:
        k = k_sampling()
    N_samples = len(redshifts)
    T_samples = np.zeros((N_samples, len(k), len(L)), dtype=np.complex64)


    if N_samples == 1:
        z = redshifts[0]
        return Transfer(k, L, z, *args, **kwargs)
    else :
        Window_norm = integrate.simps(redshift_weights, redshifts)

        for n in np.arange(N_samples) :
            z = redshifts[n]
            T_samples[n] = redshift_weights[n] \
              * Transfer(k, L, z, *args, **kwargs)

        Transfer_windowed = integrate.simps(T_samples, redshifts, axis=0) / Window_norm

        return Transfer_windowed


def Transfer_ksz_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function at a given redshift `ze`
    """
    transfer_ksz = np.zeros((len(k), len(L)), dtype=np.complex64)
    Ker = G_ksz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_ksz[:, l_id] = 0.0 + 1j * 0.0
        else:
            c = (4 * np.pi * (1j)**l) / (2 * l + 1)
            transfer_ksz[:, l_id] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    return transfer_ksz


def Transfer_ksz_bin(N, n, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function for a given binning scheme
    """
    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(n):
        Ker = G_ksz(k, z_samples[j], Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa,
                          Omega_K, h)(az(z_samples[j]))

        for l_id, l in enumerate(L):
            if l == 0:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = (4 * np.pi * (1j)**l) / (2 * l + 1)
                T_samples[:, l, j] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                    l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    transfer_ksz = (1 / n) * np.sum(T_samples, axis=-1)

    return transfer_ksz


def Transfer_ksz_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    kSZ transfer function averaged in a window
    """
    return Get_Windowed_Transfer(Transfer_ksz_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Transfer_ksz_localDopp_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function for a given redshift using only local doppler
    """
    transfer_ksz = np.zeros((len(k), len(L)), dtype=np.complex64)
    Ker = G_ksz_localDopp(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
    Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

    for l_id, l in enumerate(L):
        if l == 0:
            transfer_ksz[:, l_id] = 0.0 + 1j * 0.0
        else:
            c = (4 * np.pi * (1j)**l) / (2 * l + 1)
            transfer_ksz[:, l_id] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    return transfer_ksz


def Transfer_ksz_localDopp_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    kSZ transfer function, local doppler only, averaged in a window
    """
    return Get_Windowed_Transfer(Transfer_ksz_localDopp_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Transfer_ksz_bin_localDopp(N, n, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """
    KSZ transfer function for a given bin using only local doppler
    """
    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(0, n):
        Ker = G_ksz_localDopp(
            k, z_samples[j], Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa,
                          Omega_K, h)(az(z_samples[j]))

        for l_id, l in enumerate(L):
            if l == 0:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = (4 * np.pi * (1j)**l) / (2 * l + 1)
                T_samples[:, l, j] = c * (l * special.spherical_jn(l - 1, k * Chie) - (
                    l + 1) * special.spherical_jn(l + 1, k * Chie)) * Tk * Ker

    transfer_ksz = (1 / n) * np.sum(T_samples, axis=-1)

    return transfer_ksz


def Transfer_psz_redshift(k, L, ze, Omega_b, Omega_c, w, wa, Omega_K, h):
    """PSZ transfer function for a given redshift"""
    if ze == 0.0:
        transfer_psz = np.zeros((len(k), len(L)), dtype=np.complex64)
        Ker = G_psz(k, 0.0, Omega_b, Omega_c, w, wa, Omega_K, h)
        Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

        for l_id, l in enumerate(L):
            if l == 2:
                c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                    np.sqrt(special.factorial(l + 2) /
                            special.factorial(l - 2))
                transfer_psz[:, l_id] = c * (1. / 15.) * Tk * Ker
            else:
                transfer_psz[:, l_id] = 0.0 + 1j * 0.0

    else:
        transfer_psz = np.zeros((len(k), len(L)), dtype=np.complex64)
        Ker = G_psz(k, ze, Omega_b, Omega_c, w, wa, Omega_K, h)
        Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(az(ze))

        for l_id, l in enumerate(L):
            if l < 2:
                transfer_psz[:, l_id] = 0.0 + 1j * 0.0
            else:
                c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                    np.sqrt(special.factorial(l + 2) /
                            special.factorial(l - 2))
                transfer_psz[
                    :, l] = c * (special.spherical_jn(l, k * Chie) / ((k * Chie)**2)) * Tk * Ker

        return transfer_psz


def Transfer_psz_windowed(L, z_samples, z_sample_weights, config=conf):
    """
    pSZ transfer function averaged in a window
    """
    return Get_Windowed_Transfer(Transfer_psz_redshift, z_samples, z_sample_weights, L,
        config.Omega_b, config.Omega_c, config.w, config.wa, config.Omega_K, config.h)


def Transfer_psz_bin(N, B, k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """PSZ transfer function for a given bin"""
    n = 10

    T_samples = np.zeros((len(k), len(L),) + (n,), dtype=np.complex64)
    z_samples = Z_bin_samples(N, B, n)

    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for j in np.arange(0, n):

        Ker = G_psz(k, z_samples[j], Omega_b, Omega_c, w, wa, Omega_K, h)
        Chie = Chia_inter(Omega_b, Omega_c, w, wa,
                          Omega_K, h)(az(z_samples[j]))

        for l_id, l in enumerate(L):
            if l < 2:
                T_samples[:, 0, j] = 0.0 + 1j * 0.0
            else:
                c = -(5 * (1j)**l) * np.sqrt(3.0 / 8.0) * \
                    np.sqrt(special.factorial(l + 2) /
                            special.factorial(l - 2))
                T_samples[
                    :, l, j] = c * (special.spherical_jn(l, k * Chie) / ((k * Chie)**2)) * Tk * Ker

    transfer_psz = (1.0 / n) * np.sum(T_samples, axis=-1)

    return transfer_psz


def Transfer_CMB(k, L, Omega_b, Omega_c, w, wa, Omega_K, h):
    """CMB transfer function for large scales at z = 0"""
    transfer_CMB = np.zeros((len(k), len(L)), dtype=np.complex64)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)

    for l_id, l in enumerate(L):
        if l < 1:
            transfer_CMB[:, l_id] = 0.0 + 1j * 0.0
        else:
            Ker = G_CMB(k, l, Omega_b, Omega_c, w, wa, Omega_K, h)
            transfer_CMB[:, l_id] = Tk * Ker

    return transfer_CMB


def Transfer_E(k, L, Omega_b, Omega_c, w, wa, Omega_K, h, tau):
    """E transfer function for large scales at z = 0"""
    transfer_E = np.zeros((len(k), len(L)), dtype=np.complex64)
    Tk = T(k, Omega_b, Omega_c, w, wa, Omega_K, h)
    Chi_grid = np.linspace(0.0, chifromz(z_re(Omega_b, h, tau)), 40)
    Z_grid = zfromchi(Chi_grid)
    a_grid = az(Z_grid)
    sigma_ne = sigma_nez(Z_grid, Omega_b, h)
    etau = np.exp(-tau_grid(Chi_grid, Z_grid, Omega_b, h))

    Integrand = np.zeros((len(k), len(L), len(Chi_grid)), dtype=np.complex64)
    for i in np.arange(len(Chi_grid)):
        Integrand[:, :, i] = (-np.sqrt(6.0) / 10.0) * Transfer_psz_redshift(k, L, Z_grid[
            i], Omega_b, Omega_c, w, wa, Omega_K, h) * a_grid[i] * sigma_ne[i] * etau[i]

    transfer_E = integrate.simps(Integrand, Chi_grid, axis=-1)

    return transfer_E


###############################################################################################
################                                                            ###################
################ Transfer function and signal covariance for number counts  ###################
################                                                            ###################
###############################################################################################

###############################################################################
################                                            ###################
################ BIN CORRELATIONS AND CORRELATION MATRIX    ###################
################                                            ###################
###############################################################################

def CL_bins(T_list1, T_list2, k, L, As=conf.As, ns=conf.ns):
    """
    Compute correlation C_l's using transfer function pairs in T_list1 and T_list2
    assumes T[l_idx, k], not T[k,l].
    """
    CL = np.zeros((len(T_list1),len(L)))

    for i in np.arange(len(T_list1)):
        for l in L:
            T1 = T_list1[i]
            T2 = T_list2[i]

            I = (k**2)/(2*np.pi)**3 * Ppsi(k, As, ns) * np.conj(T1[:,l]) * T2[:,l]

            CL[i,l] = np.real(integrate.simps(I, k))

    return CL

def CL(T1, T2, k=None, L=None, P_psi=None, config=conf):
    """
    Compute correlation C_l's using transfer function pairs in T_list1 and T_list2
    assumes T[l_idx, k], not T[k,l].
    """
    # if not L :
    if not L.any():
        L = L_sampling(config)
    # if not k :
    if not k.any():
        k = k_sampling(config)
    # if not P_psi :
    if not P_psi.any():
        As=config.As
        ns=config.ns
        P_psi = Ppsi(k, As, ns)

    CL = np.zeros(len(L))

    for l in L:
        I = (k**2)/(2*np.pi)**3 * P_psi * np.conj(T1[l]) * T2[l]
        CL[l] = np.real(integrate.simps(I, k))

    return CL
