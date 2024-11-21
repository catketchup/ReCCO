import numpy as np
from math import pi
import my_remote_spectra as rs
import config as config

def Cos_theta_c(Z_c, z_e, chi_e, theta_e):
    "Capital Z_c for comoving position of the boundary at z-direction by default"
    chi_e = rs.chifromz(z_e)
    chi_edec = rs.chifromz(config.zdec) - rs.chifromz(config(z_e))
    return (Z_c -chi_e*np.cos(theta_e))/chi_edec

