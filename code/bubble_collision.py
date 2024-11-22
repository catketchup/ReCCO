import numpy as np
from math import pi
import my_remote_spectra as rs
import config as config


class BubbleCollision_Veff():
    def __init__(self, Z_e, Omega_b, Omega_c, w, wa, Omega_K, h):
        self.Z_e = Z_e
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.w = w
        self.wa = wa
        self.Omega_K = Omega_K
        self.h = h
        self.chi_e = rs.chifromz(Z_e)
        self.chi_edec = rs.chifromz(config.zdec) - rs.chifromz(Z_e)
        self.r_H = 3.e5/(h*100)
        self.Dpsi_dec = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))

    def Cos_theta_c(self, Z_c, theta_e):
        return (Z_c -self.chi_e*np.cos(theta_e))/self.chi_edec

    def Approx_Veff_SW_radial(self, A, B, Z_c, theta_e):
        cos_theta_c = self.Cos_theta_c(Z_c, theta_e)
        cos_theta_e = np.cos(theta_e)
        # return  (2*self.Dpsi_dec-3/2)*3/2*cos_theta_e*(A/self.r_H*((self.chi_e*cos_theta_e-Z_c)*1/2*(cos_theta_c**2-1)+1/3*self.chi_edec*(cos_theta_c**3-1)))
        return  (2*self.Dpsi_dec-3/2)*3/2*cos_theta_e*(A/self.r_H*((self.chi_e*cos_theta_e-Z_c)*1/2*(1-cos_theta_c**2)+1/3*self.chi_edec*(1-cos_theta_c**3)))
