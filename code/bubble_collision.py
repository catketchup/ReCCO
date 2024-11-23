import numpy as np
from math import pi
import my_remote_spectra as rs
import config as config


class BubbleCollision_Veff():
    def __init__(self, A, B, Z_c, Z_e, Omega_b, Omega_c, w, wa, Omega_K, h):
        self.A = A
        self.B = B
        self.Z_c = Z_c
        self.Z_e = Z_e
        self.Omega_b = Omega_b
        self.Omega_c = Omega_c
        self.w = w
        self.wa = wa
        self.Omega_K = Omega_K
        self.h = h
        self.chi_c = rs.chifromz(Z_c)
        self.chi_e = rs.chifromz(Z_e)
        self.chi_edec = rs.chifromz(config.zdec) - rs.chifromz(Z_e)
        self.r_H = 3.e5/(h*100)
        self.Dpsi_e = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+Z_e))
        self.Dpsi_dec = rs.Dpsi_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))
        self.Dv_e = rs.Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+Z_e))
        self.Dv_dec = rs.Dv_inter(Omega_b, Omega_c, w, wa, Omega_K, h)(1/(1+config.zdec))

    def Cos_theta_c(self, theta_e):
        return (self.chi_c -self.chi_e*np.cos(theta_e))/self.chi_edec

    def Approx_Veff_SW_radial(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_c = self.Cos_theta_c(theta_e)
        cos_theta_e = np.cos(theta_e)

        return  (2*self.Dpsi_dec-3/2)*3/2*cos_theta_e*(A/self.r_H*((self.chi_e*cos_theta_e-self.chi_c)*1/2*(1-cos_theta_c**2)+1/3*self.chi_edec*(1-cos_theta_c**3)) + B/(self.r_H**2)*((self.chi_e*cos_theta_e-self.chi_c)**2*1/2*(1-cos_theta_c**2) + (self.chi_e*cos_theta_e-self.chi_c)*self.chi_edec*2/3*(1-cos_theta_c**3) + 1/4*self.chi_edec**2*(1-cos_theta_c**4)))


    def Approx_Veff_localDopp_radial(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_e = np.cos(theta_e)
        step_array = np.ones_like(theta_e)
        step_array[np.where(cos_theta_e<(self.chi_c/self.chi_e))]=0

        return -self.Dv_e/(self.r_H)*(A + 2*B*(self.chi_e*cos_theta_e - self.chi_c))*cos_theta_e*step_array

    def Approx_Veff_decDopp_radial(self, theta_e):
        A = self.A
        B = self.B
        cos_theta_c = self.Cos_theta_c(theta_e)
        cos_theta_e = np.cos(theta_e)
        return 3/2*cos_theta_e*self.Dv_dec*(1/3*A/self.r_H*(1 - cos_theta_c**3) + 2/3*B/(self.r_H**2)*self.chi_edec*(self.chi_e*cos_theta_e-self.chi_c)* (1-cos_theta_c**3) + 1/2*B/(self.r_H**2)*self.chi_edec*(1-cos_theta_c**4))

    # def Approx_Veff_ISW_radial(self, theta_e):
